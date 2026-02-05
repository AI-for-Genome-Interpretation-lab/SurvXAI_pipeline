from sklearn.model_selection import KFold, StratifiedKFold
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored, integrated_brier_score, concordance_index_ipcw
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from tqdm import tqdm
from survshap import SurvivalModelExplainer, ModelSurvSHAP
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    # Preserve DataFrame structure for downstream components (e.g., SHAP)
    feature_names = X_train.columns if hasattr(X_train, "columns") else None
    train_index = X_train.index if hasattr(X_train, "index") else None
    test_index = X_test.index if hasattr(X_test, "index") else None

    X_train_scaled_arr = scaler.fit_transform(X_train)
    X_test_scaled_arr = scaler.transform(X_test)

    if feature_names is not None and train_index is not None and test_index is not None:
        X_train_scaled = pd.DataFrame(X_train_scaled_arr, columns=feature_names, index=train_index)
        X_test_scaled = pd.DataFrame(X_test_scaled_arr, columns=feature_names, index=test_index)
    else:
        X_train_scaled = X_train_scaled_arr
        X_test_scaled = X_test_scaled_arr

    return X_train_scaled, X_test_scaled


def cross_evaluate_survival_model_kf(
    model,
    X,
    y_time,
    y_event,
    cv=5,
    do_scale_features=True


):
    """
    Perform K-fold cross-validation for survival models.

    Parameters:
        model: a scikit-survival compatible model (must implement fit and predict)
        X: features (DataFrame)
        y_time: survival time (Series)
        y_event: event indicator (Series)
        cv: number of folds
        do_scale_features: whether to scale features
        imputation_type: how to impute missing values within each fold. One of
            - "out_of_scale": fill NaNs with -999999
            - "median": fill NaNs with the training-fold median per column
            - None: do not impute inside CV

    Returns:
        mean_c_index: mean C-index over folds
        c_indexes: list of C-index per fold
        mean_ibs: mean integrated Brier score over folds
        ibs_scores: list of integrated Brier score per fold

    """

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    c_indexes = []
    ibs_scores = []

    X = X.copy()

    # Build structured array for survival analysis
    y_struct = Surv.from_arrays(event=y_event.astype(bool), time=y_time)
    global_max_time = y_time.max()

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train = y_struct[train_idx]
        y_test = y_struct[test_idx]


        if do_scale_features:
            X_train, X_test = scale_features(X_train, X_test)


        m = clone(model)
        m.fit(X_train, y_train)

        # --- C-index ---
        risk_scores = m.predict(X_test)
        c_index, _, _, _, _ = concordance_index_censored(
            y_test["event"], y_test["time"], risk_scores
        )
        c_indexes.append(c_index)

        # --- IBS ---
        try:
            surv_funcs = m.predict_survival_function(X_test)

            # define min and max time points of test dataset and max time point of training dataset
            min_test_time = y_test["time"].min()
            max_test_time = y_test["time"].max()
            train_max_time = y_train["time"].max()

            # Ensure evaluation times are strictly smaller than the largest observed time point
            # across train/test to satisfy integrated_brier_score requirements.
            eps = np.finfo(float).eps
            strict_upper_bound = min(train_max_time, max_test_time) - eps

            # Guard against degenerate ranges
            if not np.isfinite(strict_upper_bound) or strict_upper_bound <= min_test_time:
                print("Skipping fold: invalid time range for IBS.")
                ibs_scores.append(np.nan)
                continue

            # create time grid without including the strict upper bound (endpoint=False)
            times_eval = np.linspace(min_test_time, strict_upper_bound, 20, endpoint=False)

            # filtering: ensure enough samples at risk in the test fold
            min_samples_at_risk = 5
            times_eval = np.array([
                t for t in times_eval
                if (y_test["time"] >= t).sum() >= min_samples_at_risk and t < strict_upper_bound
            ])

            if len(times_eval) < 2:
                print("Skipping fold: not enough valid time points for IBS.")
                ibs_scores.append(np.nan)
                continue

            # surv probs calculating
            surv_probs = np.array([fn(times_eval) for fn in surv_funcs])
            ibs = integrated_brier_score(y_train, y_test, surv_probs, times=times_eval)
            ibs_scores.append(ibs)

        except Exception as e:
            #print(f"[Fold {fold_idx}] IBS calculation failed: {e}")
            ibs_scores.append(np.nan)



    return np.nanmean(c_indexes), c_indexes, np.nanmean(ibs_scores), ibs_scores




def cross_validated_survival_and_shap(
    model,
    X, y_time, y_event,
    cv=5,
    do_scale_features = True,
    num_time_points=200,
    shap_method="treeshap",
    return_models=False

):
    """
    Combined cross-validation to compute survival functions and SHAP values.

    Parameters:
    -----------
    return_models : bool, default False
        If True, also return the trained models and scalers from each fold

    Returns:
        - survival_df: ['sample_index', 'time', 'survival_prob']
        - shap_df: ['feature', 'shap_value', 'sample_index', 'fold']
        - models_data: dict (only if return_models=True) containing:
            - 'models': list of trained models from each fold
            - 'scalers': list of fitted scalers from each fold
            - 'fold_indices': list of (train_idx, test_idx) for each fold
    """


    y_struct = Surv.from_arrays(event=y_event.astype(bool), time=y_time)
    global_timeline = np.linspace(0, y_time.max(), num=num_time_points)
    global_timeline = np.unique(global_timeline[global_timeline > 0])

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    survival_dict = defaultdict(list)
    shap_dict = defaultdict(list)
    
    # Store models and scalers if requested
    models_data = {'models': [], 'scalers': [], 'fold_indices': []} if return_models else None

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train_struct = y_struct[train_idx]
        # Optionally scale features during CV (data already encoded/imputed in data.py)
        scaler = None
        if do_scale_features:
            # Create scaler for this fold
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Convert back to DataFrames to preserve column names
            X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

        m = clone(model)
        m.fit(X_train, y_train_struct)
        
        # Store model and scaler if requested
        if return_models:
            models_data['models'].append(m)
            models_data['scalers'].append(scaler)
            models_data['fold_indices'].append((train_idx, test_idx))

        # Predict survival functions
        surv_funcs = m.predict_survival_function(X_test)
        for i, idx in enumerate(test_idx):
            fn = surv_funcs[i]
            interp_probs = np.interp(global_timeline, fn.x, fn.y)
            survival_dict[idx].append(interp_probs)

        # SHAP values
        try:
            explainer = SurvivalModelExplainer(model=m, data=X_train, y=y_train_struct)
            model_survshap = ModelSurvSHAP(calculation_method=shap_method)
            model_survshap.fit(explainer=explainer, new_observations=X_test, check_additivity=False)

            for i, indiv in enumerate(model_survshap.individual_explanations):
                df = indiv.result.copy()
                sample_idx = test_idx[i]

                for _, row in df.iterrows():
                    var = row['variable_name']
                    for col in df.columns:
                        if col.startswith('t = '):
                            t_val = float(col.split('=')[1].strip())
                            shap_score = row[col]
                            shap_dict[(sample_idx, var, t_val)].append(shap_score)
        except Exception as e:
            print(f"[Fold {fold_idx}] SHAP error: {e}")

    # Aggregate survival functions
    final_surv_records = []
    for idx, pred_list in survival_dict.items():
        avg_probs = np.mean(pred_list, axis=0)
        for t, p in zip(global_timeline, avg_probs):
            final_surv_records.append({"sample_index": idx, "time": t, "survival_prob": p})

    # Aggregate SHAP values
    shap_records = []
    for (sample_idx, var, t), scores in shap_dict.items():
        shap_records.append({
            "sample_index": sample_idx,
            "variable_name": var,
            "time": t,
            "shap_value": np.mean(scores)
        })

    # Defer creating DataFrames to the end (final return block)

    survival_df = pd.DataFrame(final_surv_records)
    shap_df = pd.DataFrame(shap_records)
    
    if return_models:
        return survival_df, shap_df, models_data
    else:
        return survival_df, shap_df


