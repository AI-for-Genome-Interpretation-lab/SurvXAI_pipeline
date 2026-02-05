import pandas as pd
import numpy as np

def compute_median_survival_time(survival_df):
    """
    Computes median survival time for each sample.
    Median survival time is defined as the time when survival_prob = 0.5
    Uses interpolation to find the exact time point.
    """
    median_times = []

    for idx, group in survival_df.groupby("sample_index"):
        group_sorted = group.sort_values("time")
        times = group_sorted["time"].values
        probs = group_sorted["survival_prob"].values
        
        # Check if survival probability ever drops to 0.5 or below
        if probs.min() <= 0.5:
            # Find the exact time when survival_prob = 0.5 using interpolation
            try:
                # Use numpy interpolation to find the exact time
                median_time = np.interp(0.5, probs[::-1], times[::-1])
            except:
                # Fallback to discrete approach if interpolation fails
                below_50 = group_sorted[group_sorted["survival_prob"] <= 0.5]
                if not below_50.empty:
                    median_time = below_50.iloc[0]["time"]
                else:
                    # Find closest to 0.5
                    diffs = np.abs(probs - 0.5)
                    idx_min = np.argmin(diffs)
                    median_time = times[idx_min]
        else:
            # If survival never drops to 0.5, find the time closest to 0.5
            diffs = np.abs(probs - 0.5)
            idx_min = np.argmin(diffs)
            median_time = times[idx_min]

        median_times.append({"sample_index": idx, "median_survival_time": median_time})
    median_df = pd.DataFrame(median_times)

    return median_df

def extract_median_shap_for_each_instance(median_df, shap_df, X, y_event, y_time):
    """
    Combine SHAP values at median survival time with original variable values and survival info.
    """
    results = []

    for _, row in median_df.iterrows():
        sample_idx = row["sample_index"]
        median_time = row["median_survival_time"]

        sample_shap = shap_df[shap_df["sample_index"] == sample_idx]
        if sample_shap.empty:
            continue

        for enc_var in sample_shap["variable_name"].unique():
            var_block = sample_shap[sample_shap["variable_name"] == enc_var].sort_values("time")
            times = var_block["time"].values
            shap_values = var_block["shap_value"].values
            if len(times) == 0:
                continue

            # Check if median_time is valid for interpolation
            if pd.isna(median_time) or median_time <= 0:
                continue
            try:
                shap_at_med = np.interp(median_time, times, shap_values)
            except Exception:
                continue

            # Determine feature value to report
            # Prefer original variable/category mapping if present in SHAP output
            variable_value = None
            if {"orig_variable", "category"}.issubset(var_block.columns):
                orig_var = var_block["orig_variable"].iloc[0]
                category = var_block["category"].iloc[0]
                try:
                    if pd.isna(category) or category is None:
                        variable_value = X.loc[sample_idx, orig_var]
                    else:
                        xval = str(X.loc[sample_idx, orig_var]).strip().upper()
                        variable_value = int(xval == str(category).strip().upper())
                except Exception:
                    pass
            # Fallback: use enc_var directly if available in X
            if variable_value is None:
                try:
                    variable_value = X.loc[sample_idx, enc_var]
                except Exception:
                    continue

            results.append({
                "sample_index": sample_idx,
                "variable_name": enc_var,
                "variable_value": variable_value,
                "y_event": y_event.loc[sample_idx],
                "y_time": y_time.loc[sample_idx],
                "median_time": median_time,
                "shap_value": shap_at_med
            })

    return pd.DataFrame(results)



