#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder




# Drop columns function
def drop_columns_with_missing_data(df, threshold=0.3):
    missing_fraction = df.isnull().mean()
    cols_to_drop = missing_fraction[missing_fraction > threshold].index
    df.drop(columns=cols_to_drop, inplace=True)


def encode_and_impute(X: pd.DataFrame, imputation_type: str = "median") -> pd.DataFrame:
    """
    One-hot encode categorical columns and impute numeric columns.

    - Categorical: OneHotEncoder(handle_unknown="ignore") to avoid errors and keep stable columns.
    - Imputation:
        - "median": fill NaNs with column medians; drop columns that are all-NaN.
        - "out_of_scale": fill NaNs with -999999 sentinel.
    """
    X = X.copy()

    # One-hot encode categoricals
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if len(cat_cols) > 0:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        arr = enc.fit_transform(X[cat_cols])
        enc_df = pd.DataFrame(arr, columns=enc.get_feature_names_out(cat_cols), index=X.index)
        X = pd.concat([X.drop(columns=cat_cols), enc_df], axis=1)

    # Ensure numeric/boolean are floats
    num_cols = X.select_dtypes(include=[np.number, bool]).columns.tolist()
    if len(num_cols) > 0:
        X[num_cols] = X[num_cols].astype(float)

    # Impute
    if imputation_type == "median":
        if len(num_cols) > 0:
            med = X[num_cols].median()
            all_nan_cols = med.index[med.isna()].tolist()
            if all_nan_cols:
                X.drop(columns=all_nan_cols, inplace=True)
                num_cols = [c for c in num_cols if c not in set(all_nan_cols)]
            if num_cols:
                med = X[num_cols].median()
                X[num_cols] = X[num_cols].fillna(med)
    elif imputation_type == "out_of_scale":
        if len(num_cols) > 0:
            X[num_cols] = X[num_cols].fillna(-999999)
    else:
        raise ValueError(f"Unknown imputation_type: {imputation_type}")

    return X



def prepare_metabric(imputation_type: str = "median"):
    df = pd.read_csv("data/brca_metabric_clinical_data.tsv", sep="\t")

    # drop y value with na
    y_time = df['Overall Survival (Months)'].astype(float)
    y_event = df["Overall Survival Status"].astype(str)
    vital_status = df["Patient's Vital Status"].astype(str)

    mask = ~(y_time.isna() | y_event.isna())
    df = df.loc[mask].reset_index(drop=True)
    y_time = y_time.loc[mask].reset_index(drop=True)
    y_event = y_event.loc[mask].reset_index(drop=True)
    vital_status = vital_status.loc[mask].reset_index(drop=True)

    survival_status = df["Overall Survival Status"].astype(str).str.split(":", n=1).str[0].astype(int)

    y_event = ((survival_status == 1) & (vital_status == "Died of Disease")).astype(int)

    # robust YES/NO mapping
    df["Chemotherapy"]    = df["Chemotherapy"].astype(str).str.strip().str.upper().map({"YES": 1, "NO": 0})
    df["Hormone Therapy"] = df["Hormone Therapy"].astype(str).str.strip().str.upper().map({"YES": 1, "NO": 0})
    df["Radio Therapy"]   = df["Radio Therapy"].astype(str).str.strip().str.upper().map({"YES": 1, "NO": 0})
    # map receptor statuses (fix typo and case): Positive -> 1, Negative -> 0
    df["ER Status"]   = df["ER Status"].astype(str).str.strip().str.upper().map({"POSITIVE": 1, "NEGATIVE": 0})
    df["PR Status"]   = df["PR Status"].astype(str).str.strip().str.upper().map({"POSITIVE": 1, "NEGATIVE": 0})
    df["HER2 Status"] = df["HER2 Status"].astype(str).str.strip().str.upper().map({"POSITIVE": 1, "NEGATIVE": 0})
    df["Type of Breast Surgery"] = df["Type of Breast Surgery"].astype(str).str.strip().str.upper().map({"MASTECTOMY": 1, "BREAST CONSERVING": 0})
    df["Inferred Menopausal State"] = df["Inferred Menopausal State"].astype(str).str.strip().str.upper().map({"POST": 1, "PRE": 0})
    df["Primary Tumor Laterality"] = df["Primary Tumor Laterality"].astype(str).str.strip().str.upper().map({"RIGHT": 1, "LEFT": 0})

    # features
    num_cols = [
        "Age at Diagnosis", "Tumor Size", "Lymph nodes examined positive",
        "Nottingham prognostic index", "Mutation Count", "TMB (nonsynonymous)",
        "Tumor Stage", "Neoplasm Histologic Grade","Chemotherapy", "Hormone Therapy", "Radio Therapy",
        "ER Status","PR Status", "HER2 Status",
        "Type of Breast Surgery", "Inferred Menopausal State", "Primary Tumor Laterality"
    ]
    cat_cols = [
        "Pam50 + Claudin-low subtype", "3-Gene classifier subtype", "Integrative Cluster"
    ]

    missing = [c for c in num_cols + cat_cols if c not in df.columns]
    assert not missing, f"missing：{missing}"

    # normalize string
    def norm_cat(s):
        return s.astype(str).str.strip().str.upper().replace({"NAN": "UNKNOWN", "NA": "UNKNOWN", "NONE": "UNKNOWN"})

    for c in cat_cols:
        df[c] = norm_cat(df[c])

    # Binary mapping for specific categorical variables
    # Type of Breast Surgery: MASTECTOMY -> 1, BREAST CONSERVING -> 0


    # transfer num_cols to num with coerce (incorrect format->NA)
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    X = df[num_cols + cat_cols]

    # End-of-pipeline: encode + impute
    X = encode_and_impute(X, imputation_type=imputation_type).reset_index(drop=True)
    
    # Load expected features from the saved model to ensure compatibility
    import joblib
    try:
        model_data = joblib.load("../outputs/metabric_survival_model.joblib")
        expected_features = model_data['feature_names']
        
        # Add missing features with zeros
        for feature in expected_features:
            if feature not in X.columns:
                X[feature] = 0.0
        
        # Reorder columns to match expected order
        X = X[expected_features]
        
    except FileNotFoundError:
        print("Warning: Could not load model to align features. Using current features.")
    
    return X, y_time, y_event

def prepare_sample4(imputation_type: str = "median"):
    df = pd.read_csv("../data/sample4_Pam50.csv", sep=",")

    # drop y value with na
    y_time = df['Overall Survival (Months)'].astype(float)
    y_event = df["Overall Survival Status"].astype(str)

    mask = ~(y_time.isna() | y_event.isna())
    df = df.loc[mask].reset_index(drop=True)
    y_time = y_time.loc[mask].reset_index(drop=True)
    y_event = y_event.loc[mask].reset_index(drop=True)
    y_event = df["Overall Survival Status"].astype(str).str.split(":", n=1).str[0].astype(int)


    # robust YES/NO mapping
    df["Chemotherapy"]    = df["Chemotherapy"].astype(str).str.strip().str.upper().map({"YES": 1, "NO": 0})
    df["Hormone Therapy"] = df["Hormone Therapy"].astype(str).str.strip().str.upper().map({"YES": 1, "NO": 0})
    df["Radio Therapy"]   = df["Radio Therapy"].astype(str).str.strip().str.upper().map({"YES": 1, "NO": 0})
    # map receptor statuses (fix typo and case): Positive -> 1, Negative -> 0
    df["ER Status"]   = df["ER Status"].astype(str).str.strip().str.upper().map({"POSITIVE": 1, "NEGATIVE": 0})
    df["PR Status"]   = df["PR Status"].astype(str).str.strip().str.upper().map({"POSITIVE": 1, "NEGATIVE": 0})
    df["HER2 Status"] = df["HER2 Status"].astype(str).str.strip().str.upper().map({"POSITIVE": 1, "NEGATIVE": 0})
    df["Type of Breast Surgery"] = df["Type of Breast Surgery"].astype(str).str.strip().str.upper().map({"MASTECTOMY": 1, "BREAST CONSERVING": 0})
    df["Inferred Menopausal State"] = df["Inferred Menopausal State"].astype(str).str.strip().str.upper().map({"POST": 1, "PRE": 0})
    df["Primary Tumor Laterality"] = df["Primary Tumor Laterality"].astype(str).str.strip().str.upper().map({"RIGHT": 1, "LEFT": 0})

    # features
    num_cols = [
        "Age at Diagnosis", "Tumor Size", "Lymph nodes examined positive",
        "Nottingham prognostic index", "Mutation Count", "TMB (nonsynonymous)",
        "Tumor Stage", "Neoplasm Histologic Grade","Chemotherapy", "Hormone Therapy", "Radio Therapy",
        "ER Status","PR Status", "HER2 Status",
        "Type of Breast Surgery", "Inferred Menopausal State", "Primary Tumor Laterality"
    ]
    cat_cols = [
        "Pam50 + Claudin-low subtype", "3-Gene classifier subtype", "Integrative Cluster"
    ]

    missing = [c for c in num_cols + cat_cols if c not in df.columns]
    assert not missing, f"missing：{missing}"

    # normalize string
    def norm_cat(s):
        return s.astype(str).str.strip().str.upper().replace({"NAN": "UNKNOWN", "NA": "UNKNOWN", "NONE": "UNKNOWN"})

    for c in cat_cols:
        df[c] = norm_cat(df[c])

    # Binary mapping for specific categorical variables
    # Type of Breast Surgery: MASTECTOMY -> 1, BREAST CONSERVING -> 0


    # transfer num_cols to num with coerce (incorrect format->NA)
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    X = df[num_cols + cat_cols]

    # End-of-pipeline: encode + impute
    X = encode_and_impute(X, imputation_type=imputation_type).reset_index(drop=True)
    
    # Load expected features from the saved model to ensure compatibility
    import joblib
    try:
        model_data = joblib.load("../outputs/metabric_survival_model.joblib")
        expected_features = model_data['feature_names']
        
        # Add missing features with zeros
        for feature in expected_features:
            if feature not in X.columns:
                X[feature] = 0.0
        
        # Reorder columns to match expected order
        X = X[expected_features]
        
    except FileNotFoundError:
        print("Warning: Could not load model to align features. Using current features.")
    
    return X, y_time, y_event



