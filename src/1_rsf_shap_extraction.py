import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from data import prepare_metabric
from evaluation import cross_validated_survival_and_shap
from survivaltime import extract_median_shap_for_each_instance, compute_median_survival_time
import os



def main():


    #----------Data preprocessing for METABRIC---------------------
    X, y_time, y_event = prepare_metabric(imputation_type= "out_of_scale")

    y_time = pd.Series(y_time, index=X.index)
    y_event = pd.Series(y_event, index=X.index)

    # --------Selecting the first 100 samples to test--------------
    # X = X[:100]
    # y_time = y_time[:100]
    # y_event = y_event[:100]

    #----------Generation for predictions and SHAP values----------
    model = RandomSurvivalForest(n_estimators=100, min_samples_split=15)
    df_surv, df_shap = cross_validated_survival_and_shap(
        model,
        X, y_time, y_event,
        cv=5,
        do_scale_features=True,
        num_time_points=200,
        shap_method="treeshap",
        return_models=False
    )

    median_df = compute_median_survival_time(df_surv)
    median_shap_df = extract_median_shap_for_each_instance(median_df, df_shap, X, y_event, y_time)

    os.makedirs("outputs", exist_ok=True)

    #-----------Output and store the results------------------------
    median_shap_df.to_csv("outputs/metabric_sample.csv", index=False)
    print("OUTPUTS: metabric_sample.csv")



if __name__ == '__main__':
    main()