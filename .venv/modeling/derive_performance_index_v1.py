"""
derive_performance_index.py
---------------------------------------
step 3b of phase 4: test and construct potential performance index formulas.

this script uses feature importance data (from random forest + xgboost)
to build different versions of a performance index (pi). each version
is tested by how strongly it correlates with actual player salaries.

output:
- performance_index_results.csv (shows correlation of each pi with salary)
- player_performance_index.csv (final table with best pi per player)
"""

import pandas as pd
import numpy as np
import os

# === file paths ===
INPUT_FILE = "All_Seasons_Clean.csv"
IMPORTANCE_FILE = os.path.join("model_outputs", "feature_importance_combined.csv")
OUTPUT_DIR = "performance_index_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("\nstarting step 3b: derive and test performance index formulas...")
    print("--------------------------------------------------------------")

    # 1. load cleaned dataset
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} not found — run load_all_seasons.py first")
    print(f"loading dataset from {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)
    print(f"dataset loaded. shape: {df.shape}")

    # identify salary column
    print("\nchecking for salary column...")
    if "SALARY" in df.columns:
        salary = df["SALARY"]
        print("using 'SALARY' as target variable.")
    elif "Salary_M" in df.columns:
        salary = df["Salary_M"]
        print("using 'Salary_M' as target variable.")
    else:
        raise ValueError("no salary column found in dataset")

    # 2. load feature importances
    print("\nloading feature importance results from model_outputs...")
    if not os.path.exists(IMPORTANCE_FILE):
        raise FileNotFoundError(f"{IMPORTANCE_FILE} not found — run derive_feature_importance.py first")
    importance_df = pd.read_csv(IMPORTANCE_FILE)
    print(f"feature importance file loaded. total features ranked: {len(importance_df)}")

    # normalize importances so they sum to 1
    print("normalizing importance values for weighting...")
    importance_df["Weight"] = importance_df["Avg_Importance"] / importance_df["Avg_Importance"].sum()

    # select top 15 features to build performance index
    top_features = importance_df.head(15)["Feature"].tolist()
    print(f"selected top {len(top_features)} features for performance index calculation:")
    print(top_features)

    # subset the main dataset to include only these top features
    print("\nextracting feature columns for pi calculation...")
    X = df[top_features].apply(pd.to_numeric, errors="coerce").fillna(0)
    print(f"feature matrix for pi shape: {X.shape}")

    # 3. build candidate performance index formulas
    print("\nconstructing candidate performance index formulas...")

    # formula 1: model-weighted sum (using combined feature importances)
    print("building pi_model_weighted...")
    df["PI_model_weighted"] = np.dot(X, importance_df.set_index("Feature").loc[top_features, "Weight"].values)

    # formula 2: equal-weighted average
    print("building pi_equal_weighted...")
    df["PI_equal_weighted"] = X.mean(axis=1)

    # formula 3: correlation-weighted sum (based on direct salary correlation)
    print("building pi_corr_weighted...")
    corr = X.corrwith(salary)
    corr = corr / corr.sum()
    df["PI_corr_weighted"] = np.dot(X, corr.values)

    print("\nall performance index versions created successfully.")

    # 4. evaluate correlation of each pi with actual salary
    print("\nevaluating how strongly each pi correlates with salary...")
    correlations = {
        "PI_model_weighted": df["PI_model_weighted"].corr(salary),
        "PI_equal_weighted": df["PI_equal_weighted"].corr(salary),
        "PI_corr_weighted": df["PI_corr_weighted"].corr(salary)
    }
    results = pd.DataFrame(list(correlations.items()), columns=["Formula", "Correlation_with_Salary"])
    results = results.sort_values("Correlation_with_Salary", ascending=False)
    print("\ncorrelation results:")
    print(results)

    # 5. save results to files
    print("\nsaving output files to 'performance_index_outputs/' directory...")
    results.to_csv(os.path.join(OUTPUT_DIR, "performance_index_results.csv"), index=False)
    df.to_csv(os.path.join(OUTPUT_DIR, "player_performance_index.csv"), index=False)
    print("files saved successfully:")
    print(" - performance_index_results.csv")
    print(" - player_performance_index.csv")

    # print summary of best formula
    best_formula = results.iloc[0]["Formula"]
    print(f"\nmost effective performance index based on correlation: {best_formula}")
    print("\nstep 3b complete. performance index derivation outputs ready.")


if __name__ == "__main__":
    main()
