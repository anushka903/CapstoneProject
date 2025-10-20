"""
derive_performance_index.py
---------------------------------------
step 3b (updated for season-weighted feature importances):
construct and test performance index formulas using improved
feature weights derived from per-season modeling.

this script:
- loads the cleaned all-seasons dataset
- loads averaged feature importances (season-adjusted)
- computes multiple versions of the performance index (PI)
- tests how well each correlates with actual salary

output:
- performance_index_results.csv
- player_performance_index.csv
"""

import pandas as pd
import numpy as np
import os

# === file paths ===
INPUT_FILE = "All_Seasons_Clean.csv"
IMPORTANCE_FILE = os.path.join("model_outputs", "feature_importance_averaged.csv")
OUTPUT_DIR = "performance_index_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("\nstarting step 3b (updated): derive performance index using season-weighted importances...")
    print("-------------------------------------------------------------------------------------------")

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

    # 2. load improved feature importances
    print("\nloading season-weighted feature importance file...")
    if not os.path.exists(IMPORTANCE_FILE):
        raise FileNotFoundError(f"{IMPORTANCE_FILE} not found — run derive_feature_importance.py first")
    importance_df = pd.read_csv(IMPORTANCE_FILE)
    print(f"loaded averaged importances for {len(importance_df)} features.")

    # normalize weights again for safety
    importance_df["Weight"] = importance_df["Avg_Normalized_Importance"] / importance_df["Avg_Normalized_Importance"].sum()

    # select top 15 most influential features
    top_features = importance_df.head(15)["Feature"].tolist()
    print(f"top {len(top_features)} features selected for performance index:")
    print(top_features)

    # prepare feature matrix
    print("\nextracting top features for PI computation...")
    X = df[top_features].apply(pd.to_numeric, errors="coerce").fillna(0)
    print(f"feature matrix shape: {X.shape}")

    # 3. compute different PI formulas
    print("\nconstructing performance index formulas...")

    # (a) season-weighted model PI
    print("building PI_model_weighted (season-aware)...")
    weights = importance_df.set_index("Feature").loc[top_features, "Weight"].values
    df["PI_model_weighted"] = np.dot(X, weights)

    # (b) equal-weighted PI
    print("building PI_equal_weighted...")
    df["PI_equal_weighted"] = X.mean(axis=1)

    # (c) correlation-weighted PI (direct feature-salary correlation)
    print("building PI_corr_weighted...")
    corr = X.corrwith(salary)
    corr = corr / corr.sum()
    df["PI_corr_weighted"] = np.dot(X, corr.values)

    print("\nall performance index versions created successfully.")

    # 4. evaluate correlations
    print("\nevaluating correlations with salary...")
    correlations = {
        "PI_model_weighted": df["PI_model_weighted"].corr(salary),
        "PI_equal_weighted": df["PI_equal_weighted"].corr(salary),
        "PI_corr_weighted": df["PI_corr_weighted"].corr(salary)
    }

    results = pd.DataFrame(list(correlations.items()), columns=["Formula", "Correlation_with_Salary"])
    results = results.sort_values("Correlation_with_Salary", ascending=False)
    print("\ncorrelation results:")
    print(results)

    # 5. save outputs
    print("\nsaving output files to 'performance_index_outputs/' directory...")
    results.to_csv(os.path.join(OUTPUT_DIR, "performance_index_results.csv"), index=False)
    df.to_csv(os.path.join(OUTPUT_DIR, "player_performance_index.csv"), index=False)
    print("files saved successfully:")
    print(" - performance_index_results.csv")
    print(" - player_performance_index.csv")

    best_formula = results.iloc[0]["Formula"]
    print(f"\nmost effective performance index based on correlation: {best_formula}")
    print("\nstep 3b (season-weighted) complete. performance index derivation outputs ready.")


if __name__ == "__main__":
    main()
