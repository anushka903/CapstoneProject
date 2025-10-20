"""
derive_feature_importance.py
---------------------------------------
step 3a of phase 4: construct data for performance index derivation.

this script trains random forest and xgboost models on all seasons combined
to determine which player statistics most strongly influence salary.

output:
- feature_importance_randomforest.csv
- feature_importance_xgboost.csv
- feature_importance_combined.csv
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# === input and output settings ===
INPUT_FILE = "All_Seasons_Clean.csv"
OUTPUT_DIR = "model_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("\nstarting step 3a: derive feature importance for performance index...")
    print("---------------------------------------------------------------")

    # 1. load dataset
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} not found — run load_all_seasons.py first")

    print(f"loading dataset from {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)
    print(f"dataset loaded successfully. shape: {df.shape}")
    print("preview of columns:", list(df.columns)[:10])

    # 2. select numeric features only (exclude name, team, and non-numeric stuff)
    exclude_cols = ["PLAYER", "TEAM", "Season", "SALARY", "Salary_M", "Player_Comp_Share"]
    print("\nselecting numeric columns and dropping irrelevant columns...")
    features = df.select_dtypes(include=[np.number]).drop(columns=[c for c in exclude_cols if c in df.columns],
                                                          errors="ignore")
    print(f"selected {len(features.columns)} numeric features for modeling.")

    # pick the correct salary column (depends on dataset naming)
    print("\nidentifying salary column...")
    if "SALARY" in df.columns:
        target = df["SALARY"]
        print("using column 'SALARY' as target variable.")
    elif "Salary_M" in df.columns:
        target = df["Salary_M"]
        print("using column 'Salary_M' as target variable.")
    else:
        raise ValueError("no salary column found in dataset")

    # handle missing numeric values by filling with 0
    print("\nfilling missing numeric values with 0...")
    X = features.fillna(0)
    y = target.fillna(0)
    print(f"final feature matrix shape: {X.shape}")
    print(f"target vector length: {len(y)}")

    # 3. split into train/test (use 80/20 split)
    print("\nsplitting data into train and test sets (80/20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"train set: {X_train.shape}, test set: {X_test.shape}")

    # 4. random forest model to estimate feature importance
    print("\ntraining random forest regressor (200 trees)...")
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    print("random forest training complete.")
    rf_importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance_RF": rf.feature_importances_
    }).sort_values("Importance_RF", ascending=False)
    print("top 5 random forest features:")
    print(rf_importances.head(5))

    # 5. xgboost model to estimate feature importance
    print("\ntraining xgboost regressor (300 trees, learning_rate=0.1)...")
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.1, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    print("xgboost training complete.")
    xgb_importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance_XGB": xgb.feature_importances_
    }).sort_values("Importance_XGB", ascending=False)
    print("top 5 xgboost features:")
    print(xgb_importances.head(5))

    # 6. combine results from both models and average their importances
    print("\ncombining and averaging feature importances from both models...")
    combined = rf_importances.merge(xgb_importances, on="Feature", how="outer").fillna(0)
    combined["Avg_Importance"] = combined[["Importance_RF", "Importance_XGB"]].mean(axis=1)
    combined = combined.sort_values("Avg_Importance", ascending=False)
    print(f"combined importance table created. total features ranked: {len(combined)}")

    # 7. save all outputs to csv files
    print("\nsaving output files to 'model_outputs/' directory...")
    rf_importances.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_randomforest.csv"), index=False)
    xgb_importances.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_xgboost.csv"), index=False)
    combined.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_combined.csv"), index=False)
    print("files saved successfully:")
    print(" - feature_importance_randomforest.csv")
    print(" - feature_importance_xgboost.csv")
    print(" - feature_importance_combined.csv")

    # print summary so user can quickly check top features
    print("\n=== summary of top 10 combined feature importances ===")
    print(combined.head(10))
    print("\nstep 3a complete. feature importance files ready for performance index derivation.")


if __name__ == "__main__":
    main()
