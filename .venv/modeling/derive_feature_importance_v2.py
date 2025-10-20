"""
derive_feature_importance.py
---------------------------------------
step 3a (updated for route 1): compute feature importance per season.

this version trains random forest and xgboost models for each season
separately, normalizes the importance scores for that season,
and then averages them across all seasons. this adjustment accounts
for changes in how different performance metrics are valued over time.

output:
- feature_importance_per_season.csv  (all seasons stacked)
- feature_importance_averaged.csv    (mean importance across seasons)
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


def compute_importance_for_season(df_season, season_label):
    """train models and compute normalized importances for one season"""
    # exclude identifiers
    exclude_cols = ["PLAYER", "TEAM", "Season", "SALARY", "Salary_M", "Player_Comp_Share"]
    X = df_season.select_dtypes(include=[np.number]).drop(columns=[c for c in exclude_cols if c in df_season.columns], errors="ignore")

    # pick salary column
    if "SALARY" in df_season.columns:
        y = df_season["SALARY"]
    elif "Salary_M" in df_season.columns:
        y = df_season["Salary_M"]
    else:
        return None  # skip if no salary

    # skip if not enough data points
    if len(df_season) < 10 or y.nunique() < 2:
        print(f"  skipping {season_label}: not enough data")
        return None

    # fill missing values
    X = X.fillna(0)
    y = y.fillna(0)

    # split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train random forest
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_imp = pd.DataFrame({
        "Feature": X.columns,
        "Importance_RF": rf.feature_importances_
    })

    # train xgboost
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.1, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    xgb_imp = pd.DataFrame({
        "Feature": X.columns,
        "Importance_XGB": xgb.feature_importances_
    })

    # merge, average, normalize
    combined = rf_imp.merge(xgb_imp, on="Feature", how="outer").fillna(0)
    combined["Avg_Importance"] = combined[["Importance_RF", "Importance_XGB"]].mean(axis=1)

    # normalize within this season so importances sum to 1
    combined["Normalized_Importance"] = combined["Avg_Importance"] / combined["Avg_Importance"].sum()

    # add season label
    combined["Season"] = season_label

    return combined


def main():
    print("\nstarting step 3a (season-specific): deriving feature importances per season...")
    print("-------------------------------------------------------------------------------")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} not found — run load_all_seasons.py first")

    print(f"loading dataset from {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)
    print(f"dataset loaded. total rows: {len(df)}, columns: {len(df.columns)}")

    # check for season column
    if "Season" not in df.columns:
        raise ValueError("season column not found — make sure load_all_seasons.py added it")

    all_importances = []

    # loop through each unique season
    for season_label in sorted(df["Season"].unique()):
        print(f"\nprocessing season: {season_label}")
        df_season = df[df["Season"] == season_label]
        imp_df = compute_importance_for_season(df_season, season_label)
        if imp_df is not None:
            all_importances.append(imp_df)
            print(f"  completed {season_label}, features analyzed: {len(imp_df)}")

    # combine all seasons
    if not all_importances:
        print("no valid season data found for modeling.")
        return

    all_combined = pd.concat(all_importances, ignore_index=True)
    print(f"\ncollected feature importances for {all_combined['Season'].nunique()} seasons.")

    # average normalized importances across seasons
    avg_importances = (
        all_combined.groupby("Feature")["Normalized_Importance"]
        .mean()
        .reset_index()
        .sort_values("Normalized_Importance", ascending=False)
    )
    avg_importances = avg_importances.rename(columns={"Normalized_Importance": "Avg_Normalized_Importance"})

    # save outputs
    print("\nsaving per-season and averaged importance files to model_outputs/ ...")
    all_combined.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_per_season.csv"), index=False)
    avg_importances.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_averaged.csv"), index=False)
    print("files saved successfully:")
    print(" - feature_importance_per_season.csv")
    print(" - feature_importance_averaged.csv")

    print("\ntop 10 averaged features:")
    print(avg_importances.head(10))
    print("\nstep 3a (season-specific) complete. ready for updated performance index derivation.")


if __name__ == "__main__":
    main()
