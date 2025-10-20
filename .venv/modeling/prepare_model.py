"""
prepare_model_data.py
---------------------------------------
step 2 of phase 4: this script takes the cleaned combined dataset
(all_seasons_clean.csv) and gets it ready for modeling.
it does a few key things:
- loads the cleaned file
- removes unnecessary columns that don’t help modeling
- separates features (x) from target (y = salary)
- fills in any missing values
- scales the numeric columns so they’re all on the same range
- splits the data into training and testing sets
- saves all of these as new csvs for the modeling step
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# where to read and save files
INPUT_FILE = "All_Seasons_Clean.csv"
OUT_DIR = "prepared_data"

# columns we don’t want in the model.
# these are mostly identifiers or non-predictive fields.
DROP_COLS = [
    "Unnamed:_0", "Salary", "Team", "Age", "Min", "+/-", "Id"
]

def main():
    # first check if the combined cleaned file actually exists
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"{INPUT_FILE} not found — run load_all_seasons.py first.")

    print("loading combined dataset...")
    df = pd.read_csv(INPUT_FILE)
    print(f"loaded shape: {df.shape}")

    # drop unwanted columns (only if they exist in the data)
    drop_these = [c for c in DROP_COLS if c in df.columns]
    if drop_these:
        print("dropping columns:", drop_these)
        df = df.drop(columns=drop_these)

    # make sure we still have a salary column
    if "SALARY" not in df.columns:
        raise ValueError("salary column not found in dataset.")

    # y = what we’re trying to predict (the salary)
    # x = all the other features we’ll use to predict it
    y = pd.to_numeric(df["SALARY"], errors="coerce")
    X = df.drop(columns=["SALARY"])

    # convert everything in X to numeric if it’s not already
    # (anything that can’t convert turns into NaN)
    X = X.apply(pd.to_numeric, errors="coerce")

    # fill missing numeric values with the median of that column
    # this is safer than dropping rows and avoids biasing the model
    X = X.fillna(X.median())

    # now scale the features so they’re all centered around 0
    # and have roughly the same range. this helps many models
    # that are sensitive to variable magnitude (like linear regression)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # save the mean and std of each column used for scaling
    # so we can reproduce the transformation later if needed
    scaler_info = pd.DataFrame({
        "feature": X.columns,
        "mean": scaler.mean_,
        "std": scaler.scale_
    })

    # split into 90% training data and 10% testing data
    # this is how we test the model later on data it hasn’t seen before
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.10, random_state=42
    )

    # make an output folder to hold the new csvs
    os.makedirs(OUT_DIR, exist_ok=True)

    # save the new datasets
    # note: we’re saving scaled data here (not raw)
    X_train.to_csv(os.path.join(OUT_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(OUT_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(OUT_DIR, "y_train.csv"), index=False, header=["SALARY"])
    y_test.to_csv(os.path.join(OUT_DIR, "y_test.csv"), index=False, header=["SALARY"])
    scaler_info.to_csv(os.path.join(OUT_DIR, "scaler_means_stds.csv"), index=False)

    print(f"\ndata prep complete. files saved in '{OUT_DIR}/':")
    print(" - X_train.csv")
    print(" - X_test.csv")
    print(" - y_train.csv")
    print(" - y_test.csv")
    print(" - scaler_means_stds.csv")
    print(f"\ntrain shape: {X_train.shape},  test shape: {X_test.shape}")

if __name__ == "__main__":
    main()
