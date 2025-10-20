"""
load_all_seasons.py
---------------------------------------
step 1 of phase 4: this script loads all the season csv files (2001–2002 through 2024–2025)
from the 'data/' folder, cleans them up a bit, merges everything into one large dataframe,
and then saves that combined file as 'all_seasons_clean.csv'.
"""

import pandas as pd
import numpy as np
import os
import re

# folder that holds all the raw season csvs
DATA_DIR = "data"

# final output file that will contain every season stacked together
OUTPUT_FILE = "All_Seasons_Clean.csv"


# helper function to clean a single season csv before combining
def clean_single_season(path, season_label):
    """
    loads one season’s csv file and cleans it up
    - fixes column names (removes spaces, renames weird ones)
    - converts numeric columns to actual numbers
    - adds a season label column so we know what year each row came from
    """

    # read csv as strings to avoid dtype issues
    df = pd.read_csv(path, dtype=str)

    # make column names consistent (remove spaces and make them uniform)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # some columns have strange names (like "3:00 PM" or "#ERROR!"),
    # so fix those so they’re easier to use later
    if "3:00_PM" in df.columns:
        df = df.rename(columns={"3:00_PM": "3P"})
    if "#ERROR!" in df.columns:
        df = df.rename(columns={"#ERROR!": "ERROR"})

    # list of columns we expect to be numeric
    # this covers most of the stat columns (pts, reb, ast, etc.)
    numeric_cols = [
        "SALARY","AGE","GP","W","L","MIN","PTS","FGM","FGA","FG%",
        "3P","3PA","3P%","FTM","FTA","FT%","OREB","DREB","REB",
        "AST","TOV","STL","BLK","PF","FP","DD2","TD3","ERROR","Player_Comp_Share"
    ]

    # loop through each column and convert it to numeric if it’s in that list
    # this also removes any weird symbols like $, commas, or % signs
    for col in df.columns:
        if col.upper() in numeric_cols:
            df[col] = (
                df[col]
                .str.replace(r"[\$,]", "", regex=True)  # remove $ and commas
                .str.replace("%", "", regex=False)       # remove %
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")  # convert to float

    # add a season label column so we know which season each row came from
    df["Season"] = season_label

    return df


def main():
    # step 1: get all csv filenames from the data folder
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".csv"))
    if not files:
        print(f"no csvs found in {DATA_DIR}")
        return

    print(f"found {len(files)} csv files. loading...")

    all_dfs = []

    # step 2: loop through each csv, clean it, and store it in a list
    for f in files:
        # try to extract the season label from the filename, like "2001_2002"
        season = re.findall(r"\d{4}[-_]\d{4}", f)
        season_label = season[0].replace("_", "-") if season else f
        print(f" → loading {f} as {season_label}")

        # create full file path
        path = os.path.join(DATA_DIR, f)

        # clean and append
        df_clean = clean_single_season(path, season_label)
        all_dfs.append(df_clean)

    # step 3: stack all seasons into one big dataframe
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"combined shape: {combined.shape}")

    # step 4: remove any duplicate rows that might have shown up
    combined = combined.drop_duplicates()

    # step 5: quick cleanup of player/team name spacing
    if "PLAYER" in combined.columns:
        combined["PLAYER"] = combined["PLAYER"].str.strip()
    if "TEAM" in combined.columns:
        combined["TEAM"] = combined["TEAM"].str.strip()

    # step 6: save the fully combined clean dataset
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\nall seasons combined and cleaned → {OUTPUT_FILE}")

    # optional: print a quick summary
    print("\ncolumns:", list(combined.columns))
    print("sample rows:\n", combined.head(5))


if __name__ == "__main__":
    main()
