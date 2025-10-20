import pandas as pd

def merge_salary_with_share(salary_file, stats_file, total_player_comp, output_file="Full WNBA Dataset 2024-2025.csv"):
    # Load CSVs
    salary_df = pd.read_csv(salary_file)
    stats_df = pd.read_csv(stats_file)

    # Make sure player names line up
    salary_df = salary_df.rename(columns={"NAME": "PLAYER"})

    # Clean Salary column: remove $ and commas, convert to float
    salary_df["Salary"] = (
        salary_df["Salary"]
        .replace(r'[\$,]', '', regex=True)  # remove $ and commas
        .astype(float)
    )

    # Merge salary into stats
    merged = stats_df.merge(
        salary_df[["PLAYER", "Salary"]],
        on="PLAYER",
        how="left"
    )

    # Calculate share of total player compensation
    merged["Player_Comp_Share"] = merged["Salary"] / float(total_player_comp)

    # Save to CSV
    merged.to_csv(output_file, index=False)
    print(f"✅ Done! Merged file saved as {output_file}")


# Example usage:
merge_salary_with_share(
    "WNBA Salary Info - Sheet1.csv",
    "WNBA Player Stats - Sheet1.csv",
    total_player_comp=66_030_000.00
)
