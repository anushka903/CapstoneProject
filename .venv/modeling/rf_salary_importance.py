# -----------------------------
# NBA Player Salary Optimization Framework (Optimized, AGE removed, unnecessary columns removed)
# -----------------------------
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load the dataset
# -----------------------------
df = pd.read_csv("All_Seasons_Clean.csv")  # Replace with your path

# -----------------------------
# 1a. Remove unnecessary columns
# -----------------------------
for col in ['Season', 'UNNAMED:_0', 'Unnamed:_0', 'Salary', 'Age', 'Min', '+/-', 'Id']:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# ['Season', 'UNNAMED:_0', 'Unnamed:_0', 'DD2', '3PA', 'TD3', '3P', 'Salary', 'Age', 'Min', '+/-', 'Id', 'FT%', 'FG%', 'FTA', 'FGA', 'REB']
# -----------------------------
# 1b. Handle missing target
# -----------------------------
df = df.dropna(subset=['SALARY']).reset_index(drop=True)

# Keep player info for later
player_info = df[['PLAYER']].copy()  # Only Player needed now

# -----------------------------
# 2. Preprocessing
# -----------------------------
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
numeric_cols.remove('SALARY')
if 'Player_Comp_Share' in numeric_cols:
    numeric_cols.remove('Player_Comp_Share')

X = df[numeric_cols].fillna(0)
y = df['SALARY']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 2b. Correlation Heatmap
# -----------------------------
corr_matrix = df[numeric_cols + ['SALARY']].corr()

plt.figure(figsize=(14, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)
plt.title("Correlation Heatmap Between Variables and Salary", fontsize=14)
plt.tight_layout()
plt.show()

# Optional: Focused correlation with Salary only
salary_corr = corr_matrix['SALARY'].sort_values(ascending=False)

plt.figure(figsize=(8, 10))
sns.barplot(x=salary_corr.values, y=salary_corr.index, palette="coolwarm")
plt.title("Feature Correlation with Salary", fontsize=14)
plt.xlabel("Correlation Coefficient")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# -----------------------------
# 3. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Random Forest Regression
# -----------------------------
rf = RandomForestRegressor(
    n_estimators=200,  # fewer trees for speed
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# -----------------------------
# 5. Evaluate model
# -----------------------------
y_pred = rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Test RMSE: {rmse}")
print(f"Test R2: {r2}")

# -----------------------------
# 6. Feature importance
# -----------------------------
feature_importances = pd.DataFrame({
    'feature': numeric_cols,
    'importance': rf.feature_importances_
}).sort_values(by='importance', ascending=False)

# Print feature importance for each feature
print("\nFeature Importances:")
for idx, row in feature_importances.iterrows():
    print(f"{row['feature']}: {row['importance']:.4f}")

# Plot feature importance
plt.figure(figsize=(12,6))
sns.barplot(x='importance', y='feature', data=feature_importances)
plt.title("Feature Importance from Random Forest")
plt.tight_layout()
plt.show()

# -----------------------------
# 7. Predict salaries for all players
# -----------------------------
df['predicted_salary'] = rf.predict(scaler.transform(X))

# -----------------------------
# 8. Fast Tier Assignment (Percentiles)
# -----------------------------
bounds = np.percentile(df['predicted_salary'], [25, 50, 75])

def assign_tier(salary):
    if salary <= bounds[0]:
        return 'Tier 4'
    elif salary <= bounds[1]:
        return 'Tier 3'
    elif salary <= bounds[2]:
        return 'Tier 2'
    else:
        return 'Tier 1'

df['salary_tier'] = df['predicted_salary'].apply(assign_tier)

# Combine player info
tiered_players = pd.concat([
    player_info,
    df[['predicted_salary', 'salary_tier']]
], axis=1)

# Optional: Show salary ranges for each tier
tier_summary = tiered_players.groupby('salary_tier')['predicted_salary'].agg(['min','max','mean']).reset_index()
print("\nSalary Tier Summary:")
print(tier_summary)

# -----------------------------
# 9. Save results (only necessary columns)
# -----------------------------
print("\nTop 20 Players with Predicted Salary & Tier:")
print(tiered_players.head(20).to_string(index=False))
# tiered_players.to_csv("NBA_Player_Salary_Tiers_Optimized.csv", index=False)
# print("\nResults saved to NBA_Player_Salary_Tiers_Optimized.csv")
