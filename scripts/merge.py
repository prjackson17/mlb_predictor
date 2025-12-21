""" Merge old and new MLB datasets into a single CSV file. Sorted by date."""
import pandas as pd

# Load both
df_old = pd.read_csv("../old/mlb_ml_dataset.csv")
df_new = pd.read_csv("../old/mlb_ml_dataset2020-24.csv")

# Combine and Sort
df_total = pd.concat([df_old, df_new])
df_total['date'] = pd.to_datetime(df_total['date'])
df_total = df_total.sort_values(by='date')

# Save final
df_total.to_csv("../mlb_2015_2025.csv", index=False)
print(f"Merged! Total games: {len(df_total)}")