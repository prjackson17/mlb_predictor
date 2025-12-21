""" File to fix the 'avg_runs_coming_in' feature in the MLB dataset. Corrects values by recalculating them based on
prior games only, no more zeros in those columns.

Aided by Gemini."""

import pandas as pd

# 1. Load the dataset
df = pd.read_csv('mlb_2015_2025_dataset.csv')

# 2. Sort by Year and Date
df['game_date'] = pd.to_datetime(df['game_date'])
df = df.sort_values(['year', 'game_date']).reset_index(drop=True)

# Variables to track state
history = {}  # Format: { team_id: {'runs': 0, 'games': 0} }
current_year = None
home_avgs = []
away_avgs = []

# 3. Iterate row by row
for index, row in df.iterrows():
    # RESET stats if we just started a new season
    if row['year'] != current_year:
        history = {}
        current_year = row['year']

    h_team = row['home_team']
    a_team = row['away_team']

    # --- A. FEATURE ENGINEERING (Get stats BEFORE the game) ---
    h_stats = history.get(h_team, {'runs': 0, 'games': 0})
    a_stats = history.get(a_team, {'runs': 0, 'games': 0})

    # Calculate Average (Handle 0 games played)
    h_avg = h_stats['runs'] / max(1, h_stats['games'])
    a_avg = a_stats['runs'] / max(1, a_stats['games'])

    home_avgs.append(h_avg)
    away_avgs.append(a_avg)

    # --- B. UPDATE HISTORY (Record stats AFTER the game) ---
    if h_team not in history: history[h_team] = {'runs': 0, 'games': 0}
    if a_team not in history: history[a_team] = {'runs': 0, 'games': 0}

    history[h_team]['runs'] += row['home_score']
    history[h_team]['games'] += 1

    history[a_team]['runs'] += row['away_score']
    history[a_team]['games'] += 1

# 4. Save results
df['home_avg_runs_coming_in'] = home_avgs
df['away_avg_runs_coming_in'] = away_avgs
df.to_csv('mlb_2015_2025_fixed.csv', index=False)
