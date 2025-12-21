import pandas as pd
from collections import defaultdict

# --- CONFIGURATION ---
INPUT_FILE = "../data/mlb_2015_2025_dataset.csv"  # Your existing file
OUTPUT_FILE = "../data/mlb_dataset_with_momentum.csv"  # The new improved file


def add_momentum_features():
    print(f"Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("Error: File not found. Please check the filename.")
        return

    # 1. Sort Chronologically (CRITICAL)
    # We must process games in order to build correct history
    print("Sorting data by date...")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['date', 'game_id'])

    # 2. Initialize History Tracker
    # Dictionary mapping Team ID -> List of recent results (1=Win, 0=Loss)
    team_history = defaultdict(list)

    # Lists to hold the new column data
    h_last_10, h_last_5 = [], []
    a_last_10, a_last_5 = [], []

    print("Calculating momentum...")

    # 3. Iterate Row by Row
    for index, row in df.iterrows():
        home_id = row['home_team']
        away_id = row['away_team']
        home_win = row['home_win']  # 1 if Home Won, 0 if Away Won

        # A. CALCULATE MOMENTUM (Before the game starts)
        # Home Team
        h_wins = team_history[home_id]
        h_last_10.append(sum(h_wins[-10:]))  # Sums the last 10 wins (1s)
        h_last_5.append(sum(h_wins[-5:]))

        # Away Team
        a_wins = team_history[away_id]
        a_last_10.append(sum(a_wins[-10:]))
        a_last_5.append(sum(a_wins[-5:]))

        # B. UPDATE HISTORY (After the game ends)
        if home_win == 1:
            team_history[home_id].append(1)  # Win
            team_history[away_id].append(0)  # Loss
        else:
            team_history[home_id].append(0)  # Loss
            team_history[away_id].append(1)  # Win

    # 4. Add Columns to DataFrame
    df['home_wins_last_10'] = h_last_10
    df['home_wins_last_5'] = h_last_5
    df['away_wins_last_10'] = a_last_10
    df['away_wins_last_5'] = a_last_5

    # 5. Add Difference Feature (Recommended!)
    # This is the single best predictor: Is home hotter than away?
    df['diff_wins_last_10'] = df['home_wins_last_10'] - df['away_wins_last_10']

    # 6. Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Success! Saved updated dataset to {OUTPUT_FILE}")
    print("\nPreview of new columns:")
    print(df[['date', 'home_team', 'home_wins_last_10', 'diff_wins_last_10']].tail())


if __name__ == "__main__":
    add_momentum_features()
