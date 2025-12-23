import pandas as pd

df = pd.read_csv("../data/mlb_2015_2025_dataset.csv", index_col=0)

# optimize and change data types
df['date'] = pd.to_datetime(df['date'])  # ensure dates are proper type
df = df[df['condition'] != 'Unknown']
df["condition"] = df["condition"].astype("category")  # weather conditions --> category
df["home_team"] = df["home_team"].astype("category")  # team id's and venue id are not numeric --> category
df["away_team"] = df["away_team"].astype("category")
df["venue_id"] = df["venue_id"].astype("category")
int_cols = df.select_dtypes(include=["int"])
df[int_cols.columns] = df[int_cols.columns].apply(pd.to_numeric, downcast="integer")  # downcast ints
float_cols = df.select_dtypes(include=["float"])
df[float_cols.columns] = df[float_cols.columns].apply(pd.to_numeric, downcast="float")  # downcast floats

# Load and merge park factors
park_factors = pd.read_csv("../data/venue_park_factors.csv")
park_factors['venue_id'] = park_factors['venue_id'].astype("category")

# Merge park factors with main dataset
df = df.merge(
    park_factors[['venue_id', 'park_factor']],
    on='venue_id',
    how='left'
)

# Fill any missing park factors with neutral value (100)
df['park_factor'] = df['park_factor'].fillna(100)

# Normalize park factor to be centered around 1.0 instead of 100
df['park_factor'] = df['park_factor'] / 100

# ensure value ranges are correct
df['home_games_last_7'] = df['home_games_last_7'].clip(upper=7)  # ensure "last 7" can't be greater than 7
df['away_games_last_7'] = df['away_games_last_7'].clip(upper=7)

# add values
# df["away_win"] = df["home_win"] * -1 + 1
df['diff_wins_last_5'] = df['home_wins_last_5'] - df['away_wins_last_5']
df['diff_run_diff'] = df['home_run_diff'] - df['away_run_diff']  # difference in run differentials
df['diff_avg'] = df['home_avg'] - df['away_avg']  # difference in batting average
df['diff_ops'] = df['home_ops'] - df['away_ops']  # difference in OPS
df['diff_era'] = df['home_starter_era'] - df['away_starter_era']  # difference in starter ERA
df['diff_whip'] = df['home_starter_whip'] - df['away_starter_whip']  # difference in starter WHIP
df['diff_rest'] = df['home_rest'] - df['away_rest']  # difference in rest days
df['diff_games_last_7'] = df['home_games_last_7'] - df['away_games_last_7']  # difference in games in last 7 days
df['diff_win_pct'] = df['home_win_pct'] - df['away_win_pct']  # difference in win percentage

del int_cols, float_cols

latest_stats = []

# Group by home team to get their most recent stats in the dataset
for team in df['home_team'].unique():
    # Get the most recent game for this team
    team_data = df[(df['home_team'] == team) | (df['away_team'] == team)].sort_values('date').iloc[-1]

    # Check if they were home or away in that last game to pull the correct stats
    if team_data['home_team'] == team:
        latest_stats.append({
            'team_name': team,
            'run_diff': team_data['home_run_diff'],
            'ops': team_data['home_ops'],
            'whip': team_data['home_starter_whip'],
            'wins_last_10': team_data['home_wins_last_10'],
            'games_last_7': team_data['home_games_last_7'],
            'park_factor': team_data['park_factor']
        })
    else:
        latest_stats.append({
            'team_name': team,
            'run_diff': team_data['away_run_diff'],
            'ops': team_data['away_ops'],
            'whip': team_data['away_starter_whip'],
            'wins_last_10': team_data['away_wins_last_10'],
            'games_last_7': team_data['away_games_last_7'],
            'park_factor': team_data['park_factor']  # Ideally lookup park factor separately
        })

pd.DataFrame(latest_stats).to_csv('../data/team_stats.csv', index=False)
