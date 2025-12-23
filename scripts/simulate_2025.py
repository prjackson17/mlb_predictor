import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

TEAM_ID_MAP = {     # map team id to team name
    108: 'Los Angeles Angels', 109: 'Arizona Diamondbacks', 110: 'Baltimore Orioles',
    111: 'Boston Red Sox', 112: 'Chicago Cubs', 113: 'Cincinnati Reds',
    114: 'Cleveland Guardians', 115: 'Colorado Rockies', 116: 'Detroit Tigers',
    117: 'Houston Astros', 118: 'Kansas City Royals', 119: 'Los Angeles Dodgers',
    120: 'Washington Nationals', 121: 'New York Mets', 133: 'Oakland Athletics',
    134: 'Pittsburgh Pirates', 135: 'San Diego Padres', 136: 'Seattle Mariners',
    137: 'San Francisco Giants', 138: 'St. Louis Cardinals', 139: 'Tampa Bay Rays',
    140: 'Texas Rangers', 141: 'Toronto Blue Jays', 142: 'Minnesota Twins',
    143: 'Philadelphia Phillies', 144: 'Atlanta Braves', 145: 'Chicago White Sox',
    146: 'Miami Marlins', 147: 'New York Yankees', 158: 'Milwaukee Brewers'
}

print("Loading model and statistics...")
model = joblib.load('../data/random_forest.pkl')
team_stats = pd.read_csv('../data/team_stats.csv')

df = pd.read_csv('../data/mlb_2015_2025_dataset.csv')      # grab only 2025 stats
df['date'] = pd.to_datetime(df['date'])
schedule_2025 = df[df['date'].dt.year == 2025].copy()

if len(schedule_2025) == 0:
    print("Error: No games found for 2025 in the dataset.")
    exit()

print("Preparing 2025 schedule features...")
schedule_2025 = schedule_2025.merge(
    team_stats, left_on='home_team', right_on='team_name', suffixes=('', '_h')
)

schedule_2025 = schedule_2025.merge(
    team_stats, left_on='away_team', right_on='team_name', suffixes=('_h', '_a')
)

# differential stats
schedule_2025['diff_run_diff'] = schedule_2025['run_diff_h'] - schedule_2025['run_diff_a']
schedule_2025['diff_ops'] = schedule_2025['ops_h'] - schedule_2025['ops_a']
schedule_2025['diff_whip'] = schedule_2025['whip_h'] - schedule_2025['whip_a']
schedule_2025['diff_wins_last_10'] = schedule_2025['wins_last_10_h'] - schedule_2025['wins_last_10_a']
schedule_2025['diff_games_last_7'] = schedule_2025['games_last_7_h'] - schedule_2025['games_last_7_a']
schedule_2025['park_factor'] = schedule_2025['park_factor_h']

features = [
    'temp', 'wind_speed', 'diff_run_diff', 'diff_ops',
    'diff_whip', 'diff_wins_last_10', 'diff_games_last_7', 'park_factor'
]

X = schedule_2025[features].fillna(0)
schedule_2025['home_win_prob'] = model.predict_proba(X)[:, 1]

# Monte Carlo Simulation
n_simulations = 100000
sim_results = []

print(f"Simulating the 2025 season {n_simulations} times...")
for i in tqdm(range(n_simulations)):
    draws = np.random.rand(len(schedule_2025))
    schedule_2025['sim_win'] = (schedule_2025['home_win_prob'] > draws).astype(int)

    h_wins = schedule_2025.groupby('home_team')['sim_win'].sum()
    a_wins = schedule_2025.groupby('away_team')['sim_win'].apply(lambda x: (x == 0).sum())

    total_wins = h_wins.add(a_wins, fill_value=0)
    sim_results.append(total_wins)

results_df = pd.concat(sim_results, axis=1)

summary = pd.DataFrame({
    'Avg_Wins': results_df.mean(axis=1),
    'P10_Wins': results_df.quantile(0.1, axis=1),
    'P90_Wins': results_df.quantile(0.9, axis=1),
})

summary.index = summary.index.map(TEAM_ID_MAP)
summary = summary.sort_values('Avg_Wins', ascending=False)

print("\n--- 2025 MONTE CARLO PROJECTIONS ---")
print(summary.round(1))

summary.to_csv('../data/projections_2025.csv')
print("\nResults saved to 'projections_2025.csv'")