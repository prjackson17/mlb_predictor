import streamlit as st
import pandas as pd
import joblib
import numpy as np

TEAM_ID_MAP = {
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

st.set_page_config(page_title="MLB Live Predictor", page_icon="⚾", layout="wide")


@st.cache_resource
def load_assets():
    model = joblib.load('data/random_forest.pkl')
    stats = pd.read_csv('data/team_stats.csv')
    return model, stats


try:
    model, team_stats = load_assets()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

available_ids = team_stats['team_name'].unique().tolist()
sorted_ids = sorted(available_ids, key=lambda x: TEAM_ID_MAP.get(int(x), f"Team {x}"))

st.title("⚾ Live MLB Win Predictor")
st.caption("Predictions update automatically as you change teams or weather conditions.")
st.markdown("---")

col_a, col_b, col_c = st.columns([1, 0.2, 1])

with col_a:
    st.subheader("🏠 Home Team")
    home_id = st.selectbox(
        "Select Home Team",
        options=sorted_ids,
        format_func=lambda x: TEAM_ID_MAP.get(int(x), f"Team {x}"),
        key="home_sel"
    )

with col_b:
    st.markdown("<h2 style='text-align: center; padding-top: 40px;'>VS</h2>", unsafe_allow_html=True)

with col_c:
    st.subheader("✈️ Away Team")
    away_options = [tid for tid in sorted_ids if tid != home_id]
    away_id = st.selectbox(
        "Select Away Team",
        options=away_options,
        format_func=lambda x: TEAM_ID_MAP.get(int(x), f"Team {x}"),
        index=0,
        key="away_sel"
    )

st.sidebar.header("🕹️ Live Game Conditions")
temp = st.sidebar.slider("Temperature (°F)", 40, 100, 72)
wind = st.sidebar.slider("Wind Speed (mph)", 0, 30, 5)

h_data = team_stats[team_stats['team_name'] == home_id].iloc[0]
a_data = team_stats[team_stats['team_name'] == away_id].iloc[0]

input_features = pd.DataFrame([{
    'temp': float(temp),
    'wind_speed': float(wind),
    'diff_run_diff': float(h_data['run_diff'] - a_data['run_diff']),
    'diff_ops': float(h_data['ops'] - a_data['ops']),
    'diff_whip': float(h_data['whip'] - a_data['whip']),
    'diff_wins_last_10': float(h_data['wins_last_10'] - a_data['wins_last_10']),
    'diff_games_last_7': float(h_data['games_last_7'] - a_data['games_last_7']),
    'park_factor': float(h_data['park_factor'])
}])

probabilities = model.predict_proba(input_features)[0]
home_win_prob = probabilities[1]

st.markdown("---")
res_col1, res_col2 = st.columns(2)

with res_col1:
    st.metric(
        label=f"{TEAM_ID_MAP.get(int(home_id))} Win Probability",
        value=f"{home_win_prob:.1%}"
    )
    st.progress(home_win_prob)

with res_col2:
    winner_name = TEAM_ID_MAP.get(int(home_id)) if home_win_prob > 0.5 else TEAM_ID_MAP.get(int(away_id))
    st.subheader("Model's Favorite:")
    st.title(f"🏆 {winner_name}")

with st.expander("📊 Analysis: Tale of the Tape"):
    comparison = pd.DataFrame([
        {"Team": TEAM_ID_MAP.get(int(home_id)), "Run Diff": h_data['run_diff'], "OPS": h_data['ops'],
         "WHIP": h_data['whip']},
        {"Team": TEAM_ID_MAP.get(int(away_id)), "Run Diff": a_data['run_diff'], "OPS": a_data['ops'],
         "WHIP": a_data['whip']}
    ])
    st.table(comparison.set_index("Team"))

st.markdown("---")
with st.expander("🎯 2025 Season Simulation Accuracy"):
    try:
        proj_df = pd.read_csv('data/projections_2025.csv', index_col=0)

        # 2. Actual results dictionary
        actual_wins = {
            'Toronto Blue Jays': 94, 'New York Yankees': 94, 'Boston Red Sox': 89, 'Tampa Bay Rays': 77,
            'Baltimore Orioles': 75,
            'Cleveland Guardians': 88, 'Detroit Tigers': 87, 'Kansas City Royals': 82, 'Minnesota Twins': 70,
            'Chicago White Sox': 60,
            'Seattle Mariners': 90, 'Houston Astros': 87, 'Texas Rangers': 81, 'Oakland Athletics': 76,
            'Los Angeles Angels': 72,
            'Philadelphia Phillies': 96, 'New York Mets': 83, 'Miami Marlins': 79, 'Atlanta Braves': 76,
            'Washington Nationals': 66,
            'Milwaukee Brewers': 97, 'Chicago Cubs': 92, 'Cincinnati Reds': 83, 'St. Louis Cardinals': 78,
            'Pittsburgh Pirates': 71,
            'Los Angeles Dodgers': 93, 'San Diego Padres': 90, 'San Francisco Giants': 81, 'Arizona Diamondbacks': 80,
            'Colorado Rockies': 43
        }

        # 3. Process Data
        proj_df['Actual Wins'] = proj_df.index.map(actual_wins)
        proj_df['Error'] = proj_df['Avg_Wins'] - proj_df['Actual Wins']
        mae = proj_df['Error'].abs().mean()

        # 4. Display Metrics & Table
        st.write(
            "Comparing pre-season Monte Carlo simulation results (100k runs) against actual 2025 final win totals.")
        st.metric("Mean Absolute Error (MAE)", f"{mae:.2f} Wins")

        # Highlight the table with color coding
        st.dataframe(
            proj_df[['Avg_Wins', 'Actual Wins', 'Error']].sort_values(by='Error'),
            use_container_width=True
        )

    except FileNotFoundError:
        st.info(
            "💡 To see 2025 accuracy data here, run your `simulate_2025.py` script locally to generate the "
            "`projections_2025.csv` file.")

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        <p>Developed by <b>Parker Jackson</b></p>
        <a href="https://github.com/prjackson17/mlb_predictor" target="_blank">
            <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
        </a>
    </div>
    """,
    unsafe_allow_html=True
)