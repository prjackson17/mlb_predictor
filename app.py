import streamlit as st
import pandas as pd
import joblib

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

st.set_page_config(page_title="MLB Live Predictor", page_icon="⚾", layout="wide")


@st.cache_resource
def load_assets():
    model = joblib.load('data/random_forest.pkl')   # random forest model
    stats = pd.read_csv('data/team_stats.csv')
    return model, stats


try:
    model, team_stats = load_assets()

except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# sort team names
available_ids = team_stats['team_name'].unique().tolist()
sorted_ids = sorted(available_ids, key=lambda x: TEAM_ID_MAP.get(int(x), f"Team {x}"))

st.title("⚾ MLB Game Predictor")
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
    # BLACKOUT LOGIC: Filter the sorted list to remove the home_id
    away_options = [tid for tid in sorted_ids if tid != home_id]

    away_id = st.selectbox(
        "Select Away Team",
        options=away_options,
        format_func=lambda x: TEAM_ID_MAP.get(int(x), f"Team {x}"),
        # Default to the first team in the filtered list
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

st.sidebar.markdown("---")
st.sidebar.info(f"Created by [Parker Jackson](https://github.com/prjackson17/)")