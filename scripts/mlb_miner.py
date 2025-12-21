"""
MLB Data Miner for Machine Learning (2015-2025)
"""

import statsapi
import pandas as pd
import time
from datetime import datetime, timedelta

# --- CONFIGURATION ---
START_YEAR = 2015
END_YEAR = 2025
OUTPUT_FILE = "../data/mlb_2015_2025_dataset2.csv"


# --- HELPER CLASSES ---

class TeamTracker:
    def __init__(self):
        self.games_played = 0
        self.wins = 0
        self.runs_scored = 0
        self.runs_allowed = 0
        self.hits = 0
        self.at_bats = 0
        self.walks = 0
        self.total_bases = 0
        self.strikeouts = 0
        self.last_game_date = None
        self.recent_games = []  # Tracks dates for fatigue
        self.recent_results = []  # Tracks Wins (True/False) for momentum

    def update_stats(self, bat_stats, runs_scored, runs_allowed, game_date, win):
        self.games_played += 1
        self.runs_scored += runs_scored
        self.runs_allowed += runs_allowed
        if win: self.wins += 1

        # Safe Stat Extraction
        def get_stat(key):
            try:
                return int(bat_stats.get(key, 0) or 0)
            except:
                return 0

        self.hits += get_stat('hits')
        self.at_bats += get_stat('atBats')
        self.walks += get_stat('baseOnBalls')
        self.strikeouts += get_stat('strikeOuts')

        h = get_stat('hits')
        d = get_stat('doubles')
        t = get_stat('triples')
        hr = get_stat('homeRuns')
        singles = h - (d + t + hr)
        self.total_bases += singles + (2 * d) + (3 * t) + (4 * hr)

        # Update History
        self.last_game_date = game_date

        # 1. Fatigue History (Dates)
        self.recent_games.append(game_date)
        self.recent_games = self.recent_games[-10:]

        # 2. Momentum History (Wins)
        self.recent_results.append(1 if win else 0)
        self.recent_results = self.recent_results[-10:]  # Keep last 10 games

    def get_features(self):
        if self.at_bats == 0:
            avg = 0.0
            slg = 0.0
        else:
            avg = self.hits / self.at_bats
            slg = self.total_bases / self.at_bats

        obp_denom = self.at_bats + self.walks
        obp = (self.hits + self.walks) / obp_denom if obp_denom > 0 else 0.0
        ops = obp + slg
        win_pct = self.wins / self.games_played if self.games_played > 0 else 0.0

        return {
            'win_pct': round(win_pct, 3),
            'avg': round(avg, 3),
            'ops': round(ops, 3),
            'run_diff': round((self.runs_scored - self.runs_allowed) / max(1, self.games_played), 2)
        }

    def get_momentum(self):
        """Returns wins in last 5 and last 10 games"""
        # Sum the 1s in the recent_results list
        wins_10 = sum(self.recent_results)
        wins_5 = sum(self.recent_results[-5:])  # Last 5 only

        return {
            'wins_last_10': wins_10,
            'wins_last_5': wins_5
        }

    def get_recent_fatigue(self, current_date_obj):
        if not self.recent_games: return 0
        count = 0
        for d_str in self.recent_games:
            try:
                d = datetime.strptime(d_str, "%Y-%m-%d")
                if 0 < (current_date_obj - d).days <= 7:
                    count += 1
            except:
                continue
        return count


class PitcherTracker:
    def __init__(self):
        self.innings_pitched = 0.0
        self.earned_runs = 0
        self.walks = 0
        self.hits = 0

    def update_stats(self, p_stats):
        try:
            ip_raw = str(p_stats.get('inningsPitched', '0'))
            if '.' in ip_raw:
                whole, part = ip_raw.split('.')
                ip = int(whole) + (int(part) * 0.333)
            else:
                ip = float(ip_raw)

            self.innings_pitched += ip
            self.earned_runs += int(p_stats.get('earnedRuns', 0) or 0)
            self.walks += int(p_stats.get('baseOnBalls', 0) or 0)
            self.hits += int(p_stats.get('hits', 0) or 0)
        except:
            pass

    def get_career_features(self):
        if self.innings_pitched == 0:
            return {'era': 4.50, 'whip': 1.35}
        era = 9 * (self.earned_runs / self.innings_pitched)
        whip = (self.walks + self.hits) / self.innings_pitched
        return {'era': round(era, 2), 'whip': round(whip, 2)}


# --- EXTRACTORS ---

def get_days_rest(last_date_str, current_date_str):
    if not last_date_str: return 5
    try:
        d1 = datetime.strptime(last_date_str, "%Y-%m-%d")
        d2 = datetime.strptime(current_date_str, "%Y-%m-%d")
        return max(0, (d2 - d1).days - 1)
    except:
        return 0


def get_schedule_chunked(year):
    all_games = []
    current_date = datetime(year, 3, 20)
    end_date = datetime(year, 11, 5)

    print(f"Fetching schedule for {year}...")
    while current_date < end_date:
        next_date = current_date + timedelta(days=7)
        s_str = current_date.strftime("%m/%d/%Y")
        e_str = next_date.strftime("%m/%d/%Y")

        retry = 0
        chunk = None
        while retry < 3 and chunk is None:
            try:
                chunk = statsapi.schedule(start_date=s_str, end_date=e_str, sportId=1)
            except:
                retry += 1
                time.sleep(1 * retry)

        if chunk: all_games.extend(chunk)
        current_date = next_date
        time.sleep(0.1)

    unique_games = {g['game_id']: g for g in all_games}
    return list(unique_games.values())


def extract_weather_and_venue(box):
    data = {'temp': 70, 'wind_speed': 0, 'condition': 'Unknown', 'venue_id': None}
    try:
        data['venue_id'] = box.get('gameData', {}).get('venue', {}).get('id')
        w = box.get('gameData', {}).get('weather', {})
        data['temp'] = int(w.get('temp', 70) or 70)
        data['condition'] = w.get('condition', 'Unknown')
        wind_str = w.get('wind', '')
        if 'mph' in wind_str:
            data['wind_speed'] = int(wind_str.split('mph')[0].strip())
    except:
        pass
    return data


def find_pitcher_stats_in_box(box_players, probable_id):
    p_key = f"ID{probable_id}"
    if p_key in box_players:
        return box_players[p_key].get('stats', {}).get('pitching', {})

    for pid, p_data in box_players.items():
        try:
            stats = p_data.get('stats', {}).get('pitching', {})
            if stats and 'inningsPitched' in stats:
                return stats
        except:
            continue
    return {}


# --- MAIN ---

def scrape_mlb_data():
    all_rows = []

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\n=== PROCESSING {year} ===")
        team_history = {}
        pitcher_history = {}

        schedule = get_schedule_chunked(year)
        schedule.sort(key=lambda x: x['game_date'])

        valid_types = ['R', 'F', 'D', 'L', 'W']
        schedule = [g for g in schedule if g.get('game_type') in valid_types]
        print(f"Games to Process: {len(schedule)}")

        for i, game in enumerate(schedule):
            if game['status'] != 'Final': continue

            game_id = game['game_id']
            home_id = game['home_id']
            away_id = game['away_id']
            game_date = game['game_date']

            if home_id not in team_history: team_history[home_id] = TeamTracker()
            if away_id not in team_history: team_history[away_id] = TeamTracker()

            # --- 1. PRE-GAME FEATURES ---

            h_feats = team_history[home_id].get_features()
            a_feats = team_history[away_id].get_features()

            # MOMENTUM (New!)
            h_mom = team_history[home_id].get_momentum()
            a_mom = team_history[away_id].get_momentum()

            curr_date_obj = datetime.strptime(game_date, "%Y-%m-%d")
            h_rest = get_days_rest(team_history[home_id].last_game_date, game_date)
            a_rest = get_days_rest(team_history[away_id].last_game_date, game_date)
            h_fatigue = team_history[home_id].get_recent_fatigue(curr_date_obj)
            a_fatigue = team_history[away_id].get_recent_fatigue(curr_date_obj)

            retry = 0
            box = None
            while retry < 5 and box is None:
                try:
                    box = statsapi.get("game", {"gamePk": game_id})
                except:
                    retry += 1
                    time.sleep(0.5 * retry)

            if not box: continue

            try:
                h_prob = box['gameData']['probablePitchers']['home']['id']
                a_prob = box['gameData']['probablePitchers']['away']['id']
            except:
                h_prob, a_prob = None, None

            h_p_stats = pitcher_history.get(h_prob, PitcherTracker()).get_career_features()
            a_p_stats = pitcher_history.get(a_prob, PitcherTracker()).get_career_features()

            env = extract_weather_and_venue(box)

            # --- RECORD ROW ---
            row = {
                "game_id": game_id,
                "year": year,
                "date": game_date,
                "home_team": home_id,
                "away_team": away_id,
                "venue_id": env['venue_id'],

                # Conditions
                "temp": env['temp'],
                "wind_speed": env['wind_speed'],
                "condition": env['condition'],

                # Fatigue/Rest
                "home_rest": h_rest,
                "away_rest": a_rest,
                "home_games_last_7": h_fatigue,
                "away_games_last_7": a_fatigue,

                # Momentum (New Columns)
                "home_wins_last_5": h_mom['wins_last_5'],
                "home_wins_last_10": h_mom['wins_last_10'],
                "away_wins_last_5": a_mom['wins_last_5'],
                "away_wins_last_10": a_mom['wins_last_10'],

                # Team Stats
                "home_win_pct": h_feats['win_pct'],
                "home_ops": h_feats['ops'],
                "home_avg": h_feats['avg'],
                "home_run_diff": h_feats['run_diff'],
                "away_win_pct": a_feats['win_pct'],
                "away_ops": a_feats['ops'],
                "away_avg": a_feats['avg'],
                "away_run_diff": a_feats['run_diff'],

                # Pitching
                "home_starter_era": h_p_stats['era'],
                "home_starter_whip": h_p_stats['whip'],
                "away_starter_era": a_p_stats['era'],
                "away_starter_whip": a_p_stats['whip'],

                # Targets
                "home_score": game['home_score'],
                "away_score": game['away_score'],
                "home_win": 1 if game['home_score'] > game['away_score'] else 0
            }
            all_rows.append(row)

            # --- 2. UPDATE HISTORY ---
            try:
                teams_box = box['liveData']['boxscore']['teams']
                h_bat = teams_box['home']['teamStats']['batting']
                a_bat = teams_box['away']['teamStats']['batting']

                team_history[home_id].update_stats(h_bat, game['home_score'], game['away_score'], game_date,
                                                   row['home_win'])
                team_history[away_id].update_stats(a_bat, game['away_score'], game['home_score'], game_date,
                                                   not row['home_win'])

                all_players = teams_box['home']['players']
                all_players.update(teams_box['away']['players'])

                if h_prob:
                    stats = find_pitcher_stats_in_box(teams_box['home']['players'], h_prob)
                    if h_prob not in pitcher_history: pitcher_history[h_prob] = PitcherTracker()
                    pitcher_history[h_prob].update_stats(stats)

                if a_prob:
                    stats = find_pitcher_stats_in_box(teams_box['away']['players'], a_prob)
                    if a_prob not in pitcher_history: pitcher_history[a_prob] = PitcherTracker()
                    pitcher_history[a_prob].update_stats(stats)
            except:
                pass

            if i % 50 == 0:
                print(f" {i}/{len(schedule)} | {game_date} | Rows: {len(all_rows)}")

        df = pd.DataFrame(all_rows)
        if not df.empty:
            df.to_csv(OUTPUT_FILE, index=False)
            print(f"Saved {year}")


if __name__ == "__main__":
    scrape_mlb_data()
