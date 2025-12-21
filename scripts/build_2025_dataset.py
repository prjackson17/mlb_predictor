""" Build a dataset of MLB games from 2015 to 2025 for ML modeling. """

import statsapi
import pandas as pd
import time
from requests.exceptions import HTTPError


def extract_pitcher_stats(box_data, team_id):
    """Pull starting pitcher stats."""
    try:
        for p in box_data['teamInfo'][str(team_id)]['players']:
            player = box_data['teamInfo'][str(team_id)]['players'][p]
            if player.get('position') == 'P' and player.get('gameStatus') == 'Starter':
                stats = player['stats'].get('pitching', {})
                return {
                    'starter_id': player['personId'],
                    'starter_name': player['fullName'],
                    'starter_ip': stats.get('inningsPitched'),
                    'starter_era': stats.get('era'),
                    'starter_whip': stats.get('whip'),
                    'starter_strikeouts': stats.get('strikeOuts'),
                }
    except:
        return {
            'starter_id': None,
            'starter_name': None,
            'starter_ip': None,
            'starter_era': None,
            'starter_whip': None,
            'starter_strikeouts': None,
        }


def extract_team_batting(box_data, team_id):
    """Total team batting stats for the game."""
    try:
        batting = box_data['teams'][str(team_id)]['teamStats']['batting']
        return {
            'hits': batting.get('hits'),
            'runs': batting.get('runs'),
            'home_runs': batting.get('homeRuns'),
            'strikeouts': batting.get('strikeOuts'),
            'ops': batting.get('ops'),
        }
    except:
        return {
            'hits': None,
            'runs': None,
            'home_runs': None,
            'strikeouts': None,
            'ops': None,
        }


def extract_team_pitching(box_data, team_id):
    """Total pitching stats for the game."""
    try:
        pitching = box_data['teams'][str(team_id)]['teamStats']['pitching']
        return {
            'earned_runs': pitching.get('earnedRuns'),
            'strikeouts': pitching.get('strikeOuts'),
            'walks': pitching.get('baseOnBalls'),
            'era': pitching.get('era'),
            'whip': pitching.get('whip'),
        }
    except:
        return {
            'earned_runs': None,
            'strikeouts': None,
            'walks': None,
            'era': None,
            'whip': None,
        }


def scrape_2025():
    rows = []
    team_stats_history = {}  # Stores running totals (runs, hits, games_played)

    # Scrape from 2015 to 2025
    for year in range(2015, 2026):
        start_date = f"03/20/{year}"
        end_date = f"11/15/{year}"  # Extended to cover postseason

        max_retries = 5
        retry_count = 0
        schedule = None

        # Retry logic for getting schedule
        while retry_count < max_retries and schedule is None:
            try:
                schedule = statsapi.schedule(start_date=start_date, end_date=end_date)
            except HTTPError:
                retry_count += 1
                wait_time = retry_count * 10  # Exponential backoff
                print(f"Error getting schedule for {year}, retry {retry_count}/{max_retries} after {wait_time}s...")
                time.sleep(wait_time)
                if retry_count == max_retries:
                    print(f"Failed to get schedule for {year} after {max_retries} retries. Skipping year.")
                    break

        if schedule is None:
            continue

        for game in schedule:
            # Skip games that haven't been completed
            if game['status'] != 'Final':
                continue

            home_id = game['home_id']
            away_id = game['away_id']

            # 1. RETRIEVE FEATURES (What we know BEFORE the game)
            home_season_stats = team_stats_history.get(home_id, {'runs': 0, 'games': 0})
            away_season_stats = team_stats_history.get(away_id, {'runs': 0, 'games': 0})

            home_avg_runs = home_season_stats['runs'] / max(1, home_season_stats['games'])
            away_avg_runs = away_season_stats['runs'] / max(1, away_season_stats['games'])

            # 2. RECORD THE ROW FOR ML
            row = {
                "year": year,
                "game_date": game['game_date'],
                "home_team": home_id,
                "away_team": away_id,
                "home_avg_runs_coming_in": home_avg_runs,  # VALID FEATURE
                "away_avg_runs_coming_in": away_avg_runs,  # VALID FEATURE
                "home_score": game['home_score'],  # TARGET
                "away_score": game['away_score'],  # TARGET
                "home_win": 1 if game['home_score'] > game['away_score'] else 0  # TARGET
            }
            rows.append(row)

            # 3. UPDATE HISTORY (What we know AFTER the game)
            # Retry logic for box score
            retry_count = 0
            box = None
            while retry_count < max_retries and box is None:
                try:
                    box = statsapi.boxscore(game['game_id'])
                except HTTPError:
                    retry_count += 1
                    wait_time = retry_count * 5
                    time.sleep(wait_time)
                    if retry_count == max_retries:
                        # Use game scores if we can't get boxscore
                        home_bat = {'runs': game['home_score']}
                        away_bat = {'runs': game['away_score']}
                        break

            if box is not None:
                home_bat = extract_team_batting(box, home_id)
                away_bat = extract_team_batting(box, away_id)

            # Update the dictionary for the next time this team plays
            if home_id not in team_stats_history:
                team_stats_history[home_id] = {'runs': 0, 'games': 0}
            if away_id not in team_stats_history:
                team_stats_history[away_id] = {'runs': 0, 'games': 0}

            # Handle None values by defaulting to 0
            team_stats_history[home_id]['runs'] += home_bat['runs'] or 0
            team_stats_history[home_id]['games'] += 1

            team_stats_history[away_id]['runs'] += away_bat['runs'] or 0
            team_stats_history[away_id]['games'] += 1

            time.sleep(0.3)  # Increased delay to avoid hitting API rate limits

        # Reset stats history at the start of each new season
        team_stats_history = {}

        # Save after each year and print progress
        df = pd.DataFrame(rows)
        df.to_csv("mlb_2015_2025_dataset.csv", index=False)
        print(f"Completed {year} season - Total games: {len(rows)}")

    print(f"\nFinal dataset saved to mlb_2015_2025_dataset.csv with {len(rows)} games")


scrape_2025()
