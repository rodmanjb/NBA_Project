"""
Step 1: Pull game logs for current Lakers players + team stats for opponents.

Setup:
    pip install nba_api pandas

Run:
    python pull_data.py

Output (in ./data/):
    roster.csv          - current Lakers roster
    player_games.csv    - every game for each player over the last N seasons
    team_stats.csv      - each team's pace + defensive rating per season (for opponent features later)
"""

import os
import time
from datetime import date

import pandas as pd
from nba_api.stats.static import teams
from nba_api.stats.endpoints import commonteamroster, playergamelog, leaguedashteamstats

NUM_SEASONS = 3          # how many seasons back to pull
TEAM_ABBR = "LAL"
DATA_DIR = "data"
PAUSE = 0.8              # seconds between API calls (NBA.com blocks you if you spam it)


def season_string(start_year):
    """2025 -> '2025-26'"""
    return f"{start_year}-{str(start_year + 1)[2:]}"


def current_season_start():
    """NBA season starts in October, so Oct-Dec counts as the new season."""
    today = date.today()
    return today.year if today.month >= 10 else today.year - 1


def call_api(fn, retries=3):
    """NBA.com times out a lot - retry a few times before giving up."""
    for attempt in range(retries):
        try:
            result = fn()
            time.sleep(PAUSE)
            return result
        except Exception as e:
            print(f"   attempt {attempt + 1} failed: {e}")
            time.sleep(3 * (attempt + 1))
    return None


def get_roster(team_id, season):
    res = call_api(lambda: commonteamroster.CommonTeamRoster(
        team_id=team_id, season=season, timeout=30))
    if res is None:
        raise SystemExit("Couldn't reach NBA.com - check your internet / try again in a minute.")
    df = res.get_data_frames()[0]
    return df[["PLAYER_ID", "PLAYER", "POSITION", "AGE"]]


def get_player_games(player_id, season):
    res = call_api(lambda: playergamelog.PlayerGameLog(
        player_id=player_id, season=season,
        season_type_all_star="Regular Season", timeout=30))
    if res is None:
        return pd.DataFrame()
    return res.get_data_frames()[0]


def get_team_stats(season):
    res = call_api(lambda: leaguedashteamstats.LeagueDashTeamStats(
        season=season, measure_type_detailed_defense="Advanced", timeout=30))
    if res is None:
        return pd.DataFrame()
    df = res.get_data_frames()[0][["TEAM_ID", "TEAM_NAME", "PACE", "DEF_RATING"]]
    df["SEASON"] = season
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    lakers = teams.find_team_by_abbreviation(TEAM_ABBR)
    start = current_season_start()
    seasons = [season_string(start - i) for i in range(NUM_SEASONS)]
    print(f"Seasons: {seasons}")

    # 1. Current roster
    print("Getting roster...")
    roster = get_roster(lakers["id"], seasons[0])
    roster.to_csv(f"{DATA_DIR}/roster.csv", index=False)
    print(f"   {len(roster)} players")

    # 2. Game logs for each player, each season
    all_games = []
    for _, p in roster.iterrows():
        for s in seasons:
            print(f"Pulling {p['PLAYER']} {s}...")
            df = get_player_games(p["PLAYER_ID"], s)
            if df.empty:
                continue  # rookie / didn't play that season
            df["PLAYER_NAME"] = p["PLAYER"]
            df["SEASON"] = s
            all_games.append(df)

    games = pd.concat(all_games, ignore_index=True)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"], format="mixed")
    games = games.sort_values(["Player_ID", "GAME_DATE"])

    # Handy columns for step 2
    games["HOME"] = games["MATCHUP"].str.contains("vs.").astype(int)   # "LAL vs. BOS" = home, "LAL @ BOS" = away
    games["OPP_ABBR"] = games["MATCHUP"].str.split().str[-1]
    games["TEAM_ABBR"] = games["MATCHUP"].str.split().str[0]           # players traded mid-season will show old teams

    games.to_csv(f"{DATA_DIR}/player_games.csv", index=False)
    print(f"   {len(games)} player-games saved")

    # 3. Team pace + defence per season (for opponent strength later)
    print("Getting team stats...")
    team_stats = pd.concat([get_team_stats(s) for s in seasons], ignore_index=True)
    abbr = {t["id"]: t["abbreviation"] for t in teams.get_teams()}
    team_stats["TEAM_ABBR"] = team_stats["TEAM_ID"].map(abbr)
    team_stats.to_csv(f"{DATA_DIR}/team_stats.csv", index=False)

    print("\nDone. Quick look:")
    print(games[["PLAYER_NAME", "GAME_DATE", "MATCHUP", "MIN", "PTS", "REB", "AST"]].tail(10))


if __name__ == "__main__":
    main()
