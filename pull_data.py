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
from nba_api.stats.endpoints import commonteamroster, leaguegamelog, leaguedashteamstats

NUM_SEASONS = 3          # how many seasons back to pull
TEAM_ABBR = "LAL" #This is our target team atm (Evenetually will expand to all)
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
            time.sleep(5 * (attempt + 1))
    raise SystemExit("Couldn't reach NBA.com.")


def get_roster(team_id, season):
    res = call_api(lambda: commonteamroster.CommonTeamRoster(
        team_id=team_id, season=season, timeout=60))
    return res.get_data_frames()[0][["PLAYER_ID", "PLAYER", "POSITION", "AGE"]]


def get_league_games(season):
    """Every player's box score for every game in the season - one request."""
    res = call_api(lambda: leaguegamelog.LeagueGameLog(
        season=season, player_or_team_abbreviation="P",
        season_type_all_star="Regular Season", timeout=60))
    df = res.get_data_frames()[0]
    df["SEASON"] = season
    return df


def get_team_stats(season):
    res = call_api(lambda: leaguedashteamstats.LeagueDashTeamStats(
        season=season, measure_type_detailed_defense="Advanced", timeout=60))
    df = res.get_data_frames()[0][["TEAM_ID", "TEAM_NAME", "PACE", "DEF_RATING"]]
    df["SEASON"] = season
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    lakers = teams.find_team_by_abbreviation(TEAM_ABBR)
    start = current_season_start()
    seasons = [season_string(start - i) for i in range(NUM_SEASONS)]
    print(f"Seasons: {seasons}")

    # 1. Current LAKERS roster
    print("Getting Lakers roster...")
    roster = get_roster(lakers["id"], seasons[0])
    roster.to_csv(f"{DATA_DIR}/roster.csv", index=False)
    print(f"   {len(roster)} players")

    # 2. Game logs for each player, each season
    all_games = []
    for s in seasons:
            print(f"Pulling all games for {s}...")
            df = get_league_games(s)
            all_games.append(df)
    games = pd.concat(all_games, ignore_index=True)


    games = games.rename(columns = {"PLAYER_ID": "Player_ID", "GAME_ID": "Game_ID"})
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"], format="mixed")
    games = games[games["MIN"].notna()]
    games = games.sort_values(["Player_ID", "GAME_DATE"])

    # Handy columns for step 2
    games["HOME"] = games["MATCHUP"].str.contains("vs.").astype(int)   # "LAL vs. BOS" = home, "LAL @ BOS" = away
    games["OPP_ABBR"] = games["MATCHUP"].str.split().str[-1]
    games["TEAM_ABBR"] = games["MATCHUP"].str.split().str[0]           # players traded mid-season will show old teams

    games.to_csv(f"{DATA_DIR}/player_games.csv", index=False)
    print(f"   {len(games)} player-games saved ({games['Player_ID'].nunique()} players)")


    # 3. Team pace + defence per season (for opponent strength later)
    print("Getting team stats...")
    team_stats = pd.concat([get_team_stats(s) for s in seasons], ignore_index=True)
    abbr = {t["id"]: t["abbreviation"] for t in teams.get_teams()}
    team_stats["TEAM_ABBR"] = team_stats["TEAM_ID"].map(abbr)
    team_stats.to_csv(f"{DATA_DIR}/team_stats.csv", index=False)



    print("\nDone. Quick look:")
    print(games[["PLAYER_NAME", "GAME_DATE", "MATCHUP", "MIN", "PTS", "REB", "AST"]].tail(10))
    lakers_games = games[games["Player_ID"].isin(roster["PLAYER_ID"])]
    print(f"\nDone. {len(lakers_games)} of those games are by current Lakers players.")


if __name__ == "__main__":
    main()
