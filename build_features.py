import pandas as pd
DATA_DIR = "data"
STATS = ["PTS", "REB", "AST", "MIN"]
WINDOWS = [5, 10]
MIN_PRIOR_GAMES = 5 


#Function for turning minutes into a consistsant format
def to_minutes(m): 
    if isinstance(m, str) and ":" in m:
        mins, secs = m.split(":")
        return int(mins) + int(secs)/60
    return float(m)

def main():
    games = pd.read_csv(f"{DATA_DIR}/player_games.csv", parse_dates=["GAME_DATE"])
    team_stats = pd.read_csv(f"{DATA_DIR}/team_stats.csv")

    games["MIN"] = games["MIN"].apply(to_minutes)
    games = games.sort_values(["Player_ID", "GAME_DATE"]).reset_index(drop=True)
    by_player = games.groupby("Player_ID")


    #First - Calculate the recent form of each player (Rolling averages per stat over last 5 & 10 Games)
    for stat in STATS:
        for w in WINDOWS:
          games[f"{stat}_L{w}"] = by_player[stat].transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean())


    #Second - Season averages
    by_season = games.groupby(["PLAYER_ID", "SEASON"])
    for stat in STATS:
        games[f"{stat}_L{w}"] = by_season[stat].transform(
            lambda s: s.shift(1).expanding.mean())
    # first game of a season has no season avg yet -> fall back to last-10
    for stat in STATS:
        games[f"{stat}_SZN"] = games[f"{stat}_SZN"].fillna(games[f"{stat}_L10"])


    #Third - Count associated Rest days
    games["REST_DAYS"] = by_player["GAME_DATE"].diff.dt.days
    games["REST_DAYS"] = games["REST_DAYS"].clip(upper=7).fillna(7) #a long break/season opener counts as 7
    games["B2B"] = (games["REST_DAYS"] == 1).astype(int)


    #Fourth - Factor in opponents strength


    #Fifth - Tidy


    if __name__ == "__main__":
        main()

