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
    by_season = games.groupby(["Player_ID", "SEASON"])
    for stat in STATS:
        games[f"{stat}_SZN"] = by_season[stat].transform(
            lambda s: s.shift(1).expanding().mean())
    # first game of a season has no season avg yet -> fall back to last-10
    for stat in STATS:
        games[f"{stat}_SZN"] = games[f"{stat}_SZN"].fillna(games[f"{stat}_L10"])


    #Third - Count associated Rest days
    gap = by_player["GAME_DATE"].diff().dt.days
    games["REST_DAYS"] = gap.clip(upper=3).fillna(3)
    games["B2B"] = (games["REST_DAYS"] == 1).astype(int)
    games["LONG_BREAK"] = ((gap >= 5) | gap.isna()).astype(int)


    #Fourth - Factor in opponents strength
    opp = team_stats[["TEAM_ABBR", "SEASON", "PACE", "DEF_RATING"]].rename(
        columns={"TEAM_ABBR": "OPP_ABBR", "PACE": "OPP_PACE", "DEF_RATING": "OPP_DEF_RATING"})
    games = games.merge(opp, on=["OPP_ABBR", "SEASON"], how= "left")


    #Fifth - Tidy
    games["PRIOR_GAMES"] = games.groupby("Player_ID").cumcount()
    games = games[games["PRIOR_GAMES"] >= MIN_PRIOR_GAMES]

    id_cols = ["Player_ID", "PLAYER_NAME", "SEASON", "GAME_DATE", "MATCHUP"]
    feature_cols = ([f"{s}_L{w}" for s in STATS for w in WINDOWS]
                    + [f"{s}_SZN" for s in STATS]
                    + ["HOME", "REST_DAYS", "B2B", "LONG_BREAK", "OPP_PACE", "OPP_DEF_RATING"])
    target_cols = ["PTS", "REB", "AST", "MIN"]

    table = games[id_cols + feature_cols + target_cols].dropna()
    table.to_csv(f"{DATA_DIR}/features.csv", index=False)

    print(f"Feature table: {len(table)} rows x {len(feature_cols)} features")
    print(table[["PLAYER_NAME", "GAME_DATE", "PTS_L5", "PTS_L10", "PTS_SZN",
                 "REST_DAYS", "OPP_DEF_RATING", "PTS"]].tail(8).to_string(index=False))

if __name__ == "__main__":
    main()

