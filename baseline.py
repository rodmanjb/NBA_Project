import pandas as pd

DATA_DIR = "data"
TARGETS = ["PTS", "REB", "AST"]
TEST_FRAC = 0.2 #to keep the recent 20% to test

def time_split(df, test_frac=TEST_FRAC):
    cutoff = df["GAME_DATA"].quantile(1-test_frac)
    return df[df["GAME_DATA"] < cutoff], df[df["GAME_DATA"] >= cutoff], cutoff

def mae(actual, pred):
    return (actual - pred).abs().mean()

def main():
    df = pd.read_csv(f"{DATA_DIR}/features.csv", parse_dates=["GAME_DATE"])
    train, test, cutoff = time_split(df)
    print(f"Train: {len(train)} games | Test: {len(test)} games (from {cutoff.date()} on)\n")

    baselines = {"Last 5": "_L5", "Last 10": "_L10", "Season avg": "_SZN"}

    rows = []
    for name, suffix in baselines.items():
        row = {"Baseline": name}
        for t in TARGETS:
            row[t] = mae(test[t], test[t + suffix])
        rows.append(row)
    results = pd.DataFrame(rows).set_index("Baseline").round(2)

    print("Mean Absolute Error (lower = better)")
    print("e.g. PTS 5.1 = on average the guess is off by 5.1 points\n")
    print(results.to_string())

    best = results.idxmin()
    print("\nBest baseline per stat (the number to beat):")
    for t in TARGETS:
        print(f"  {t}: {best[t]} -> {results.loc[best[t], t]}")

    # Per-player breakdown using last-10 - shows who's easy/hard to predict
    print("\nPer player (Last 10 baseline, PTS):")
    per_player = (test.assign(err=(test["PTS"] - test["PTS_L10"]).abs())
                      .groupby("PLAYER_NAME")["err"].agg(["mean", "count"])
                      .rename(columns={"mean": "PTS MAE", "count": "games"})
                      .sort_values("PTS MAE").round(2))
    print(per_player.to_string())

    results.to_csv(f"{DATA_DIR}/baseline_results.csv")


if __name__ == "__main__":
    main()