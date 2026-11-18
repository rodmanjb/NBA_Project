import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_DIR = "data"
TARGETS = ["PTS", "REB", "AST"]
TEST_FRACTION = 0.2

FEATURES = (
    [f"{s}_L{w}" for s in ["PTS", "REB", "AST", "MIN"] for w in [5, 10]]
    + [f"{s}_SZN" for s in ["PTS", "REB", "AST", "MIN"]]
    + ["HOME", "REST_DAYS", "B2B", "LONG_BREAK", "OPP_PACE", "OPP_DEF_RATING"]
)


def time_split(df, test_fraction=TEST_FRACTION):
    """Same split as baseline.py - train on earlier games, test on later ones."""
    cutoff = df["GAME_DATE"].quantile(1 - test_fraction)
    return df[df["GAME_DATE"] < cutoff], df[df["GAME_DATE"] >= cutoff]


def mae(actual, predicted):
    return (actual - predicted).abs().mean()


def main():
    df = pd.read_csv(f"{DATA_DIR}/features.csv", parse_dates=["GAME_DATE"])
    train, test = time_split(df)
    X_train, X_test = train[FEATURES], test[FEATURES]
    print(f"Train: {len(train)} games | Test: {len(test)} games\n")

    results = {}
    linear_models = {}

    for t in TARGETS:
        y_train, y_test = train[t], test[t]

        # Baseline to beat (last 10 avg)
        base = mae(y_test, test[f"{t}_L10"])

        # 1. Linear regression
        lin = LinearRegression().fit(X_train, y_train)
        lin_mae = mae(y_test, lin.predict(X_test))
        linear_models[t] = lin

        # 2. Gradient boosting (kept small so it doesn't overfit)
        gb = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=300, random_state=0)
        gb.fit(X_train, y_train)
        gb_mae = mae(y_test, gb.predict(X_test))

        results[t] = {"Baseline (L10)": base, "Linear": lin_mae, "Boosting": gb_mae}

    table = pd.DataFrame(results).round(2)
    print("Test MAE (lower = better)\n")
    print(table.to_string())

    print("\nImprovement over baseline:")
    for t in TARGETS:
        best_model = table[t].drop("Baseline (L10)").idxmin()
        gain = table.loc["Baseline (L10)", t] - table.loc[best_model, t]
        verdict = "beats baseline" if gain > 0 else "does NOT beat baseline"
        print(f"  {t}: {best_model} {verdict} by {gain:+.2f}")

    # Show the actual equation for points
    lin = linear_models["PTS"]
    coefs = pd.Series(lin.coef_, index=FEATURES).sort_values(key=abs, ascending=False)
    print("\nYour PTS equation (biggest effects first):")
    print(f"  PTS = {lin.intercept_:.2f}")
    for name, c in coefs.items():
        print(f"        {c:+.3f} x {name}")

    table.to_csv(f"{DATA_DIR}/model_results.csv")


if __name__ == "__main__":
    main()
