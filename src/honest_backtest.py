"""Leakage-free, cost-aware walk-forward backtest of the weekly hold/cash classifiers.

Why this file exists
--------------------
The original notebooks (04-14) predict week t's label from week t's own mean
return and volatility. Those features are only known once week t is over, so
the model is effectively shown the answer before it trades. This script fixes
that and adds the checks a markets or risk reviewer would ask for:

* Features for week t use only data available at the close of week t-1.
* Walk-forward (expanding window): the model is refit before every test week
  using all weeks up to and including t-1.
* Transaction costs charged every time the strategy switches in or out.
* Sharpe ratio, max drawdown, 95% weekly VaR, time in market, trade count.
* A random-strategy benchmark with the same time in market, so a "win" can be
  judged against luck.
* The original same-week (leaky) setup is re-run side by side so the size of
  the leakage effect is visible.

Run from the repo root:  python src/honest_backtest.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
WEEKS_PER_YEAR = 52


def load_weekly(root: Path = ROOT) -> pd.DataFrame:
    """One row per week: label, same-week features, lagged features, realised return."""
    daily = pd.read_csv(root / "KO_weekly_return_detailed.csv")
    labels = pd.read_csv(root / "KO_weekly_return_volatility.csv")

    daily["r"] = daily["Return"].fillna(0.0) / 100.0
    weekly_ret = (
        daily.groupby(["Year", "Week_Number"], sort=False)["r"]
        .apply(lambda s: float(np.prod(1.0 + s) - 1.0))
        .rename("week_return")
        .reset_index()
    )
    df = labels.merge(weekly_ret, on=["Year", "Week_Number"], how="inner")
    df["y"] = (df["decision"] == "YES").astype(int)

    # Same-week features: what the original notebooks used (leaky).
    df["mu_t"] = df["mean_return"]
    df["sigma_t"] = df["volatility"]

    # Lagged features: everything known at the close of week t-1.
    df["mu_lag1"] = df["mean_return"].shift(1)
    df["sigma_lag1"] = df["volatility"].shift(1)
    df["mu_lag2"] = df["mean_return"].shift(2)
    df["ret_4w"] = (1.0 + df["week_return"]).rolling(4).apply(np.prod, raw=True).shift(1) - 1.0
    return df.dropna().reset_index(drop=True)


def models() -> dict:
    return {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.1),
        "Gaussian Naive Bayes": GaussianNB(),
        "kNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Decision tree (entropy)": DecisionTreeClassifier(criterion="entropy", random_state=42),
        "Decision tree (Gini)": DecisionTreeClassifier(criterion="gini", random_state=42),
        "Random forest (N=10, d=5)": RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42),
        "Linear SVM": make_pipeline(StandardScaler(), SVC(kernel="linear")),
    }


def walk_forward(df: pd.DataFrame, model, features: list[str], test_mask: np.ndarray) -> np.ndarray:
    """Refit on all rows before each test week, predict that week only."""
    preds = np.zeros(len(df), dtype=int)
    X, y = df[features].to_numpy(), df["y"].to_numpy()
    for i in np.flatnonzero(test_mask):
        m = clone(model)
        m.fit(X[:i], y[:i])
        preds[i] = int(m.predict(X[i : i + 1])[0])
    return preds


def backtest(positions: np.ndarray, week_returns: np.ndarray, cost_bps: float) -> dict:
    """positions[t] in {0,1} is held during week t. Cost charged per switch, per side."""
    pos = positions.astype(float)
    switches = np.abs(np.diff(np.concatenate([[0.0], pos])))
    strat = pos * week_returns - switches * cost_bps / 1e4
    equity = 100.0 * np.cumprod(1.0 + strat)
    peak = np.maximum.accumulate(np.concatenate([[100.0], equity]))[1:]
    drawdown = equity / peak - 1.0
    vol = strat.std(ddof=1)
    return {
        "final": equity[-1],
        "sharpe": np.nan if vol == 0 else strat.mean() / vol * np.sqrt(WEEKS_PER_YEAR),
        "max_dd": drawdown.min(),
        "var95": -np.quantile(strat, 0.05),
        "in_market": pos.mean(),
        "trades": int(switches.sum()),
        "returns": strat,
    }


def random_benchmark(exposure: float, week_returns: np.ndarray, cost_bps: float, n: int, seed: int) -> np.ndarray:
    """Final balances of n random strategies with the same number of invested weeks."""
    rng = np.random.default_rng(seed)
    k = int(round(exposure * len(week_returns)))
    finals = np.empty(n)
    for j in range(n):
        pos = np.zeros(len(week_returns))
        pos[rng.choice(len(week_returns), size=k, replace=False)] = 1
        finals[j] = backtest(pos, week_returns, cost_bps)["final"]
    return finals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-bps", type=float, default=5.0, help="cost per switch, per side, in basis points")
    ap.add_argument("--n-random", type=int, default=2000)
    args = ap.parse_args()

    df = load_weekly()
    test_mask = df["Year"].isin([2024, 2025]).to_numpy()
    r_test = df.loc[test_mask, "week_return"].to_numpy()

    leaky = ["mu_t", "sigma_t"]
    honest = ["mu_lag1", "sigma_lag1", "mu_lag2", "ret_4w"]

    bh = backtest(np.ones(test_mask.sum()), r_test, cost_bps=0.0)
    bh["trades"] = 1
    rows = [{
        "Strategy": "Buy-and-hold",
        "Acc (leaky)": np.nan, "Final (leaky)": np.nan,
        "Accuracy": np.nan, "Final": bh["final"], "Sharpe": bh["sharpe"],
        "Max DD": bh["max_dd"], "VaR95 (wk)": bh["var95"], "In market": 1.0,
        "Trades": 1, "Random pctile": np.nan,
    }]

    for name, model in models().items():
        p_leak = walk_forward(df, model, leaky, test_mask)[test_mask]
        p_hon = walk_forward(df, model, honest, test_mask)[test_mask]
        y_test = df.loc[test_mask, "y"].to_numpy()
        leak = backtest(p_leak, r_test, args.cost_bps)
        hon = backtest(p_hon, r_test, args.cost_bps)
        if 0 < hon["in_market"] < 1:
            rand = random_benchmark(hon["in_market"], r_test, args.cost_bps, args.n_random, seed=0)
            pct = (rand < hon["final"]).mean()
        else:
            pct = np.nan
        rows.append({
            "Strategy": name,
            "Acc (leaky)": (p_leak == y_test).mean(), "Final (leaky)": leak["final"],
            "Accuracy": (p_hon == y_test).mean(), "Final": hon["final"], "Sharpe": hon["sharpe"],
            "Max DD": hon["max_dd"], "VaR95 (wk)": hon["var95"], "In market": hon["in_market"],
            "Trades": hon["trades"], "Random pctile": pct,
        })

    out = pd.DataFrame(rows)
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out.to_csv(out_dir / "honest_backtest.csv", index=False)

    fmt = out.copy()
    for c in ["Acc (leaky)", "Accuracy", "Max DD", "VaR95 (wk)", "In market", "Random pctile"]:
        fmt[c] = fmt[c].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
    for c in ["Final (leaky)", "Final"]:
        fmt[c] = fmt[c].map(lambda v: "" if pd.isna(v) else f"${v:.2f}")
    fmt["Sharpe"] = fmt["Sharpe"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    print(f"Test weeks: {test_mask.sum()} | cost: {args.cost_bps} bps per switch")
    print(fmt.to_markdown(index=False))
    plot_best(df, honest, test_mask, r_test, args.cost_bps, args.n_random)


def plot_best(df, features, test_mask, r_test, cost_bps, n_random) -> None:
    """Equity curve of the best honest model vs buy-and-hold, and where it falls among random strategies."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = walk_forward(df, models()["kNN (k=5)"], features, test_mask)[test_mask]
    knn = backtest(p, r_test, cost_bps)
    bh = backtest(np.ones(len(r_test)), r_test, 0.0)
    rand = random_benchmark(knn["in_market"], r_test, cost_bps, n_random, seed=0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(100 * np.cumprod(1 + bh["returns"]), label="Buy-and-hold", color="#444")
    ax[0].plot(100 * np.cumprod(1 + knn["returns"]), label="kNN (best honest model)", color="#1f77b4")
    ax[0].set(title=f"Equity, 2024-25 test weeks ({cost_bps:g} bps costs)", xlabel="Test week", ylabel="$")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].hist(rand, bins=50, color="#bbb")
    ax[1].axvline(knn["final"], color="#1f77b4", lw=2, label=f"kNN ${knn['final']:.0f}")
    ax[1].axvline(bh["final"], color="#444", ls="--", label=f"Buy-and-hold ${bh['final']:.0f}")
    ax[1].set(title=f"{n_random:,} random strategies with the same time in market", xlabel="Final balance ($)")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "honest_backtest.png", dpi=140)


if __name__ == "__main__":
    main()
