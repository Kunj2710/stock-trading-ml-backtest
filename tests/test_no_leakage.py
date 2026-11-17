"""Guards against the look-ahead bug the original notebooks had."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from honest_backtest import backtest, load_weekly  # noqa: E402

HONEST = ["mu_lag1", "sigma_lag1", "mu_lag2", "ret_4w"]


def test_lagged_features_use_only_past_weeks():
    df = load_weekly()
    assert np.allclose(df["mu_lag1"].iloc[1:].to_numpy(), df["mean_return"].iloc[:-1].to_numpy())
    assert np.allclose(df["sigma_lag1"].iloc[1:].to_numpy(), df["volatility"].iloc[:-1].to_numpy())


def test_momentum_feature_ends_at_previous_week():
    df = load_weekly()
    r = df["week_return"].to_numpy()
    for t in range(10, 20):
        expected = np.prod(1 + r[t - 4 : t]) - 1
        assert np.isclose(df["ret_4w"].iloc[t], expected)


def test_costs_reduce_returns():
    r = np.array([0.01, -0.02, 0.03, 0.01])
    pos = np.array([1, 0, 1, 0])
    assert backtest(pos, r, 10)["final"] < backtest(pos, r, 0)["final"]


def test_labels_and_returns_aligned():
    df = load_weekly()
    assert df["week_return"].notna().all() and set(df["y"].unique()) <= {0, 1}
