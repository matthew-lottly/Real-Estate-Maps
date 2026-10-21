import numpy as np
import pandas as pd
import pytest

from real_estate_maps.forecast.backtest import Band, Metrics, WalkForward
from real_estate_maps.forecast.features import neighbor_mean


def synthetic(n_zip=60, start="2012-01", end="2026-05", seed=0):
    rng = np.random.default_rng(seed)
    months = pd.period_range(start, end, freq="M")
    idx = pd.MultiIndex.from_product([months, [f"{i:05d}" for i in range(n_zip)]], names=["month", "zip"])
    X = pd.DataFrame({"a": rng.normal(size=len(idx)), "b": rng.normal(size=len(idx))}, index=idx)
    Y = pd.DataFrame({"rel_12": X["a"] * 0.05 + rng.normal(scale=0.05, size=len(idx))}, index=idx)
    return X, Y, pd.Series(100, index=idx)


def test_neighbor_mean_ignores_nan_and_self():
    A = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0.0]])
    X = np.array([[1.0, 2.0, np.nan]])
    out = neighbor_mean(X, A)
    assert out[0, 0] == 2.0            # only ZIP 1 is a valid neighbor of ZIP 0
    assert out[0, 1] == 1.0
    assert out[0, 2] == 1.0


def test_walk_forward_never_trains_on_unknown_outcomes():
    X, Y, sold = synthetic()
    wf = WalkForward(X, Y, sold, first_test_year=2020, last_origin=pd.Period("2026-05", "M"))
    months = X.index.get_level_values("month")
    for yr in wf.years(12):
        tr, te = wf._fold(12, yr)
        # an outcome 12 months ahead must be known before the first test origin
        assert months[tr].max() + 12 < months[te].min()
        assert not (tr & te).any()


def test_metrics_reward_a_true_signal_and_not_noise():
    X, Y, sold = synthetic()
    y = Y["rel_12"]
    good = Metrics.of(X["a"], y, sold).summary()
    bad = Metrics.of(pd.Series(np.random.default_rng(1).normal(size=len(y)), index=y.index), y, sold).summary()
    assert good["rho"] > 0.3 and abs(bad["rho"]) < 0.08
    assert good["top10"] > bad["top10"]


def test_conformal_band_covers_roughly_80_percent():
    X, Y, sold = synthetic(seed=3)
    y = Y["rel_12"]
    months = X.index.get_level_values("month")
    tr = np.asarray(months <= pd.Period("2022-12", "M"))
    te = np.asarray(months >= pd.Period("2024-01", "M"))
    band = Band(X, y, tr, ["a", "b"])
    lo, hi = band.predict(X[te])
    cover = float(((y[te].to_numpy() >= lo) & (y[te].to_numpy() <= hi)).mean())
    assert 0.7 < cover < 0.9
