"""python -m pytest test_obp.py   (synthetic data, no network)"""
import numpy as np
import pandas as pd
import pytest

import obp


def synthetic_returns(T=260, n=8, seed=0):
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(n, 3))
    f = rng.normal(0, 0.01, size=(T, 3))
    return 1 + 0.0004 + 0.7 * f @ B.T + rng.normal(0, 0.012, size=(T, n))


@pytest.fixture
def window():
    R = synthetic_returns()
    return np.cov(R, rowvar=False), R.mean(axis=0)


def test_principal_portfolios_are_uncorrelated_and_sum_to_one(window):
    Sigma, _ = window
    L, Ht, lt = obp.principal_portfolios(Sigma)
    assert np.all(np.diff(L) <= 0)
    assert np.allclose(Ht.sum(axis=0), 1)
    C = Ht.T @ Sigma @ Ht                                                   # eq. (8)
    assert np.allclose(C, np.diag(lt), atol=1e-10 * lt.max())


def test_reward_is_the_sharpe_ratio_of_the_portfolio(window):
    """The first version divided by sqrt(lambda) instead of sqrt(lambda tilde): wrong by 1 / |H_i^T 1|."""
    Sigma, mu = window
    L, Ht, lt = obp.principal_portfolios(Sigma)
    r = obp.sharpe_rewards(Ht, lt, mu)
    for i in range(len(L)):
        w = Ht[:, i]
        assert r[i] == pytest.approx((w @ mu) / np.sqrt(w @ Sigma @ w))
    old = Ht.T @ mu / np.sqrt(L)
    assert not np.allclose(old, r)


def test_reward_does_not_depend_on_the_sign_of_the_eigenvectors(window, monkeypatch):
    Sigma, mu = window
    _, Ht, lt = obp.principal_portfolios(Sigma)
    eigh = np.linalg.eigh

    def flipped(a):
        L, H = eigh(a)
        return L, -H

    monkeypatch.setattr(np.linalg, "eigh", flipped)
    _, Ht2, lt2 = obp.principal_portfolios(Sigma)
    assert np.allclose(obp.sharpe_rewards(Ht, lt, mu), obp.sharpe_rewards(Ht2, lt2, mu))


def test_inputs_are_not_modified(window):
    Sigma, mu = window
    S0, mu0 = Sigma.copy(), mu.copy()
    _, Ht, lt = obp.principal_portfolios(Sigma)
    obp.sharpe_rewards(Ht, lt, mu)
    assert np.array_equal(Sigma, S0) and np.array_equal(mu, mu0)


def test_theta_minimizes_the_variance_of_the_mix(window):
    Sigma, _ = window
    _, Ht, lt = obp.principal_portfolios(Sigma)
    theta, w = obp.mix(Ht, lt, 0, 5)
    grid = np.linspace(0, 1, 2001)
    var = [(lambda v: v @ Sigma @ v)((1 - t) * Ht[:, 0] + t * Ht[:, 5]) for t in grid]
    assert theta == pytest.approx(grid[int(np.argmin(var))], abs=1e-3)
    assert w.sum() == pytest.approx(1)


def test_ucb_index():
    rewards = np.array([1.00, 1.05])
    assert obp.ucb_select(rewards, np.array([0.0, 0.0]), 1, 100) == 1
    # an arm chosen very often loses its confidence bound
    assert obp.ucb_select(rewards, np.array([0.0, 1e9]), 1, 100) == 0
    bonus = np.sqrt(2 * np.log(1 + 100) / (100 + 0))
    assert bonus > 0.05


def test_no_short_sale():
    w = obp.no_short_sale(np.array([1.5, -0.7, 0.2]))
    assert np.all(w >= 0) and w.sum() == pytest.approx(1) and w[1] == 0
    assert w[0] / w[2] == pytest.approx(7.5)


def test_check_raises():
    with pytest.raises(RuntimeError):
        obp.check(abs(np.sum([0.1, 0.1]) - 1) < 1e-8, "weights")            # the old assert let this pass
    with pytest.raises(RuntimeError):
        obp.principal_portfolios(np.ones((3, 3)))                           # singular covariance


def test_backtest_trades_every_period_after_the_first_window():
    R = synthetic_returns()
    out = obp.backtest(R, tau=200, l=3)
    assert len(out["obp"]) == len(R) - 200                                   # the first version lost the last one
    assert np.allclose(out["ew"], R[200:].mean(axis=1) - 1)
    assert np.allclose(out["obp"], np.sum(out["weights"] * R[200:], axis=1) - 1)
    assert np.allclose(out["weights"].sum(axis=1), 1)
    assert all(i < 3 <= j for i, j in out["chosen"])


def test_no_look_ahead():
    R = synthetic_returns()
    other = R.copy()
    other[230:] = synthetic_returns(seed=5)[230:]
    a, b = obp.backtest(R, 200, 3), obp.backtest(other, 200, 3)
    assert np.allclose(a["weights"][:31], b["weights"][:31])                 # periods 200 .. 230 see only the past
    assert not np.allclose(a["weights"][31:], b["weights"][31:])


def test_performance():
    s = obp.performance([0.1, -0.05, 0.02])
    r = np.array([0.1, -0.05, 0.02])
    assert s["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1))
    assert s["wealth"] == pytest.approx(1.1 * 0.95 * 1.02)


def test_parameters_are_validated():
    R = synthetic_returns()
    for tau, l in ((200, 0), (200, 8), (5, 3), (10_000, 3)):
        with pytest.raises(ValueError):
            obp.backtest(R, tau, l)


def fake_close():
    """BBB has no price on the third day."""
    days = pd.date_range("2020-01-06", periods=5, freq="B")
    return pd.DataFrame({"AAA": [10.0, 11.0, 12.0, 13.0, 14.0], "BBB": [20.0, 21.0, np.nan, 23.0, 24.0]}, index=days)


def test_prices_keep_the_order_of_the_tickers_and_lose_no_move():
    p = obp.align_prices(fake_close(), ["BBB", "AAA"])
    assert list(p.columns) == ["BBB", "AAA"] and len(p) == 5
    assert np.allclose(p["BBB"], [20, 21, 21, 23, 24])
    assert obp.gross_returns(p)["BBB"].prod() == pytest.approx(24 / 20)     # the move across the hole is not lost


def test_missing_data_are_reported():
    with pytest.raises(ValueError, match="CCC"):
        obp.align_prices(fake_close(), ["AAA", "CCC"])
    days = pd.date_range("2020-01-06", periods=12, freq="B")
    gap = pd.DataFrame({"AAA": [1.0] * 12, "DDD": [1.0, 1.0] + [np.nan] * 9 + [1.0]}, index=days)
    with pytest.raises(ValueError, match="DDD"):
        obp.align_prices(gap, ["AAA", "DDD"])
    with pytest.raises(ValueError, match="currency"):
        obp.load_prices(["VOW.DE"], "2020-01-06", "2020-01-11", download=lambda *a: fake_close())


def test_prices_are_cached(tmp_path):
    cache = tmp_path / "p.csv"
    calls = []

    def download(symbols, start, end):
        calls.append(symbols)
        return fake_close()

    a = obp.load_prices(["AAA", "BBB"], "2020-01-06", "2020-01-11", cache=cache, download=download)
    b = obp.load_prices(["AAA", "BBB"], "2020-01-06", "2020-01-11", cache=cache, download=download)
    assert calls == [["AAA", "BBB"]]
    assert np.allclose(a, b)
    with pytest.raises(ValueError, match="CCC"):
        obp.load_prices(["CCC"], "2020-01-06", "2020-01-11", cache=cache, download=download)
    with pytest.raises(ValueError, match="requested period"):
        obp.load_prices(["AAA"], "2019-01-01", "2020-01-11", cache=cache, download=download)


# ---------------------------------------------------------------- the estimators of the paper
def factor_data(T=400, n=6, K=3, seed=3):
    rng = np.random.default_rng(seed)
    f = rng.normal(0, [0.012, 0.006, 0.005], size=(T, K))
    B = rng.normal(1, 0.5, size=(n, K))
    idio = rng.uniform(0.008, 0.02, n)
    return f @ B.T + rng.normal(0, idio, size=(T, n)), f, B, idio


def test_factor_covariance():
    """Eq. (1.4) of Fan, Fan and Lv, written independently."""
    y, f, B, idio = factor_data()
    S = obp.factor_covariance(y, f)
    assert np.allclose(S, S.T) and np.all(np.linalg.eigvalsh(S) > 0)
    Y, X, n = y.T, f.T, len(f)                                               # the notation of the paper
    Bhat = Y @ X.T @ np.linalg.inv(X @ X.T)
    E = Y - Bhat @ X
    one = np.ones((n, 1))
    cov_f = X @ X.T / (n - 1) - X @ one @ one.T @ X.T / (n * (n - 1))
    assert np.allclose(S, Bhat @ cov_f @ Bhat.T + np.diag(np.diag(E @ E.T / n)))
    off = ~np.eye(6, dtype=bool)
    assert not np.allclose(S[off], np.cov(y.T)[off])


def test_factor_covariance_is_close_to_the_truth():
    y, f, B, idio = factor_data(T=100_000)
    true = B @ np.diag(np.array([0.012, 0.006, 0.005]) ** 2) @ B.T + np.diag(idio**2)
    assert np.allclose(obp.factor_covariance(y, f), true, rtol=0.05, atol=2e-6)


def test_james_stein_formula():
    rng = np.random.default_rng(1)
    x = 1 + rng.normal(0.001, 0.02, size=(100, 8))
    est, alpha = obp.james_stein_mean(x)
    m = x.mean(0)
    S = np.cov(x.T)
    lam = np.linalg.eigvalsh(S)
    raw = (8 * lam.mean() - 2 * lam.max()) / (100 * (m - 1) @ (m - 1))      # eq. (4.139)
    assert alpha == pytest.approx(min(max(raw, 0), 1)) and alpha > 0
    assert np.allclose(est, (1 - alpha) * m + alpha)                         # eq. (4.138), b = 1
    assert np.linalg.norm(est - 1) <= np.linalg.norm(m - 1)
    net, a2 = obp.james_stein_mean(x - 1, target=0.0)                        # net returns and b = 0: the same
    assert a2 == pytest.approx(alpha) and np.allclose(net, est - 1)


def test_james_stein_does_not_shrink_with_a_dominant_eigenvalue():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.05, size=(200, 1)) @ np.ones((1, 5)) + rng.normal(0, 0.001, size=(200, 5))
    est, alpha = obp.james_stein_mean(x)
    assert alpha == 0 and np.allclose(est, x.mean(0))


def test_backtest_with_the_paper_estimators_and_no_look_ahead():
    y, f, _, _ = factor_data(T=260)
    R = 1 + y
    a = obp.backtest(R, 200, 3, factors=f, rf=0.0001, shrink=True)
    assert len(a["obp"]) == 60 and np.allclose(a["weights"].sum(1), 1)
    assert not np.allclose(a["weights"], obp.backtest(R, 200, 3)["weights"])
    R2, f2 = R.copy(), f.copy()
    other_y, other_f, _, _ = factor_data(T=260, seed=9)
    R2[230:], f2[230:] = 1 + other_y[230:], other_f[230:]
    b = obp.backtest(R2, 200, 3, factors=f2, rf=0.0001, shrink=True)
    assert np.allclose(a["weights"][:31], b["weights"][:31])
    with pytest.raises(ValueError, match="one row"):
        obp.backtest(R, 200, 3, factors=f[:-1])


FRENCH = """This file was created by using the 202609 CRSP database.
Some more text.

,Mkt-RF,SMB,HML,RF
20200106,    0.50,   -0.20,    0.10,    0.01
20200107,   -1.00,    0.30,   -0.40,    0.01

Copyright 2026
"""


def test_factors_are_parsed_and_cached(tmp_path):
    table = obp.parse_factors(FRENCH)
    assert list(table.columns) == ["Mkt-RF", "SMB", "HML", "RF"] and len(table) == 2
    assert table.loc["2020-01-07", "Mkt-RF"] == pytest.approx(-0.01)
    cache, calls = tmp_path / "f.csv", []

    def download():
        calls.append(1)
        return table

    days = pd.DatetimeIndex(["2020-01-07"])
    a = obp.load_factors(days, cache=cache, download=download)
    b = obp.load_factors(days, cache=cache, download=download)
    assert len(calls) == 1 and np.allclose(a, b)
    with pytest.raises(ValueError, match="no factors"):
        obp.load_factors(pd.DatetimeIndex(["2020-01-08"]), cache=cache, download=download)


# ---------------------------------------------------------------- costs and years
def test_turnover_and_costs():
    w = np.array([[0.5, 0.5], [0.5, 0.5], [1.5, -0.5]])
    R = np.array([[1.2, 0.8], [1.0, 1.0], [1.1, 1.0]])
    tv = obp.turnover(w, R)
    assert tv[0] == pytest.approx(1.0)                                       # bought from cash
    assert tv[1] == pytest.approx(abs(0.5 - 0.6) + abs(0.5 - 0.4))           # back from the drifted weights
    assert tv[2] == pytest.approx(2.0)
    net = obp.after_costs([0.01, 0.0], [1.0, 0.0], 10)
    assert np.allclose(net, [0.999 * 1.01 - 1, 0.0])
    assert np.allclose(obp.after_costs([0.01], [3.0], 0), [0.01])


def test_wealth_stays_at_zero_after_a_total_loss():
    assert np.allclose(obp.wealth([0.5, -1.2, 0.3]), [1.5, 0.0, 0.0])


def test_yearly_returns():
    dates = pd.to_datetime(["2020-12-30", "2020-12-31", "2021-01-04"])
    y = obp.yearly([0.1, 0.1, -0.5], dates)
    assert y[2020] == pytest.approx(0.21) and y[2021] == pytest.approx(-0.5)
