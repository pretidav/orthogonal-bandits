#David Preti
"""Orthogonal Bandit Portfolio (OBP).

    W. Shen, J. Wang, Y.-G. Jiang, H. Zha. Portfolio choices with orthogonal bandit learning.
    IJCAI 2015, pp. 974-980.  https://www.ijcai.org/Abstract/15/142

At every step k, from the gross returns R = S_k / S_{k-1} of the previous tau periods:

    Sigma = H Lambda H^T                                      eq. (5), eigenvalues in decreasing order
    Ht_i = H_i / (H_i^T 1),   lt_i = lambda_i / (H_i^T 1)^2   eq. (7), (8): principal portfolios and their variances
    r_i = Ht_i^T E[R] / sqrt(lt_i)                            eq. (10): Sharpe ratio of portfolio i, the reward
    i* = argmax  r_i + sqrt(2 ln(k + tau) / (tau + k_i))      eq. (11), one among the first l, one among the others
    theta = lt_i* / (lt_i* + lt_j*)                           eq. (12): minimum variance mix
    w = (1 - theta) Ht_i* + theta Ht_j*                       eq. (13)

and the net return w^T R_k - 1 is realized on the period k, which is not in the window.

Sigma and E[R] are estimated in two ways, and both are run:

    sample     the sample covariance and the sample mean of the window
    paper      Sigma from the factor model of Fan, Fan and Lv (J. of Econometrics 147, 2008, eq. 1.4), with the three
               Fama-French factors: excess returns y = B f + e (no intercept), Sigma = B cov(f) B^T + diag(E E^T / T);
               E[R] from the James-Stein shrinkage estimator of A. Meucci, Risk and Asset Allocation, eq. (4.138),
               (4.139): (1 - alpha) mean + alpha b,  alpha = (tr S - 2 lambda_max(S)) / (T |mean - b|^2), kept in
               [0, 1], with S the sample covariance and b a fixed vector

The paper does not say which factors nor which b it uses: the Fama-French factors and b = 1 (no net return) are my
choices. Other differences: the number
l of significant portfolios is given (the paper chooses it where the eigenvalues drop) and the data are daily. The
variant without short sales is not in the paper.

Transaction costs are not in the paper: --cost charges a given number of basis points on the value traded every day.
There is no cost for borrowing the assets that are sold short.

    python obp.py                              # download (or read the csv files in data/), run, save figures in fig/
    python obp.py --start=2007-09-01 --end=2010-01-09                        # the period of the first version
    python obp.py --prices=my_prices.csv --tickers AAA BBB CCC DDD --l 2
"""
import argparse
import io
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# US listed stocks only, so that there is one currency and one trading calendar. The first version had VOW.DE, CVGW,
# CAMP, WSCI and LNDC, replaced by F, ADM, CSCO, CAT and KO (Yahoo does not serve the last four for 2007-2010 any more,
# and its euro rate has holes and errors in 2008).
TICKERS = ["F", "BA", "AMD", "AAPL", "GME", "ADM", "CSCO", "CAT", "KO", "WOR"]
MAX_GAP = 5  # longest run of days without a price that is filled with the last one
# daily Fama-French factors, from the data library of Kenneth French
FACTORS_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
FACTOR_NAMES = ["Mkt-RF", "SMB", "HML"]
TOL = 1e-8
EW_BLUE = "#1f4fff"  # equally weighted in the wealth figure: not a colour of any other line there
SMOOTH = 63  # days (a quarter): theta and the weights of long runs are plotted as moving averages


def check(condition, message):
    """Like assert, but it is not removed by python -O."""
    if not condition:
        raise RuntimeError(message)


# ---------------------------------------------------------------------------------------------- data
def download_close(symbols, start, end):
    """Closing prices adjusted for splits and dividends, from Yahoo (needs the network)."""
    import yfinance as yf

    close = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    return close.to_frame(symbols[0]) if isinstance(close, pd.Series) else close


def align_prices(close, tickers):
    """One calendar (the days on which at least one asset traded) and the columns in the order of `tickers`.

    A missing price (a halt, a hole in the data) is replaced by the last one, for at most MAX_GAP days: the return of
    that day is zero and the whole move is counted on the next one, so no part of it is thrown away.
    """
    missing = [t for t in tickers if t not in close.columns or close[t].isna().all()]
    if missing:
        raise ValueError(f"no prices for {missing}")
    prices = close.sort_index()[list(tickers)].dropna(how="all").ffill(limit=MAX_GAP)
    complete = prices.notna().all(axis=1)
    check(complete.any(), "there is no day with a price for every asset")
    prices = prices.loc[complete.idxmax():]  # start when every asset has a price
    gaps = [t for t in tickers if prices[t].isna().any()]
    if gaps:
        raise ValueError(f"more than {MAX_GAP} days without a price for {gaps}")
    check((prices > 0).all().all(), "prices must be positive")
    return prices


def load_prices(tickers, start, end, cache=None, download=download_close):
    """Prices in USD for start <= date < end. Read from the csv `cache` if it exists, else downloaded and saved there."""
    if cache is not None and Path(cache).exists():
        prices = pd.read_csv(cache, index_col=0, parse_dates=True)
        missing = [t for t in tickers if t not in prices.columns]
        if missing:
            raise ValueError(f"{cache} has no column for {missing}: delete it to download again")
        slack = pd.Timedelta(days=10)
        if prices.index[0] > pd.Timestamp(start) + slack or prices.index[-1] < pd.Timestamp(end) - slack:
            raise ValueError(f"{cache} covers {prices.index[0].date()} .. {prices.index[-1].date()}, not the "
                             f"requested period: delete it to download again")
        prices = prices.loc[(prices.index >= start) & (prices.index < end), list(tickers)]
        check(prices.notna().all().all() and (prices > 0).all().all(), f"missing or non positive prices in {cache}")
        return prices
    foreign = [t for t in tickers if "." in t]  # a suffix is another exchange, hence another currency (VOW.DE)
    if foreign:
        raise ValueError(f"{foreign}: only US listed tickers are downloaded, there is no currency conversion. "
                         f"Give the prices in USD with --prices")
    prices = align_prices(download(list(tickers), start, end), tickers)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache)
    return prices


def gross_returns(prices):
    """R_k = S_k / S_{k-1}."""
    return (prices / prices.shift(1)).iloc[1:]


def parse_factors(text):
    """The csv of the data library (some lines of text, a header, rows yyyymmdd in percent) -> decimal returns."""
    columns = FACTOR_NAMES + ["RF"]
    lines = text.splitlines()
    header = next((line for line in lines if "Mkt-RF" in line), None)
    check(header is not None and [c.strip() for c in header.split(",")][1:] == columns, "unknown factor file")
    rows = [line.split(",") for line in lines if re.match(r"\s*\d{8}\s*,", line)]
    check(len(rows) > 0, "the factor file has no data")
    index = pd.to_datetime([r[0].strip() for r in rows], format="%Y%m%d")
    return pd.DataFrame([[float(v) / 100 for v in r[1:]] for r in rows], index=index, columns=columns)


def download_factors():
    """Download the daily factors (needs the network). The zip is read in memory, nothing is extracted to disk."""
    with urllib.request.urlopen(FACTORS_URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
    return parse_factors(archive.read(name).decode("latin-1"))


def load_factors(dates, cache=None, download=download_factors):
    """Factors and risk free rate on the given dates. Read from the csv `cache` if it exists, else downloaded."""
    if cache is not None and Path(cache).exists():
        factors = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        factors = download()
        if cache is not None:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            factors.to_csv(cache)
    missing = dates.difference(factors.index)
    if len(missing):
        raise ValueError(f"no factors for {len(missing)} days, from {missing[0].date()} to {missing[-1].date()}")
    return factors.loc[dates]


# ---------------------------------------------------------------------------------------------- the estimators
def factor_covariance(excess, factors):
    """Fan, Fan and Lv (2008), eq. (1.4). Regress the excess returns (T, assets) on the factors (T, K), without
    intercept, and return B cov(f) B^T + diag(E E^T / T), with E the residuals and cov(f) the sample covariance."""
    excess, factors = np.asarray(excess, float), np.asarray(factors, float)
    T = len(excess)
    B = np.linalg.lstsq(factors, excess, rcond=None)[0].T  # (assets, K)
    residual = excess - factors @ B.T
    cov_f = np.atleast_2d(np.cov(factors, rowvar=False))
    return B @ cov_f @ B.T + np.diag((residual**2).sum(axis=0) / T)


def james_stein_mean(window, target=1.0):
    """James-Stein shrinkage of the sample mean towards the fixed target b (Meucci, eq. 4.138, 4.139). The default
    b = 1 is a gross return of one for every asset, that is no net return.

    Returns the estimate and alpha. alpha = 0 (no shrinkage) if one eigenvalue is more than half of the trace."""
    window = np.asarray(window, float)
    T = len(window)
    mean = window.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(np.cov(window, rowvar=False))
    distance = ((mean - target) ** 2).sum()
    alpha = 1.0 if distance == 0 else (eigenvalues.sum() - 2 * eigenvalues[-1]) / (T * distance)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (1 - alpha) * mean + alpha * target, alpha


# ---------------------------------------------------------------------------------------------- the algorithm
def principal_portfolios(Sigma):
    """Eigenvalues (decreasing), the principal portfolios Ht (columns, weights summing to 1) and their variances lt."""
    L, H = np.linalg.eigh(Sigma)
    L, H = L[::-1], H[:, ::-1]
    n = len(L)
    check(np.allclose(H.T @ H, np.eye(n), atol=TOL), "the eigenvectors are not orthonormal")
    check(np.allclose((H * L) @ H.T, Sigma, rtol=TOL, atol=TOL * np.abs(L).max()), "wrong eigendecomposition")
    check(L[-1] > 0, "the covariance matrix is not positive definite")
    s = H.sum(axis=0)
    check(np.all(np.abs(s) > TOL), "a dollar neutral eigenvector cannot be normalized")
    Ht = H / s  # a new array: H is not modified
    lt = L / s**2
    check(np.allclose(Ht.sum(axis=0), 1.0, atol=1e-6), "the portfolio weights do not sum to one")
    return L, Ht, lt


def sharpe_rewards(Ht, lt, mean_return):
    """Eq. (10): expected return of every principal portfolio over its standard deviation."""
    return Ht.T @ mean_return / np.sqrt(lt)


def ucb_select(rewards, counts, k, tau):
    """Eq. (11): index of the arm with the largest reward plus confidence bound. k is the step, starting from 1."""
    bonus = np.sqrt(2.0 * np.log(k + tau) / (tau + counts))
    return int(np.argmax(rewards + bonus))


def mix(Ht, lt, i, j):
    """Eq. (12), (13): the mix of the portfolios i (significant) and j (the others) with minimum variance."""
    theta = lt[i] / (lt[i] + lt[j])
    return theta, (1.0 - theta) * Ht[:, i] + theta * Ht[:, j]


def no_short_sale(w):
    """Negative weights set to zero, the others rescaled to sum to one (not in the paper)."""
    positive = np.clip(w, 0.0, None)
    return positive / positive.sum()  # the sum is positive because the weights sum to one


def backtest(R, tau, l, factors=None, rf=0.0, shrink=False):
    """Rolling window backtest on the gross returns R (periods, assets). Trades the periods tau, ..., len(R) - 1.

    factors  None: sample covariance. An array (periods, K): factor model covariance of the excess returns R - 1 - rf
    rf       risk free return of every period (used only with factors)
    shrink   False: sample mean. True: James-Stein mean
    """
    R = np.asarray(R, dtype=float)
    T, n = R.shape
    if not 0 < l < n:
        raise ValueError(f"l must be between 1 and {n - 1}")
    if not n < tau < T:
        raise ValueError(f"tau must be larger than the number of assets ({n}) and smaller than the periods ({T})")
    if factors is not None:
        factors = np.asarray(factors, dtype=float)
        if factors.ndim != 2 or len(factors) != T:
            raise ValueError("factors must have one row for every period")
        excess = R - 1.0 - np.broadcast_to(np.asarray(rf, dtype=float), (T,))[:, None]
    counts = np.zeros(n)
    out = {key: [] for key in ("weights", "weights_nss", "theta", "chosen", "alpha", "obp", "nss", "ew")}
    for k in range(T - tau):
        window = R[k:k + tau]  # the traded period k + tau is not included
        if factors is None:
            Sigma = np.cov(window, rowvar=False)
        else:
            Sigma = factor_covariance(excess[k:k + tau], factors[k:k + tau])
        mean, alpha = james_stein_mean(window) if shrink else (window.mean(axis=0), 0.0)
        L, Ht, lt = principal_portfolios(Sigma)
        rewards = sharpe_rewards(Ht, lt, mean)
        i = ucb_select(rewards[:l], counts[:l], k + 1, tau)
        j = l + ucb_select(rewards[l:], counts[l:], k + 1, tau)
        counts[[i, j]] += 1
        theta, w = mix(Ht, lt, i, j)
        w_nss = no_short_sale(w)
        check(abs(w.sum() - 1.0) < 1e-6 and abs(w_nss.sum() - 1.0) < 1e-6, "the portfolio weights do not sum to one")
        today = R[k + tau]
        out["weights"].append(w)
        out["weights_nss"].append(w_nss)
        out["theta"].append(theta)
        out["chosen"].append((i, j))
        out["alpha"].append(alpha)
        out["obp"].append(w @ today - 1.0)
        out["nss"].append(w_nss @ today - 1.0)
        out["ew"].append(today.mean() - 1.0)
    return {key: np.array(value) for key, value in out.items()}


def wealth(returns):
    """Cumulative wealth of one dollar, eq. (16). A loss of everything or more leaves zero for ever."""
    return np.cumprod(np.clip(1.0 + np.asarray(returns), 0.0, None))


def performance(returns):
    """Eq. (14)-(16): mean and standard deviation of the net returns, Sharpe ratio, terminal wealth of one dollar."""
    returns = np.asarray(returns)
    mean, std = returns.mean(), returns.std(ddof=1)
    return {"mean": mean, "std": std, "sharpe": mean / std, "wealth": float(wealth(returns)[-1])}


def turnover(weights, R):
    """Value traded at the start of every period, per unit of wealth: sum_i |w_k,i - w'_k,i|, where w' are the weights
    of the previous period after the prices have moved. R are the gross returns of the periods of the weights. The
    first period is bought from cash."""
    weights, R = np.asarray(weights, float), np.asarray(R, float)
    moved = weights * R
    moved = moved / moved.sum(axis=1, keepdims=True)
    before = np.vstack([np.zeros(weights.shape[1]), moved[:-1]])
    return np.abs(weights - before).sum(axis=1)


def after_costs(returns, traded, cost_bps):
    """Net returns when every unit of value traded costs cost_bps / 10000 of wealth."""
    return (1.0 - cost_bps * 1e-4 * np.asarray(traded)) * (1.0 + np.asarray(returns)) - 1.0


def yearly(returns, dates):
    """Compounded return of every calendar year."""
    return (1.0 + pd.Series(np.asarray(returns), index=pd.DatetimeIndex(dates))).groupby(lambda d: d.year).prod() - 1.0


# ---------------------------------------------------------------------------------------------- figures
def plot_yearly(series, dates, folder):
    """Bars of the return of every year. series: list of (label, returns, colour)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style

    plot_style.use()
    tables = [(label, yearly(returns, dates), color) for label, returns, color in series]
    years = tables[0][1].index
    x = np.arange(len(years))
    width = 0.8 / len(tables)
    fig, ax = plt.subplots(figsize=(max(9, 0.6 * len(years)), 4.8))
    for k, (label, table, color) in enumerate(tables):
        ax.bar(x + (k - (len(tables) - 1) / 2) * width, 100 * table.to_numpy(), width, color=color, label=label)
    ax.axhline(0.0, color=plot_style.INK, lw=1.2)
    ax.set_xticks(x, years, rotation=90 if len(years) > 12 else 0)
    ax.set_ylabel("RETURN OF THE YEAR (%)")
    ax.set_title("[ ORTHOGONAL BANDITS ]  YEAR BY YEAR, NO COSTS")
    plot_style.style_axes(ax)
    plot_style.legend(ax, "best")
    plot_style.savefig(fig, Path(folder) / "yearly.png", bbox_inches="tight")
    plt.close(fig)


def make_figures(sample, paper, dates, tickers, folder, smooth=None):
    """The results of the backtest with the sample estimators and with those of the paper."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import plot_style

    plot_style.use()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    ACCENT, INK, GREY = plot_style.ACCENT, plot_style.INK, plot_style.GREY

    def date_axis(ax):
        locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    long_run = len(dates) > 3 * 252
    if smooth is None:
        smooth = SMOOTH if long_run else 1
    average = f"   (MOVING AVERAGE OVER {smooth} DAYS)" if smooth > 1 else ""

    def moving(values):
        return pd.DataFrame(np.asarray(values)).rolling(smooth, min_periods=1).mean().to_numpy()

    # cumulative wealth
    fig, ax = plt.subplots(figsize=(9, 5))
    for returns, label, color, ls, lw in (
            (sample["obp"], "OBP, SAMPLE ESTIMATORS", ACCENT, "-", 2.4),
            (paper["obp"], "OBP, PAPER ESTIMATORS", INK, "-", 2.0),
            (sample["nss"], "OBP SAMPLE, WITHOUT SHORT SALE", GREY, "-", 1.4),
            (sample["ew"], "EQUALLY WEIGHTED", EW_BLUE, "-", 1.6)):
        ax.plot(dates, wealth(returns), color=color, ls=ls, lw=min(lw, 1.4) if long_run else lw, label=label)
    ax.axhline(1.0, color=INK, lw=1.0, ls=plot_style.DOTTED)
    scale = ""
    if long_run:
        from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        scale = "  (LOG SCALE)"
    ax.set_ylabel("CUMULATIVE WEALTH OF ONE DOLLAR" + scale)
    ax.set_title("[ ORTHOGONAL BANDITS ]  CUMULATIVE WEALTH, NO COSTS")
    date_axis(ax)
    plot_style.style_axes(ax)
    plot_style.legend(ax, "best")
    plot_style.savefig(fig, folder / "OBPvsEW.png", bbox_inches="tight")
    plt.close(fig)

    # theta
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(dates, moving(sample["theta"]), color=ACCENT, lw=2.2, label="SAMPLE ESTIMATORS")
    ax.plot(dates, moving(paper["theta"]), color=INK, lw=1.4, label="PAPER ESTIMATORS")
    ax.set_ylim(-0.03, 1.03)
    for level in (0.0, 1.0):
        ax.axhline(level, color=INK, lw=1.0, ls=plot_style.DASHED)
    ax.set_ylabel(r"MIXING WEIGHT  $\theta$")
    ax.set_title("[ ORTHOGONAL BANDITS ]  1 = ONLY ACTIVE, 0 = ONLY PASSIVE" + average)
    date_axis(ax)
    plot_style.style_axes(ax)
    plot_style.legend(ax, "best")
    plot_style.savefig(fig, folder / "theta.png", bbox_inches="tight")
    plt.close(fig)

    # weights: one panel for every asset
    n = len(tickers)
    ncols = min(5 if n > 6 else 3, n)
    nrows = int(np.ceil(n / ncols))
    for key, name, title in (("weights", "weights.png", "INVESTED PERCENTAGE, OBP"),
                             ("weights_nss", "weights_nss.png", "INVESTED PERCENTAGE, OBP WITHOUT SHORT SALE")):
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.9 * nrows), sharex=True, sharey=True,
                                 squeeze=False)
        for a, ax in enumerate(axes.flat):
            if a >= n:
                ax.axis("off")
                continue
            ax.axhline(0.0, color=INK, lw=0.8)
            ax.axhline(100.0 / n, color=GREY, lw=1.4, ls=plot_style.DASHED, label="EQUALLY WEIGHTED")
            ax.plot(dates, 100 * moving(sample[key])[:, a], color=ACCENT, lw=1.8 if long_run else 2.4,
                    label="SAMPLE ESTIMATORS")
            ax.plot(dates, 100 * moving(paper[key])[:, a], color=INK, lw=1.0 if long_run else 1.4,
                    label="PAPER ESTIMATORS")
            ax.set_title(tickers[a])
            date_axis(ax)
            plot_style.style_axes(ax)
        for row in axes:
            row[0].set_ylabel("% OF WEALTH")
        fig.suptitle(f"[ {title} ]{average}", x=0.02, ha="left", fontweight="bold", fontsize=11)
        fig.tight_layout()
        handles, labels = axes[0, 0].get_legend_handles_labels()
        plot_style.style_legend(fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.995, 1.01), ncol=3,
                                           fancybox=False, framealpha=1, edgecolor=INK, facecolor=plot_style.PAPER))
        plot_style.savefig(fig, folder / name, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------------------------- command line
def main(argv=None):
    here = Path(__file__).parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="+", default=TICKERS)
    p.add_argument("--start", default="2002-03-01", help="first day of the prices (GME is listed from February 2002)")
    p.add_argument("--end", default="2026-01-01", help="day after the last one")
    p.add_argument("--tau", type=int, default=500, help="periods of the rolling window")
    p.add_argument("--l", type=int, default=3, help="number of significant (passive) portfolios")
    p.add_argument("--prices", default=None,
                   help="csv of prices in USD (date, one column per ticker): read if it exists, else written. "
                        "Default: data/prices_<start>_<end>.csv")
    p.add_argument("--cost", type=float, default=10.0, help="transaction cost, basis points of the value traded")
    p.add_argument("--factors", default=str(here / "data" / "factors.csv"),
                   help="csv of the daily Fama-French factors: read if it exists, else downloaded and written")
    p.add_argument("--out", default=str(here / "fig"), help="folder of the figures")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    cache = args.prices or here / "data" / f"prices_{args.start}_{args.end}.csv"
    prices = load_prices(args.tickers, args.start, args.end, cache=cache)
    R = gross_returns(prices)
    factors = load_factors(R.index, cache=args.factors)
    sample = backtest(R.to_numpy(), args.tau, args.l)
    paper = backtest(R.to_numpy(), args.tau, args.l, factors=factors[FACTOR_NAMES].to_numpy(),
                     rf=factors["RF"].to_numpy(), shrink=True)
    dates = R.index[args.tau:]
    traded_R = R.to_numpy()[args.tau:]
    equal = np.full(traded_R.shape, 1.0 / traded_R.shape[1])

    print(f"assets   : {' '.join(args.tickers)}")
    print(f"returns  : {len(R)}  ({R.index[0].date()} .. {R.index[-1].date()})")
    print(f"traded   : {len(dates)}  ({dates[0].date()} .. {dates[-1].date()}), tau = {args.tau}, l = {args.l}")
    for name, result in (("sample", sample), ("paper ", paper)):
        print(f"{name} estimators: mean theta {result['theta'].mean():.3f}, mean gross exposure sum|w| "
              f"{np.abs(result['weights']).sum(axis=1).mean():.2f}, shrinkage alpha > 0 on "
              f"{int((result['alpha'] > 0).sum())} days")
    strategies = (("OBP, sample", sample["obp"], sample["weights"]),
                  ("OBP, paper estimators", paper["obp"], paper["weights"]),
                  ("OBP, sample, without short sale", sample["nss"], sample["weights_nss"]),
                  ("OBP, paper, without short sale", paper["nss"], paper["weights_nss"]),
                  ("equally weighted", sample["ew"], equal))
    print(f"\n{'strategy':32s} {'mean':>8s} {'std':>7s} {'Sharpe':>7s} {'wealth':>8s} {'t vs EW':>8s} {'turnover':>9s} "
          f"| with {args.cost:g} bp: {'Sharpe':>6s} {'wealth':>8s}   (per day)")
    for name, returns, weights in strategies:
        traded = turnover(weights, traded_R)
        v, c = performance(returns), performance(after_costs(returns, traded, args.cost))
        d = returns - sample["ew"]
        t = f"{d.mean() / (d.std(ddof=1) / np.sqrt(len(d))):8.2f}" if d.any() else f"{'':8s}"
        print(f"{name:32s} {v['mean']:8.5f} {v['std']:7.4f} {v['sharpe']:7.4f} {v['wealth']:8.2f} {t} "
              f"{traded.mean():9.4f} | {'':12s} {c['sharpe']:6.4f} {c['wealth']:8.2f}")
    table = pd.DataFrame({name: yearly(returns, dates) for name, returns, _ in strategies})
    table.columns = ["sample", "paper", "sample nss", "paper nss", "equal"]
    print("\nreturn of every year, no costs, per cent\n" + (100 * table).round(1).to_string())
    if not args.no_plot:
        make_figures(sample, paper, dates, args.tickers, args.out)
        plot_yearly([("OBP, SAMPLE ESTIMATORS", sample["obp"], "#ff3b00"),
                     ("OBP, PAPER ESTIMATORS", paper["obp"], "#000000"),
                     ("EQUALLY WEIGHTED", sample["ew"], "#c9b9a0")], dates, args.out)
        print(f"\nfigures saved in {args.out}")
    return {"sample": sample, "paper": paper}


if __name__ == "__main__":
    main()
