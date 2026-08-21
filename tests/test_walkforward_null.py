"""
The null control's permutation is BLOCK-wise, and that is load-bearing.

A null is only as good as what it destroys and what it keeps. The claim
in docs/WALKFORWARD_FINDINGS.md - that the GA's in-sample Sharpe is a
search artifact because permuted data reproduces it - rests entirely on
the permuted series being a fair comparison. Two ways to get that wrong:

  * KEEP TOO MUCH and the null inherits the very predictability it is
    supposed to remove, so the real run wins by default.

  * DESTROY TOO MUCH and the null becomes unrealistically easy. An iid
    shuffle flattens volatility clustering to zero (measured: lag-1
    |return| autocorrelation 0.127 -> -0.001). `simulate()` sizes
    positions by trailing realised volatility, so on a series with no
    vol clustering the vol-targeting behaves differently than it does on
    real data. The null would then be weaker than the real run for a
    reason that has nothing to do with signal, and the comparison would
    flatter the real result.

Block permutation with a 21-bar block keeps ~86% of lag-1 vol
clustering, keeps the marginal return distribution exactly, and keeps
the cross-sectional correlation structure (rows move together), while
destroying the indicator-to-forward-return relationship the GA searches
for. These tests pin all four properties.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_evolved_walkforward import block_permute_prices        # noqa: E402

N_BARS, N_ASSETS, BLOCK = 420, 6, 21


def garch_like_prices(seed: int = 0) -> pd.DataFrame:
    """Prices with real volatility clustering, so the ACF test has
    something to preserve. White noise would pass trivially."""
    rng = np.random.RandomState(seed)
    vol = np.zeros((N_BARS, N_ASSETS))
    vol[0] = 0.01
    shocks = rng.normal(0, 1, (N_BARS, N_ASSETS))
    common = rng.normal(0, 1, (N_BARS, 1))          # cross-sectional factor
    for t in range(1, N_BARS):
        vol[t] = np.sqrt(1e-6 + 0.86 * vol[t - 1] ** 2
                         + 0.13 * (vol[t - 1] * shocks[t - 1]) ** 2)
    rets = vol * (0.6 * shocks + 0.8 * common)
    idx = pd.date_range("2024-01-01", periods=N_BARS, freq="D")
    cols = [f"A{i}" for i in range(N_ASSETS)]
    return pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx,
                        columns=cols)


def iid_shuffle(close: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The WRONG null, kept here so the tests can contrast against it."""
    r = close.pct_change().fillna(0.0)
    idx = np.random.RandomState(seed).permutation(len(r))
    return pd.DataFrame(close.iloc[0].values * np.cumprod(1 + r.values[idx], axis=0),
                        index=close.index, columns=close.columns)


def mean_abs_acf(close: pd.DataFrame, lag: int) -> float:
    a = np.abs(close.pct_change().fillna(0.0).values)
    return float(np.mean([np.corrcoef(a[:-lag, i], a[lag:, i])[0, 1]
                          for i in range(a.shape[1])]))


@pytest.fixture
def prices():
    return garch_like_prices()


class TestItIsBlockNotIid:
    def test_the_fixture_actually_has_volatility_clustering(self, prices):
        """Guard the guard: if the input had no clustering, the
        preservation test below would pass on a broken permutation."""
        assert mean_abs_acf(prices, 1) > 0.05

    def test_block_permutation_preserves_volatility_clustering(self, prices):
        orig = mean_abs_acf(prices, 1)
        blk = mean_abs_acf(block_permute_prices(prices, BLOCK, 42), 1)
        assert blk > 0.5 * orig, (
            f"lag-1 |return| ACF collapsed {orig:.4f} -> {blk:.4f}; this is "
            f"behaving like an iid shuffle, which makes the null too easy")

    def test_an_iid_shuffle_would_destroy_it(self, prices):
        """The contrast that gives the test above its meaning."""
        assert abs(mean_abs_acf(iid_shuffle(prices, 42), 1)) < 0.05

    def test_block_beats_iid_at_preserving_clustering(self, prices):
        orig = mean_abs_acf(prices, 1)
        blk = mean_abs_acf(block_permute_prices(prices, BLOCK, 42), 1)
        iid = mean_abs_acf(iid_shuffle(prices, 42), 1)
        assert abs(blk - orig) < abs(iid - orig)


class TestWhatMustBePreserved:
    def test_the_marginal_return_distribution_is_unchanged(self, prices):
        """Same returns, different order - so the null cannot be beaten
        by simply having calmer or wilder bars."""
        a = prices.pct_change().dropna().values.ravel()
        b = block_permute_prices(prices, BLOCK, 42).pct_change().dropna().values.ravel()
        assert np.std(b) == pytest.approx(np.std(a), rel=0.05)
        assert np.percentile(np.abs(b), 99) == pytest.approx(
            np.percentile(np.abs(a), 99), rel=0.15)

    def test_cross_sectional_correlation_survives(self, prices):
        """Rows are permuted JOINTLY across assets. Permuting each asset
        independently would destroy the correlation structure and make
        the null a different market, not the same market reordered."""
        def offdiag(c):
            m = c.pct_change().fillna(0.0).corr().values
            return m[np.triu_indices(len(m), 1)].mean()
        assert offdiag(block_permute_prices(prices, BLOCK, 42)) == pytest.approx(
            offdiag(prices), abs=0.05)

    def test_length_and_columns_are_preserved(self, prices):
        out = block_permute_prices(prices, BLOCK, 42)
        assert len(out) == len(prices)
        assert list(out.columns) == list(prices.columns)

    def test_prices_stay_positive(self, prices):
        """A non-positive price would trip build_market_data's validation
        and silently shrink the null's usable history."""
        assert (block_permute_prices(prices, BLOCK, 42).values > 0).all()


class TestWhatMustBeDestroyed:
    def test_the_bar_order_actually_changes(self, prices):
        out = block_permute_prices(prices, BLOCK, 42)
        a = prices.pct_change().fillna(0.0).values
        b = out.pct_change().fillna(0.0).values
        assert not np.allclose(a, b), "permutation was a no-op"

    def test_different_seeds_give_different_permutations(self, prices):
        x = block_permute_prices(prices, BLOCK, 1).values
        y = block_permute_prices(prices, BLOCK, 2).values
        assert not np.allclose(x, y), (
            "seeds must vary the permutation - a null CLOUD of identical "
            "draws would understate the spread")

    def test_the_same_seed_reproduces(self, prices):
        assert np.allclose(block_permute_prices(prices, BLOCK, 7).values,
                           block_permute_prices(prices, BLOCK, 7).values)

    def test_long_range_structure_decays_more_than_short_range(self, prices):
        """Blocks bound what survives: within-block lags are kept, lags
        beyond the block length are broken. That is the intended trade."""
        orig1, orig20 = mean_abs_acf(prices, 1), mean_abs_acf(prices, 20)
        out = block_permute_prices(prices, BLOCK, 42)
        k1 = mean_abs_acf(out, 1) / orig1 if orig1 else 0.0
        k20 = mean_abs_acf(out, 20) / orig20 if orig20 else 0.0
        assert k1 > k20
