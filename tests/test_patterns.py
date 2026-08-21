"""
Classical TA and microstructure channels.

THREE BUGS WERE FOUND BY LOOKING AT THE OUTPUT, not by any assertion,
and each is pinned below. All three produced a channel that was present,
correctly named, finite, and carrying nothing or duplicating something -
the failure mode where a feature matrix has the right shape and the
wrong contents.

  rogers_satchell was identically 0.000 with sd 0.000 across the panel:
  the formula's sign was inverted, and since RS is non-negative by
  construction the subsequent max(rs, 0) clamped every value to zero.

  hammer and shooting_star were ALGEBRAICALLY IDENTICAL - measured
  correlation +1.000000 - because each was written as the mirror of the
  other through a symmetric clip.

  kyle_lambda was ~1e-5 with a standard deviation that rounded to zero at
  display precision, leaving a rank statistic no resolution to work with.

None of these would have raised. Two of them would have silently added a
dead or duplicate column to a search already prone to overfitting.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from nightevolver.patterns import (
    FEATURE_NAMES, build_pattern_features, candlestick_features,
    channel_features, microstructure_features,
)


def ohlc(n=300, a=4, seed=0):
    """OHLCV with CLUSTERED volatility, not constant.

    The first version used a fixed sigma, and the redundancy test then
    flagged gk_vol and range_expansion as duplicates at r=0.9705. They
    are not: range_expansion is the range divided by its own trailing
    mean, so under constant volatility that denominator is nearly
    constant and the ratio collapses onto the level. Measured on 651
    bars of real NSE data the same pair correlates +0.7683.

    A fixture too simple to contain the phenomenon under test produces
    confident false positives - the same failure as an iid null that
    destroys volatility clustering and makes a strategy look better than
    it is.
    """
    rng = np.random.RandomState(seed)
    vol = np.zeros((n, a))
    vol[0] = 0.012
    shocks = rng.normal(0, 1, (n, a))
    for t in range(1, n):
        vol[t] = np.sqrt(1e-6 + 0.85 * vol[t - 1] ** 2
                         + 0.14 * (vol[t - 1] * shocks[t - 1]) ** 2)
    c = 100 * np.cumprod(1 + vol * shocks, axis=0)
    o = c * (1 + vol * rng.normal(0, 0.3, (n, a)))
    h = np.maximum(o, c) * (1 + np.abs(vol * rng.normal(0, 0.5, (n, a))))
    l = np.minimum(o, c) * (1 - np.abs(vol * rng.normal(0, 0.5, (n, a))))
    # Volume co-moves with volatility, as it does in real markets.
    v = np.abs(rng.normal(1e6, 2e5, (n, a))) * (1 + 10 * vol)
    return o, h, l, c, v


@pytest.fixture
def data():
    return ohlc()


class TestTheThreeMeasuredBugs:
    def test_rogers_satchell_is_not_identically_zero(self, data):
        o, h, l, c, v = data
        rs = microstructure_features(o, h, l, c, v)["rogers_satchell"]
        g = rs[np.isfinite(rs)]
        assert g.std() > 1e-6, "RS is constant - the sign inversion is back"
        assert (g > 0).mean() > 0.9, "RS must be non-negative volatility"

    def test_rogers_satchell_matches_the_published_formula(self):
        """RS = log(H/C)log(H/O) + log(L/C)log(L/O), pinned on one bar."""
        H, L, O, C = 105.0, 98.0, 100.0, 103.0
        want = np.log(H / C) * np.log(H / O) + np.log(L / C) * np.log(L / O)
        got = microstructure_features(
            np.array([[O]]), np.array([[H]]), np.array([[L]]), np.array([[C]]),
        )["rogers_satchell"][0, 0]
        assert got == pytest.approx(np.sqrt(max(want, 0.0) * 252.0), rel=1e-9)

    def test_hammer_and_shooting_star_are_not_the_same_channel(self, data):
        o, h, l, c, _ = data
        f = candlestick_features(o, h, l, c)
        a, b = f["hammer"], f["shooting_star"]
        assert not np.allclose(a, b), "identical channels"
        m = np.isfinite(a) & np.isfinite(b)
        r = np.corrcoef(a[m], b[m])[0, 1]
        assert abs(r) < 0.95, f"near-duplicate, r={r:+.4f}"

    def test_hammer_fires_on_a_hammer_and_not_on_a_star(self):
        """A long lower shadow with the body at the top is a hammer.
        The same bar flipped must not also read as a hammer."""
        # body 99-100 at the top, long lower wick to 90
        f = candlestick_features(np.array([[99.0]]), np.array([[100.0]]),
                                 np.array([[90.0]]), np.array([[99.5]]))
        assert f["hammer"][0, 0] > 0.2
        assert f["shooting_star"][0, 0] == pytest.approx(0.0, abs=1e-9)

    def test_shooting_star_fires_on_a_star_and_not_on_a_hammer(self):
        # body 99-99.5 at the bottom, long upper wick to 110
        f = candlestick_features(np.array([[99.0]]), np.array([[110.0]]),
                                 np.array([[99.0]]), np.array([[99.5]]))
        assert f["shooting_star"][0, 0] < -0.2
        assert f["hammer"][0, 0] == pytest.approx(0.0, abs=1e-9)

    def test_kyle_lambda_has_usable_resolution(self, data):
        o, h, l, c, v = data
        k = microstructure_features(o, h, l, c, v)["kyle_lambda"]
        g = k[np.isfinite(k)]
        assert g.std() > 1e-3, "resolution rounds to zero for a rank statistic"


class TestNoInfiniteValues:
    def test_nothing_is_inf_on_degenerate_bars(self):
        """A flat bar (h == l) divides by a zero range; a zero price
        divides by zero. Neither may produce inf - np.nan_to_num maps
        +inf to 1.8e308, which is how a poisoned target got past
        tanh-squashed indicators once already."""
        o = np.array([[100.0], [0.0], [50.0]])
        h = np.array([[100.0], [0.0], [50.0]])
        l = np.array([[100.0], [0.0], [50.0]])
        c = np.array([[100.0], [0.0], [50.0]])
        v = np.array([[0.0], [0.0], [0.0]])
        for name, arr in build_pattern_features(c, h, l, v, open_=o).items():
            assert not np.isinf(arr).any(), f"{name} produced inf"

    def test_all_declared_channels_are_produced(self, data):
        o, h, l, c, v = data
        f = build_pattern_features(c, h, l, v, open_=o)
        assert set(f) == set(FEATURE_NAMES)


class TestCausality:
    def test_a_future_bar_cannot_change_the_present(self, data):
        """Every channel at t must use only bars <= t."""
        o, h, l, c, v = data
        base = build_pattern_features(c, h, l, v, open_=o)
        o2, h2, l2, c2, v2 = (x.copy() for x in (o, h, l, c, v))
        cut = 200
        for x in (o2, h2, l2, c2):
            x[cut:] *= 1.5
        v2[cut:] *= 3.0
        after = build_pattern_features(c2, h2, l2, v2, open_=o2)
        for name in FEATURE_NAMES:
            a, b = base[name][:cut], after[name][:cut]
            both = np.isfinite(a) & np.isfinite(b)
            assert np.allclose(a[both], b[both]), \
                f"{name} at t depends on bars after t"


class TestKnownRedundancy:
    def test_the_three_vol_estimators_are_near_collinear(self, data):
        """NOT a bug - they estimate the same quantity, and Rogers-
        Satchell is kept because it is unbiased under drift where the
        others are not. Pinned so the AUDIT's interpretation stays
        honest: FDR treats them as three independent tests when they are
        effectively one, which inflates the apparent number of
        survivors among them."""
        o, h, l, c, v = data
        f = microstructure_features(o, h, l, c, v)
        gk, pk = f["gk_vol"], f["parkinson_vol"]
        m = np.isfinite(gk) & np.isfinite(pk)
        assert np.corrcoef(gk[m], pk[m])[0, 1] > 0.9

    def test_no_other_channel_pair_is_a_duplicate(self, data):
        """The sweep that found hammer/shooting_star. Volatility
        estimators are exempted as documented above."""
        o, h, l, c, v = data
        f = build_pattern_features(c, h, l, v, open_=o)
        vols = {"gk_vol", "parkinson_vol", "rogers_satchell"}
        M = np.column_stack([f[k].ravel() for k in FEATURE_NAMES])
        M = M[np.isfinite(M).all(axis=1)]
        C = np.corrcoef(M, rowvar=False)
        for i, j in itertools.combinations(range(len(FEATURE_NAMES)), 2):
            a, b = FEATURE_NAMES[i], FEATURE_NAMES[j]
            if {a, b} <= vols:
                continue
            assert abs(C[i, j]) < 0.97, f"{a} and {b} duplicate: r={C[i, j]:+.4f}"


class TestChannelSystems:
    def test_donchian_position_is_bounded(self, data):
        _, h, l, c, _ = data
        d = channel_features(h, l, c)["donchian_pos"]
        g = d[np.isfinite(d)]
        assert g.min() >= 0.0 and g.max() <= 1.0

    def test_ichimoku_spans_are_not_shifted_forward(self, data):
        """Plotting senkou 26 bars ahead is a CHARTING convention.
        Shifting it into a feature matrix would place a value derived
        from bar t at bar t+26 - a look-ahead wearing a convention's
        clothes. Covered by the causality test; asserted here as intent."""
        _, h, l, c, _ = data
        f = channel_features(h, l, c)
        assert np.isfinite(f["ichimoku_cloud_pos"][60:]).mean() > 0.9
