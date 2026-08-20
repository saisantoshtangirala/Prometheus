"""
Tests for the non-monotonic dependence audit.

The whole point of this module is to cover a blind spot in the Spearman
audit, so the headline tests plant relationships that rank correlation
provably cannot see and require the new statistics to find them. If
those ever fail, the "no directional edge" conclusion reverts to the
weaker claim it started as: no MONOTONIC dependence.
"""

from __future__ import annotations

import numpy as np
import pytest

from nightevolver.data_loader import build_market_data
from nightevolver.nonlinear_audit import (
    _double_center, _pairwise_abs, audit_nonlinear, distance_correlation,
    kruskal_bin_statistic,
)
from nightevolver.targets import build_targets


def _walk(n=400, a=4, seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    px = 100 * np.cumprod(1 + rng.normal(0, 0.012, size=(n, a)), axis=0)
    return pd.DataFrame(px, index=pd.bdate_range("2022-01-01", periods=n),
                        columns=[f"T{i}" for i in range(a)])


# --------------------------------------------------------------------------
# distance correlation
# --------------------------------------------------------------------------

class TestDistanceCorrelation:
    def test_independent_variables_score_near_zero(self):
        rng = np.random.default_rng(0)
        d = distance_correlation(rng.normal(size=600), rng.normal(size=600))
        assert d < 0.15, f"independent pair scored {d:.4f}"

    def test_identical_variables_score_one(self):
        x = np.random.default_rng(1).normal(size=300)
        assert distance_correlation(x, x) == pytest.approx(1.0, abs=1e-9)

    def test_linear_relationship_is_detected(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=600)
        assert distance_correlation(x, x + 0.3 * rng.normal(size=600)) > 0.8

    @pytest.mark.parametrize("shape,fn", [
        ("v", lambda x: np.abs(x)),
        ("quadratic", lambda x: x ** 2),
        ("cosine", lambda x: np.cos(2 * x)),
    ])
    def test_non_monotonic_shapes_are_detected_where_spearman_is_blind(self, shape, fn):
        """THE headline test. Spearman cancels to ~0 on all three; dCor
        must not. This is the exact blind spot that made the linear
        audit's null weaker than it appeared."""
        from scipy.stats import spearmanr
        rng = np.random.default_rng(3)
        x = rng.normal(size=600)
        y = fn(x) + 0.3 * rng.normal(size=600)

        rho = abs(spearmanr(x, y).statistic)
        dcor = distance_correlation(x, y)
        assert rho < 0.15, f"{shape}: Spearman saw it ({rho:.3f}) - test is vacuous"
        assert dcor > 0.25, f"{shape}: dCor missed it ({dcor:.4f})"

    def test_symmetry(self):
        rng = np.random.default_rng(4)
        x, y = rng.normal(size=200), rng.normal(size=200)
        assert distance_correlation(x, y) == pytest.approx(
            distance_correlation(y, x), abs=1e-12)

    def test_bounded_in_unit_interval(self):
        rng = np.random.default_rng(5)
        for _ in range(10):
            d = distance_correlation(rng.normal(size=150), rng.normal(size=150))
            assert 0.0 <= d <= 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            distance_correlation(np.zeros(10), np.zeros(11))

    def test_constant_input_returns_zero_not_nan(self):
        rng = np.random.default_rng(6)
        assert distance_correlation(np.ones(100), rng.normal(size=100)) == 0.0


class TestPermutationShortcut:
    def test_reindexing_the_centered_matrix_equals_recomputing(self):
        """The permutation loop reuses the double-centered matrices
        instead of rebuilding them, which is what makes a 104-pair grid
        tractable. Permuting then centering must equal centering then
        permuting, or the null is silently wrong."""
        rng = np.random.default_rng(7)
        n = 250
        x = rng.normal(size=n)
        y = np.abs(x) + 0.3 * rng.normal(size=n)
        idx = rng.permutation(n)

        direct = distance_correlation(x, y[idx])

        Ac = _double_center(_pairwise_abs(x))
        Bc = _double_center(_pairwise_abs(y))
        Bp = Bc[np.ix_(idx, idx)]
        dcov2 = float((Ac * Bp).mean())
        denom = np.sqrt(float((Ac * Ac).mean()) * float((Bc * Bc).mean()))
        fast = np.sqrt(dcov2 / denom) if dcov2 > 0 else 0.0

        assert fast == pytest.approx(direct, abs=1e-12)


# --------------------------------------------------------------------------
# quantile-bin test
# --------------------------------------------------------------------------

class TestKruskalBins:
    def test_v_shape_produces_a_large_statistic(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=600)
        flat = kruskal_bin_statistic(x, rng.normal(size=600))
        vee = kruskal_bin_statistic(x, np.abs(x) + 0.3 * rng.normal(size=600))
        assert vee > 10 * max(flat, 1.0)

    def test_independent_data_is_small(self):
        rng = np.random.default_rng(9)
        assert kruskal_bin_statistic(rng.normal(size=600),
                                     rng.normal(size=600)) < 25

    def test_too_few_samples_returns_zero(self):
        assert kruskal_bin_statistic(np.arange(5.0), np.arange(5.0)) == 0.0


# --------------------------------------------------------------------------
# end-to-end audit
# --------------------------------------------------------------------------

class TestAuditEndToEnd:
    def test_random_walk_yields_no_survivors(self):
        """Calibration. Real indicators on a random walk must produce
        nothing under the stronger statistic too."""
        md = build_market_data(_walk(360, 4, seed=11))
        from nightevolver.genome import INDICATOR_NAMES
        res = audit_nonlinear(md.indicators[:, :, :6], list(INDICATOR_NAMES[:6]),
                              build_targets(md.close), n_permutations=600, seed=1)
        assert res.pairs, "audit produced no pairs at all"
        assert not res.survivors, \
            f"false positives on a random walk: {[str(p) for p in res.survivors]}"

    def test_planted_v_shape_is_recovered_and_flagged_as_hidden(self):
        """A U-shaped predictor is invisible to Spearman. The audit must
        find it AND mark it as one the monotonic test would have missed."""
        md = build_market_data(_walk(360, 4, seed=12))
        feats = md.indicators[:, :, :4].copy()
        fwd = md.forward_returns

        # channel 0 becomes |forward return| shifted into the feature slot:
        # a symmetric, entirely non-monotonic relationship with direction.
        rng = np.random.default_rng(2)
        feats[:, :, 0] = np.tanh(
            (np.abs(fwd) - np.abs(fwd).mean()) / (np.abs(fwd).std() + 1e-9)
            + 0.15 * rng.normal(size=fwd.shape))

        names = ["planted_v", "c1", "c2", "c3"]
        res = audit_nonlinear(feats, names, {"direction_1d": build_targets(md.close)["direction_1d"]},
                              spearman_ref={("planted_v", "direction_1d"): 0.0},
                              n_permutations=200, seed=1)
        found = {p.feature for p in res.survivors}
        assert "planted_v" in found, f"planted non-monotonic signal missed; got {found}"
        hidden = {p.feature for p in res.hidden}
        assert "planted_v" in hidden, "not flagged as invisible to Spearman"

    def test_mismatched_names_raise(self):
        md = build_market_data(_walk(200, 3, seed=13))
        with pytest.raises(ValueError, match="names for"):
            audit_nonlinear(md.indicators[:, :, :4], ["a", "b"],
                            build_targets(md.close), n_permutations=5)

    def test_summary_states_the_stronger_null(self):
        md = build_market_data(_walk(320, 3, seed=14))
        from nightevolver.genome import INDICATOR_NAMES
        res = audit_nonlinear(md.indicators[:, :, :4], list(INDICATOR_NAMES[:4]),
                              build_targets(md.close), n_permutations=400, seed=0)
        text = res.summary()
        assert "NON-MONOTONIC" in text
        if not res.survivors:
            assert "if and" in text.lower() and "only if" in text.lower()


class TestPermutationFloor:
    """A permutation p cannot go below 1/(P+1), and BH multiplies the
    smallest p by the pair count. If n_pairs/(P+1) > alpha, NOTHING can
    be rejected however strong the dependence - the audit would report a
    null that is a property of its own configuration, not of the data.

    Measured while building this module: 104 pairs at 200 permutations
    gives q_min = 0.517. The first real run was structurally incapable of
    finding anything, and would have looked like a clean null."""

    def test_underpowered_configuration_raises(self):
        md = build_market_data(_walk(300, 3, seed=20))
        from nightevolver.genome import INDICATOR_NAMES
        with pytest.raises(ValueError, match="cannot support"):
            audit_nonlinear(md.indicators[:, :, :20], list(INDICATOR_NAMES[:20]),
                            build_targets(md.close), n_permutations=200)

    def test_the_stated_requirement_is_sufficient(self):
        """n_perm + 1 >= n_pairs / alpha must actually clear the floor."""
        for n_pairs, alpha in [(52, 0.05), (104, 0.05), (20, 0.10)]:
            need = int(np.ceil(n_pairs / alpha)) - 1
            assert n_pairs / (need + 1.0) <= alpha + 1e-12


class TestMagnitudeConfound:
    """dCor detects ANY dependence, including on a target's SCALE. A
    volatility feature scores against `r` because it predicts |r| - and
    knowing tomorrow's move will be large says nothing about which way
    to take it.

    This confound produced five apparent 'hidden directional signals' on
    the real panel, every one of them a volatility measure. Reporting
    them without decomposing sign from magnitude would have been a false
    discovery of exactly the kind this project exists to avoid."""

    def test_pure_magnitude_dependence_is_flagged(self):
        """y's SCALE depends on x, its SIGN does not. dCor must fire and
        magnitude_only must catch it."""
        from nightevolver.nonlinear_audit import NonlinearPair, _noise_floor
        rng = np.random.default_rng(30)
        n = 500
        x = np.abs(rng.normal(size=n))
        y = x * rng.normal(size=n)          # scale from x, sign independent

        d_raw = distance_correlation(x, y)
        d_abs = distance_correlation(x, np.abs(y))
        d_sgn = distance_correlation(x, np.sign(y))
        floor = _noise_floor(n, np.random.default_rng(0))

        assert d_abs > d_raw, "magnitude channel should dominate"
        assert d_sgn <= floor * 1.2, \
            f"sign channel {d_sgn:.4f} should sit at the noise floor {floor:.4f}"

        p = NonlinearPair("v", "direction_1d", d_raw, 0.001, 0.0, 0.5, 0.0, n,
                          q_value=0.01, significant=True,
                          dcor_sign=d_sgn, dcor_abs=d_abs, noise_floor=floor)
        assert p.magnitude_only, "magnitude-only dependence was not flagged"

    def test_genuine_sign_dependence_is_not_flagged(self):
        """The complement: if the SIGN really is predictable, the pair
        must survive as a real directional finding."""
        from nightevolver.nonlinear_audit import NonlinearPair, _noise_floor
        rng = np.random.default_rng(31)
        n = 500
        x = rng.normal(size=n)
        y = np.sign(x) * np.abs(rng.normal(size=n))     # sign from x

        d_sgn = distance_correlation(x, np.sign(y))
        floor = _noise_floor(n, np.random.default_rng(0))
        assert d_sgn > floor * 2, "planted sign dependence not detected"

        p = NonlinearPair("s", "direction_1d", 0.3, 0.001, 0.0, 0.5, 0.0, n,
                          q_value=0.01, significant=True,
                          dcor_sign=d_sgn, dcor_abs=0.1, noise_floor=floor)
        assert not p.magnitude_only

    def test_noise_floor_is_meaningfully_above_zero(self):
        """dCor is biased upward at finite n. Treating 0 as the reference
        makes a 0.09 look like signal when independent data scores 0.07."""
        from nightevolver.nonlinear_audit import _noise_floor
        floor = _noise_floor(588, np.random.default_rng(0))
        assert 0.04 < floor < 0.12, f"unexpected floor {floor:.4f}"
