"""
The one-bar look-ahead that manufactured a directional edge.

WHAT HAPPENED. run_target_walkforward feeds extra data channels
(derivatives, delivery) alongside the price-derived indicators.
build_market_data keeps slice(WARMUP_BARS, len-1) - 60 bars off the
front and ONE off the back - so the extras have to be cut the same way.
They were cut by length arithmetic instead:

    trim = len(ex) - md.n_bars        # = 61
    ex = ex[trim:trim + md.n_bars]    # 61 off the front, NONE off the back

which shifts every extra channel one bar EARLY against the prices, so
feature[t] holds day t+1's data.

WHAT IT PRODUCED. A result that looked like the thing this project has
been trying to find for months:

    pcr_volume -> direction_1d     OOS |rho| = 0.3042, null cloud
                                   [0.2763, 0.2892], p = 0.032
                                   ABOVE the cloud, 24/24 windows

Measured against the correctly-aligned series, the same feature:

    pcr_volume vs same-bar return (t-1 -> t)   rho = -0.2826
    pcr_volume vs FORWARD return (t -> t+1)    rho = +0.0239

The entire effect is contemporaneous. High put volume accompanies down
days - an everyday fact about how people trade options - and dating it
one bar early turns it into a forecast. The tell was visible before the
fix and is worth naming: a feature FIVE TIMES more correlated with the
future than with the present is not a predictor, it is a calendar error.
Genuine predictors are weaker on the future, not stronger.

The null cloud did not catch it, and could not have: extras ride the
same permutation as prices by design, so a same-row relationship is
preserved in every draw. The cloud sat at 0.283 rather than near zero,
which was itself the visible symptom - a null that far from zero means
the statistic is measuring something mechanical.
"""

from __future__ import annotations

import numpy as np
import pytest

from nightevolver.data_loader import WARMUP_BARS, build_market_data


def _panel(T=400, A=6, seed=3):
    import pandas as pd
    rng = np.random.RandomState(seed)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.012, (T, A)), axis=0)
    idx = pd.bdate_range("2020-01-01", periods=T)
    return pd.DataFrame(close, index=idx, columns=[f"S{i}" for i in range(A)])


class TestExtraChannelAlignment:
    def test_the_slice_matches_what_build_market_data_keeps(self):
        """The arithmetic identity the old code got wrong: 60 off the
        front and 1 off the back is NOT the same as 61 off the front."""
        close = _panel()
        md = build_market_data(close)
        T = len(close)
        assert md.n_bars == T - WARMUP_BARS - 1

        old_trim = T - md.n_bars                     # 61
        assert old_trim == WARMUP_BARS + 1
        assert old_trim != WARMUP_BARS, \
            "the front trim must not absorb the back trim"

    def test_a_marker_channel_lands_on_the_bar_it_belongs_to(self):
        """A channel carrying its own row index must still carry it after
        trimming. If it is off by one, feature[t] holds t+1."""
        close = _panel()
        md = build_market_data(close)
        T, A = close.shape
        marker = np.tile(np.arange(T, dtype=float)[:, None], (1, A))

        lo = WARMUP_BARS
        aligned = marker[lo:lo + md.n_bars]
        assert aligned[0, 0] == WARMUP_BARS
        assert aligned[-1, 0] == T - 2, \
            "the last kept bar must be len-2, since build_market_data " \
            "drops the final bar (it has no forward return)"

        old = marker[T - md.n_bars: T - md.n_bars + md.n_bars]
        assert old[0, 0] == WARMUP_BARS + 1
        assert np.all(old[:, 0] - aligned[:, 0] == 1), \
            "the old slice is exactly one bar ahead - the look-ahead"

    def test_the_script_slices_by_warmup_not_by_length(self):
        """Guards the specific regression. Length arithmetic reads as
        equivalent and is not."""
        import inspect
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        import run_target_walkforward as W

        # Checked on the AST, not the text: the source COMMENT quotes the
        # old expression on purpose, and a string search cannot tell a
        # warning about a bug from the bug.
        import ast
        tree = ast.parse(inspect.getsource(W._one_draw))
        # ALL assignments per name, not the last one: `ex` is rebound
        # several times in this function and a dict silently keeps only
        # one of them.
        assigns = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                tgt = node.targets[0]
                if isinstance(tgt, ast.Name):
                    assigns.setdefault(tgt.id, []).append(
                        ast.unparse(node.value))

        assert "WARMUP_BARS" in assigns.get("lo", []), (
            f"extras must be sliced from WARMUP_BARS, got "
            f"lo={assigns.get('lo')!r}")
        assert not any("len(ex) - md.n_bars" in v
                       for v in assigns.get("trim", [])), \
            "the length-arithmetic trim is back; extras are one bar early"
        assert "ex[lo:lo + md.n_bars]" in assigns.get("ex", []), \
            "the extras are not sliced by the WARMUP_BARS offset"


class TestTheDiagnosticThatCatchesThis:
    """A shifted feature is detectable WITHOUT knowing the bug: compare
    its correlation with the past bar against the future bar. This is
    cheap and belongs in front of any claimed edge."""

    def _shifted_feature(self, close, lead=0):
        c = close.to_numpy()
        r = np.full_like(c, np.nan)
        r[1:] = c[1:] / c[:-1] - 1.0
        f = -r.copy()                       # contemporaneous with returns
        if lead:                            # move future data into t
            f = np.vstack([f[lead:], np.full((lead, c.shape[1]), np.nan)])
        return f

    def _rhos(self, close, f):
        from scipy import stats
        c = close.to_numpy()
        r = np.full_like(c, np.nan)
        r[1:] = c[1:] / c[:-1] - 1.0
        fwd = np.full_like(c, np.nan)
        fwd[:-1] = c[1:] / c[:-1] - 1.0
        out = []
        for t in (r, fwd):
            m = np.isfinite(f) & np.isfinite(t)
            out.append(abs(stats.spearmanr(f[m], t[m]).statistic))
        return out            # (same-bar, forward)

    def test_an_honest_feature_is_stronger_on_the_present(self):
        close = _panel()
        same, fwd = self._rhos(close, self._shifted_feature(close, lead=0))
        assert same > fwd, "a contemporaneous feature must lead on same-bar"

    def test_a_one_bar_lead_flips_the_comparison(self):
        """The exact fingerprint measured on pcr_volume: near-zero on the
        present, large on the future."""
        close = _panel()
        same, fwd = self._rhos(close, self._shifted_feature(close, lead=1))
        assert fwd > 3 * max(same, 1e-9), (
            f"a one-bar-early feature should dominate on the forward "
            f"return (same={same:.4f} fwd={fwd:.4f})")
