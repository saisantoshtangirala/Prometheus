"""
Tests for the new-data / new-target work.

Each test is written so it can actually FAIL if the thing it covers is
wrong. The look-ahead tests in particular are the ones that matter: a
one-day leak in a flow feature or a target would manufacture an edge
that looks spectacular and is not real, and it would do so silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nightevolver.flows import (
    FLOW_FEATURE_NAMES, N_FLOW_FEATURES, PUBLISH_LAG_BARS, ParticipantDay,
    align_flow_features, build_flow_frame, parse_participant_csv,
)
from nightevolver.information_audit import (
    audit_features, benjamini_hochberg, _block_permutation_indices,
)
from nightevolver.targets import (
    TARGET_NAMES, build_targets, persistence_baseline,
)

# A real NSCCL participant-volume body, trimmed to the columns used.
SAMPLE_CSV = (
    '""Participant wise Trading Volume (no. of contracts) in Equity Derivatives '
    'as on Aug 19, 2026"",,,,,,,,,,,,,,\n'
    "Client Type,Future Index Long,Future Index Short,Future Stock Long,"
    "Future Stock Short       ,Option Index Call Long,Option Index Put Long,"
    "Option Index Call Short,Option Index Put Short,Option Stock Call Long,"
    "Option Stock Put Long,Option Stock Call Short,Option Stock Put Short,"
    "Total Long Contracts      ,Total Short Contracts\n"
    "Client,44546,32934,472480,463908,9468033,9497123,9490710,9512542,"
    "1886455,828432,1906691,840164,22197069,22246949\n"
    "DII,401,3905,271007,281905,0,2396,0,5500,7442,2105,10773,2227,283351,304310\n"
    "FII,8934,24583,411499,399725,1743000,1507569,1824925,1547024,"
    "262663,149793,278137,149336,4083458,4223730\n"
    "Pro,25444,17903,389362,398810,12464576,10453283,12359974,10395305,"
    "3675913,2085518,3636872,2074121,29094096,28882985\n"
    "TOTAL,79325,79325,1544348,1544348,23675609,21460371,23675609,21460371,"
    "5832473,3065848,5832473,3065848,55657974,55657974\n"
)


# --------------------------------------------------------------------------
# flows: parsing
# --------------------------------------------------------------------------

def test_parses_real_participant_csv_including_whitespace_headers():
    day = parse_participant_csv(SAMPLE_CSV, pd.Timestamp("2026-08-19"))
    assert day is not None
    # "Future Stock Short       " has trailing spaces in the real file;
    # if _norm() regressed, this lookup returns NaN.
    assert day.get("FII", "Future Stock Short") == 399725.0
    assert day.get("FII", "Future Index Long") == 8934.0
    assert day.get("Client", "Future Index Long") == 44546.0


def test_parse_returns_none_on_junk_rather_than_raising():
    assert parse_participant_csv("not,a,participant,file\n1,2,3,4\n",
                                 pd.Timestamp("2026-08-19")) is None
    assert parse_participant_csv("", pd.Timestamp("2026-08-19")) is None


def test_net_ratio_sign_matches_the_real_market_structure():
    """FII net-short index futures against retail net-long is the known
    structural picture on NSE; if the long/short fields were swapped the
    signs would invert and this catches it."""
    frame = build_flow_frame({pd.Timestamp("2026-08-19"):
                              parse_participant_csv(SAMPLE_CSV,
                                                    pd.Timestamp("2026-08-19"))})
    assert frame["fii_idx_fut_net"].iloc[0] < 0      # FII 8934 long vs 24583 short
    assert frame["client_idx_fut_net"].iloc[0] > 0   # Client 44546 vs 32934


def test_flow_frame_has_exactly_the_declared_columns():
    frame = build_flow_frame({pd.Timestamp("2026-08-19"):
                              parse_participant_csv(SAMPLE_CSV,
                                                    pd.Timestamp("2026-08-19"))})
    assert tuple(frame.columns) == FLOW_FEATURE_NAMES
    assert len(FLOW_FEATURE_NAMES) == N_FLOW_FEATURES


# --------------------------------------------------------------------------
# flows: look-ahead. This is the important one.
# --------------------------------------------------------------------------

def _fake_flow_frame(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {name: rng.normal(size=n) for name in FLOW_FEATURE_NAMES}, index=idx)


def test_flow_alignment_refuses_zero_lag():
    """A zero lag would apply a report published after the close of bar t
    to bar t. That is look-ahead and must raise, not warn."""
    frame = _fake_flow_frame()
    with pytest.raises(ValueError, match="look-ahead"):
        align_flow_features(frame, frame.index, lag_bars=0)


def test_flow_features_do_not_see_the_future():
    """Mutate flow values at and after a cut point; every aligned row
    strictly before the cut must be unchanged."""
    frame = _fake_flow_frame()
    dates = frame.index
    base = align_flow_features(frame, dates)

    cut = 200
    tampered = frame.copy()
    tampered.iloc[cut:] += 50.0
    after = align_flow_features(tampered, dates)

    # Row t uses the report from t-PUBLISH_LAG_BARS, so rows before
    # cut+lag are the ones that must be untouched.
    safe = cut + PUBLISH_LAG_BARS - 1
    np.testing.assert_allclose(base[:safe], after[:safe], atol=1e-12)
    assert not np.allclose(base[cut + PUBLISH_LAG_BARS:],
                           after[cut + PUBLISH_LAG_BARS:]), \
        "mutating the future changed nothing later - the feature is inert"


def test_flow_lag_actually_shifts_by_one_bar():
    """A distinctive spike must appear exactly PUBLISH_LAG_BARS later."""
    frame = _fake_flow_frame()
    frame.iloc[:] = 0.0
    frame.iloc[100] = 25.0
    out = align_flow_features(frame, frame.index)
    col = out[:, 0]
    assert abs(col[100]) < 1e-9, "spike leaked into the same bar"
    assert abs(col[100 + PUBLISH_LAG_BARS]) > 0.0, "spike never arrived"


def test_flow_features_are_bounded_and_finite():
    out = align_flow_features(_fake_flow_frame(), _fake_flow_frame().index)
    assert np.isfinite(out).all()
    assert out.min() >= -1.0 and out.max() <= 1.0


# --------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------

def _close(n=400, a=5, seed=3):
    rng = np.random.default_rng(seed)
    return 100 * np.cumprod(1 + rng.normal(0, 0.012, size=(n, a)), axis=0)


def test_build_targets_produces_all_declared_targets():
    tg = build_targets(_close())
    assert set(tg) == set(TARGET_NAMES)


def test_targets_are_strictly_forward_looking():
    """target[t] must depend only on bars after t.

    Mutate bar k and assert earlier rows are untouched. The safe cut is
    k - horizon - 1, not k - 1: vol_5d[t] legitimately reads bars t+1..t+5,
    so row k-5 SHOULD move when bar k moves. Asserting k-1 here would be
    testing the wrong thing and would fail on a correct implementation.
    """
    horizon = 5
    close = _close()
    base = build_targets(close, horizon=horizon)
    k = 250
    tampered = close.copy()
    tampered[k:] *= 1.10
    after = build_targets(tampered, horizon=horizon)

    safe = k - horizon - 1
    for name in TARGET_NAMES:
        b, a = base[name], after[name]
        np.testing.assert_allclose(
            b.values[:safe], a.values[:safe], atol=1e-10,
            err_msg=f"{name} changed at a row that cannot see the mutated bar")


def test_targets_do_react_to_their_own_forward_window():
    """The complement of the test above: a target that ignores the future
    entirely would pass a no-look-ahead test trivially."""
    close = _close()
    base = build_targets(close)
    tampered = close.copy()
    tampered[251:] *= 1.30
    after = build_targets(tampered)
    assert not np.allclose(base["vol_5d"].values[246:250],
                           after["vol_5d"].values[246:250]), \
        "vol_5d ignored a large change inside its own forward window"


def test_vol_target_invalidates_its_tail():
    """vol_5d needs 5 future bars; the last 5 rows must be invalid, not
    silently filled with zeros."""
    tg = build_targets(_close(n=200))
    v = tg["vol_5d"]
    assert not v.valid[-5:].any(), "tail rows claim validity without future bars"
    assert v.valid[:-5].all()


def test_rel_strength_sums_to_zero_across_assets():
    tg = build_targets(_close())
    rel = tg["rel_strength_1d"]
    rows = rel.valid.all(axis=1)
    np.testing.assert_allclose(rel.values[rows].sum(axis=1), 0.0, atol=1e-10)


def test_rel_strength_degenerates_loudly_with_one_asset():
    tg = build_targets(_close(a=1))
    assert np.allclose(tg["rel_strength_1d"].values, 0.0)


def test_volatility_is_flagged_as_persistence_prone_and_direction_is_not():
    """The audit relies on this flag to decide whether a high raw score
    is cheap. If it inverts, the interpretation inverts with it."""
    tg = build_targets(_close())
    assert tg["vol_5d"].autocorr_baseline is True
    assert tg["direction_1d"].autocorr_baseline is False


def test_persistence_baseline_uses_only_the_past():
    close = _close()
    tg = build_targets(close)
    k = 250
    tampered = close.copy()
    tampered[k:] *= 1.25
    for name in TARGET_NAMES:
        b = persistence_baseline(tg[name], close)
        a = persistence_baseline(tg[name], tampered)
        np.testing.assert_allclose(
            b[:k], a[:k], atol=1e-10,
            err_msg=f"persistence baseline for {name} looked into the future")


def test_trailing_vol_actually_predicts_forward_vol():
    """Sanity check on the baseline itself: on a GARCH-like series with
    real vol clustering, trailing vol must correlate with forward vol.
    If this fails the baseline is broken and 'incremental' is meaningless."""
    rng = np.random.default_rng(0)
    n = 1200
    vol = np.empty(n)
    vol[0] = 0.01
    for t in range(1, n):
        vol[t] = np.sqrt(0.02e-4 + 0.9 * vol[t - 1] ** 2
                         + 0.08 * (rng.normal(0, vol[t - 1])) ** 2)
    rets = rng.normal(0, vol)
    close = (100 * np.cumprod(1 + rets))[:, None]
    tg = build_targets(close)
    v = tg["vol_5d"]
    base = persistence_baseline(v, close)
    rows = v.valid[:, 0] & np.isfinite(base[:, 0]) & (base[:, 0] > 0)
    r = np.corrcoef(base[rows, 0], v.values[rows, 0])[0, 1]
    # 0.15 rather than a larger number because 5-day realised vol is a
    # noisy estimator of itself: even under a strongly persistent GARCH,
    # the correlation between trailing and forward 5-day realised vol
    # sits near 0.2. Measured 0.207 on this series. The claim under test
    # is that the baseline is INFORMATIVE, not that it is strong.
    assert r > 0.15, f"trailing vol barely predicts forward vol (r={r:.3f})"


# --------------------------------------------------------------------------
# information audit
# --------------------------------------------------------------------------

def test_benjamini_hochberg_matches_known_values():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
    q, rejected = benjamini_hochberg(p, alpha=0.05)
    assert rejected[:2].all()
    assert not rejected[-1]
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12), "q-values not monotone"


def test_bh_rejects_nothing_when_all_p_are_large():
    _, rejected = benjamini_hochberg([0.4, 0.5, 0.6, 0.9], alpha=0.05)
    assert not rejected.any()


def test_block_permutation_preserves_contiguity():
    """The blocks are the whole point: an iid shuffle would destroy the
    autocorrelation the null is supposed to keep."""
    rng = np.random.default_rng(0)
    idx = _block_permutation_indices(100, 10, rng)
    assert sorted(idx) == list(range(100))
    # count adjacent pairs that stayed adjacent; blocks of 10 guarantee many
    adjacent = int(np.sum(np.diff(idx) == 1))
    assert adjacent >= 80, f"only {adjacent} contiguous steps - blocks broken"


def test_iid_permutation_when_block_is_one():
    rng = np.random.default_rng(0)
    idx = _block_permutation_indices(200, 1, rng)
    assert sorted(idx) == list(range(200))


def _noise_features(T, A, F, seed=0, autocorr=0.9):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(T, A, F))
    for i in range(F):
        for a in range(A):
            s = f[:, a, i]
            for t in range(1, T):
                s[t] = autocorr * s[t - 1] + np.sqrt(1 - autocorr ** 2) * s[t]
    return f


def test_audit_finds_nothing_in_pure_noise():
    """THE headline calibration test. Autocorrelated features against a
    random walk must not produce survivors. If this ever fails, the null
    is wrong and every 'discovery' downstream is suspect."""
    T, A, F = 700, 6, 6
    rng = np.random.default_rng(11)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.012, size=(T, A)), axis=0)
    feats = _noise_features(T, A, F, seed=5)
    res = audit_features(feats, [f"n{i}" for i in range(F)],
                         build_targets(close), close,
                         n_permutations=300, seed=2)
    assert len(res.survivors) == 0, \
        f"false positives on pure noise: {[str(p) for p in res.survivors]}"


def test_audit_recovers_a_planted_signal():
    """The other half of calibration: a detector that never fires is
    useless. Plant a genuine predictor and require it to be found."""
    T, A, F = 700, 6, 6
    rng = np.random.default_rng(3)
    signal = rng.normal(size=(T, A))
    rets = 0.012 * (0.3 * signal + np.sqrt(1 - 0.09) * rng.normal(size=(T, A)))
    close = 100 * np.cumprod(1 + rets, axis=0)
    feats = _noise_features(T, A, F, seed=6)
    feats[:-1, :, 2] = signal[1:]          # feature at t drives return t->t+1

    res = audit_features(feats, [f"n{i}" for i in range(F)],
                         build_targets(close), close,
                         n_permutations=300, seed=2)
    found = {p.feature for p in res.survivors if p.target == "direction_1d"}
    assert "n2" in found, f"planted predictor not recovered; survivors={found}"
    assert found <= {"n2"}, f"flagged decoys as well: {found}"


def test_audit_rejects_mismatched_feature_names():
    close = _close(n=200, a=3)
    feats = _noise_features(200, 3, 4)
    with pytest.raises(ValueError, match="names for"):
        audit_features(feats, ["a", "b"], build_targets(close), close,
                       n_permutations=10)


def test_no_mechanical_artefact_between_vol_indicators_and_vol_targets():
    """REGRESSION TEST for a false positive that a random-feature null
    could not catch.

    The bug: regime_shift_5d = log(fwd_vol / trail_vol) has trailing vol
    in its denominator, and atr_pct IS trailing vol. So they correlate
    mechanically, with no information involved. On a pure random walk the
    audit reported atr_pct -> regime_shift_5d at incremental rho = -0.3864,
    q = 0.04 - a significant finding on structureless data.

    It survived the earlier noise test because that test used RANDOM
    features. The artefact only appears when the features are computed
    from the same price series as the target, which is exactly the real
    configuration. So this test builds real indicators from a random walk.

    If this fails, the volatility results are artefacts and must not be
    reported as findings.
    """
    import pandas as pd
    from nightevolver.data_loader import build_market_data
    from nightevolver.genome import INDICATOR_NAMES

    rng = np.random.default_rng(0)
    n, a = 900, 6
    px = 100 * np.cumprod(1 + rng.normal(0, 0.012, size=(n, a)), axis=0)
    idx = pd.bdate_range("2022-01-01", periods=n)
    md = build_market_data(pd.DataFrame(
        px, index=idx, columns=[f"S{i}" for i in range(a)]))

    res = audit_features(md.indicators, list(INDICATOR_NAMES),
                         build_targets(md.close), md.close,
                         n_permutations=400, seed=0)
    offenders = [str(p) for p in res.survivors]
    assert not offenders, (
        "audit found structure in a random walk using real indicators - "
        f"these are mechanical artefacts, not signal:\n  "
        + "\n  ".join(offenders))


def test_audit_summary_states_the_null_result_plainly():
    T, A, F = 400, 4, 3
    rng = np.random.default_rng(1)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.012, size=(T, A)), axis=0)
    res = audit_features(_noise_features(T, A, F), ["a", "b", "c"],
                         build_targets(close), close, n_permutations=100, seed=0)
    text = res.summary()
    assert "NOTHING survives" in text
    assert "new inputs" in text


class TestTargetValidityIsPerAssetNotPerDate:
    """THE TARGET THAT VANISHED.

    vol_5d and regime_shift_5d gated each date on
    `np.isfinite(window).all()`, which tests the whole [horizon, A] slice
    across EVERY asset. On a dense panel that is harmless. On a ragged
    one - which is what an honest panel looks like once delisted names
    stop being forward-filled - a single name with a gap invalidates that
    date for all the others.

    Measured on the 2019 top-100, 1,764 x 100:

        direction_1d      valid 93.6%
        rel_strength_1d   valid 93.6%
        vol_5d            valid  0.0%
        regime_shift_5d   valid  0.0%

    The walk-forward did not crash. It reported no windows for those two
    targets and carried on, which reads as "no signal" and is in fact
    "no data" - the most expensive kind of silent failure here, since
    atm_iv -> vol_5d is the one live result in the project.
    """

    def _ragged(self, T=300, A=5, seed=7):
        rng = np.random.RandomState(seed)
        close = 100 * np.cumprod(1 + rng.normal(0, 0.012, (T, A)), axis=0)
        close[150:, 0] = np.nan          # one name delists mid-panel
        return close

    def test_one_dead_name_does_not_invalidate_the_others(self):
        close = self._ragged()
        for name, t in build_targets(close, horizon=5).items():
            live = t.valid[:, 1:]        # every name except the dead one
            assert live.mean() > 0.90, (
                f"{name} valid on only {live.mean():.1%} of live cells - "
                "validity is being computed per DATE, not per ASSET")

    def test_the_dead_name_itself_is_invalid_after_death(self):
        """The other half: it must not be quietly filled either."""
        close = self._ragged()
        for name, t in build_targets(close, horizon=5).items():
            assert not t.valid[160:, 0].any(), \
                f"{name} is valid for a name that stopped trading"

    def test_vol_5d_values_match_a_dense_panel_where_both_are_defined(self):
        """Per-asset gating must not change any number that was already
        correct - only widen where it applies."""
        close = self._ragged()
        dense = close[:, 1:]             # same names, no ragged column
        rag = build_targets(close, horizon=5)["vol_5d"]
        den = build_targets(dense, horizon=5)["vol_5d"]
        both = rag.valid[:, 1:] & den.valid
        assert both.sum() > 1000
        assert np.allclose(rag.values[:, 1:][both], den.values[both])

    def test_the_persistence_baseline_is_also_per_asset(self):
        """persistence_baseline carried the same pattern in two more
        places. A baseline that is NaN everywhere makes every
        incremental score fall back to raw, silently undoing the
        partialling that vol targets depend on."""
        close = self._ragged()
        targets = build_targets(close, horizon=5)
        for name in ("vol_5d", "regime_shift_5d"):
            base = persistence_baseline(targets[name], close, horizon=5)
            live = np.isfinite(base[:, 1:]) & (base[:, 1:] != 0.0)
            assert live.mean() > 0.80, (
                f"{name} baseline is empty on {1 - live.mean():.1%} of live "
                "cells - the per-date gate is still there")
