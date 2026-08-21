"""
The kronos <-> nightevolver seam.

This wiring is worth testing hard because every way it can fail is
SILENT. A checkpoint that loads and then contributes nothing looks
identical, in a log, to a checkpoint that loads and contributes
correctly - the system trades, the report renders, and the only
difference is that months of GPU time is doing nothing. Three distinct
mechanisms produce exactly that outcome, and each has a test here:

  * ticker namespaces that do not intersect (RELIANCE vs RELIANCE.NS),
  * a price history too short for the indicators to warm up,
  * a blend weight that quietly rounds the contribution away.

The fourth hazard is the opposite - contributing too much - and is
covered by the double-Kelly test: the decoder hands back an ALREADY
SIZED weight, and feeding that into a pipeline that sizes again would
square the position.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kronos.nightevolver_bridge import NightEvolverBridge, normalise_ticker
from nightevolver.genome import GENOME_LENGTH, GENOME_VERSION, INDICATOR_NAMES

NSE_NAMES = ["RELIANCE", "TCS", "HDFCBANK", "INFY"]
KRONOS_NAMES = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"]


def write_checkpoint(path: Path, dsr: float = 0.99, tickers=None,
                     genome_version: int = GENOME_VERSION) -> Path:
    """A checkpoint in saver.py's real format."""
    rng = np.random.RandomState(7)
    payload = {
        "genome_version": genome_version,
        "genome_length": GENOME_LENGTH,
        "indicator_names": list(INDICATOR_NAMES),
        "mode": "ga",
        "trained_at": "2026-08-01T00:00:00+00:00",
        "tickers": list(tickers if tickers is not None else NSE_NAMES),
        "genome": rng.rand(GENOME_LENGTH).tolist(),
        "strategy": {},
        "metrics": {
            "in_sample": None,
            "out_of_sample": None,
            "search_budget": 5000,
            "noise_benchmark_sharpe": 0.5,
            "deflated_sharpe_prob": dsr,
            "overfitting_gap": None,
            "beats_noise": True,
        },
        "history": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def price_history(n_bars: int, names=None) -> pd.DataFrame:
    names = names or NSE_NAMES
    rng = np.random.RandomState(3)
    idx = pd.date_range("2026-01-01", periods=n_bars, freq="D")
    data = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_bars, len(names))), axis=0))
    return pd.DataFrame(data, index=idx, columns=names)


@pytest.fixture
def loaded(tmp_path):
    """A bridge with a passing checkpoint and enough history to act."""
    p = write_checkpoint(tmp_path / "ck.json")
    b = NightEvolverBridge(p, blend_weight=1.0, history_days=180)
    assert b.maybe_reload(), b.status
    b.set_history(price_history(200))
    return b


class TestTickerNamespaces:
    """The failure that matches ZERO rows, or the worse 'fix' that
    matches the WRONG ones."""

    @pytest.mark.parametrize("raw,want", [
        ("RELIANCE.NS", "RELIANCE"), ("reliance", "RELIANCE"),
        ("HDFCBANK.BO", "HDFCBANK"), ("INFY", "INFY"),
        (" tcs.ns ", "TCS"),
    ])
    def test_normalisation(self, raw, want):
        assert normalise_ticker(raw) == want

    def test_suffixed_kronos_names_match_bare_checkpoint_names(self, loaded):
        assert loaded.covered(KRONOS_NAMES) == KRONOS_NAMES

    def test_disjoint_universes_produce_no_opinion_not_a_wrong_one(self, tmp_path):
        """A checkpoint about other companies must stay silent.

        The dangerous alternative is pairing the two ticker lists by
        index, which would apply RELIANCE's evolved weights to whatever
        happens to sit in slot 0 of the live book.
        """
        p = write_checkpoint(tmp_path / "ck.json",
                             tickers=["ADANIENT", "WIPRO", "CIPLA", "TITAN"])
        b = NightEvolverBridge(p, blend_weight=1.0)
        assert b.maybe_reload()
        b.set_history(price_history(200, ["ADANIENT", "WIPRO", "CIPLA", "TITAN"]))
        assert b.raw_signals(KRONOS_NAMES) == {}
        base = [0.4, -0.3, 0.2, 0.1]
        assert b.blend(KRONOS_NAMES, base) == base


class TestTheStatisticalGate:
    def test_a_checkpoint_below_the_gate_is_refused(self, tmp_path):
        p = write_checkpoint(tmp_path / "ck.json", dsr=0.10)
        b = NightEvolverBridge(p, blend_weight=1.0)
        assert b.maybe_reload() is False
        assert "REJECTED" in b.status
        assert b.active is False

    def test_a_refused_checkpoint_leaves_the_base_signal_untouched(self, tmp_path):
        p = write_checkpoint(tmp_path / "ck.json", dsr=0.10)
        b = NightEvolverBridge(p, blend_weight=1.0)
        b.maybe_reload()
        b.set_history(price_history(200))
        base = [0.5, -0.5, 0.25, 0.0]
        assert b.blend(KRONOS_NAMES, base) == base

    def test_the_gate_can_be_waived_only_explicitly(self, tmp_path):
        p = write_checkpoint(tmp_path / "ck.json", dsr=0.10)
        b = NightEvolverBridge(p, blend_weight=1.0, require_gate=False)
        assert b.maybe_reload() is True

    def test_a_genome_version_mismatch_is_refused(self, tmp_path):
        """Gene N meaning something else is silent corruption, not an error."""
        p = write_checkpoint(tmp_path / "ck.json",
                             genome_version=GENOME_VERSION + 1)
        b = NightEvolverBridge(p, blend_weight=1.0)
        assert b.maybe_reload() is False
        assert "REJECTED" in b.status

    def test_a_missing_checkpoint_is_idle_not_an_error(self, tmp_path):
        b = NightEvolverBridge(tmp_path / "absent.json", blend_weight=1.0)
        assert b.maybe_reload() is False
        assert b.active is False
        assert b.blend(KRONOS_NAMES, [0.1, 0.2, 0.3, 0.4]) == [0.1, 0.2, 0.3, 0.4]


class TestWarmupDepth:
    """The hazard that made a naive wiring a permanent no-op.

    EvolvedStrategy needs WARMUP_BARS + 2 = 62 bars. Kronos's daily
    pipeline fetches data.lookback_days = 30. Wiring the bridge to
    DailyMemory would have returned zeros every tick, forever, behind a
    single warning line.
    """

    def test_thirty_bars_is_not_enough_and_the_bridge_says_so(self, tmp_path):
        p = write_checkpoint(tmp_path / "ck.json")
        b = NightEvolverBridge(p, blend_weight=1.0)
        b.maybe_reload()
        b.set_history(price_history(30))       # kronos's lookback_days
        assert b.active is False
        assert "too short" in b.status

    def test_short_history_contributes_nothing_rather_than_zeros(self, tmp_path):
        """Zeros are an OPINION - 'go flat'. Silence is not.

        This is the distinction that matters: a bridge that cannot see
        enough history must not be able to flatten the book.
        """
        p = write_checkpoint(tmp_path / "ck.json")
        b = NightEvolverBridge(p, blend_weight=1.0)
        b.maybe_reload()
        b.set_history(price_history(30))
        base = [0.9, -0.9, 0.5, -0.5]
        assert b.blend(KRONOS_NAMES, base) == base

    def test_enough_history_activates_it(self, loaded):
        assert loaded.active is True
        assert loaded.raw_signals(KRONOS_NAMES) != {}


class TestBlending:
    def test_weight_zero_is_observe_only(self, tmp_path):
        """The shipped default: loaded, verified, reported, trading nothing."""
        p = write_checkpoint(tmp_path / "ck.json")
        b = NightEvolverBridge(p, blend_weight=0.0)
        b.maybe_reload()
        b.set_history(price_history(200))
        base = [0.4, -0.2, 0.1, 0.3]
        assert b.blend(KRONOS_NAMES, base) == base

    def test_weight_one_replaces_on_covered_tickers(self, loaded):
        base = [0.4, -0.2, 0.1, 0.3]
        out = loaded.blend(KRONOS_NAMES, base)
        ne = loaded.raw_signals(KRONOS_NAMES)
        for i, t in enumerate(KRONOS_NAMES):
            assert out[i] == pytest.approx(ne[t])

    def test_partial_weight_is_a_convex_mix(self, tmp_path):
        p = write_checkpoint(tmp_path / "ck.json")
        b = NightEvolverBridge(p, blend_weight=0.25)
        b.maybe_reload()
        b.set_history(price_history(200))
        base = [0.8, 0.8, 0.8, 0.8]
        ne = b.raw_signals(KRONOS_NAMES)
        out = b.blend(KRONOS_NAMES, base)
        for i, t in enumerate(KRONOS_NAMES):
            assert out[i] == pytest.approx(0.75 * 0.8 + 0.25 * ne[t])

    def test_uncovered_tickers_keep_their_base_signal(self, loaded):
        """A 48-name checkpoint must not drag name 49 toward zero merely
        by having no opinion about it."""
        tickers = KRONOS_NAMES + ["SBIN.NS", "ITC.NS"]
        base = [0.1, 0.2, 0.3, 0.4, 0.55, -0.65]
        out = loaded.blend(tickers, base)
        assert out[4] == pytest.approx(0.55)
        assert out[5] == pytest.approx(-0.65)

    def test_blend_never_changes_the_vector_length(self, loaded):
        base = [0.1, 0.2, 0.3, 0.4]
        assert len(loaded.blend(KRONOS_NAMES, base)) == len(base)


class TestSizingHappensOnce:
    def test_raw_signals_are_scores_not_position_weights(self, loaded):
        """LiveSignal.target_weight is already Kelly-scaled and capped.

        Kronos multiplies by kelly_fraction * max_position_pct downstream,
        so returning target_weight here would size the position twice -
        and it would look plausible, just too small or too large by a
        factor nobody would notice in a log.
        """
        sigs = loaded.raw_signals(KRONOS_NAMES)
        decoded = loaded._strategy
        live = {normalise_ticker(s.ticker): s
                for s in decoded.signal(price_history(200))}
        for t, v in sigs.items():
            s = live[normalise_ticker(t)]
            if s.direction != 0:
                assert v == pytest.approx(s.score)
                assert v != pytest.approx(s.target_weight) or s.score == 0.0

    def test_scores_stay_in_the_signal_range(self, loaded):
        for v in loaded.raw_signals(KRONOS_NAMES).values():
            assert -1.0 <= v <= 1.0

    def test_below_conviction_floor_reads_as_zero(self, loaded):
        """An abstention must arrive as 0.0, not as a small opinion."""
        live = loaded._strategy.signal(price_history(200))
        sigs = loaded.raw_signals(KRONOS_NAMES)
        for s in live:
            if s.direction == 0:
                key = [t for t in KRONOS_NAMES
                       if normalise_ticker(t) == normalise_ticker(s.ticker)]
                if key:
                    assert sigs[key[0]] == 0.0


class TestReload:
    def test_the_same_file_is_not_re_read(self, loaded):
        assert loaded.maybe_reload() is False

    def test_a_newer_checkpoint_is_adopted(self, loaded, tmp_path):
        time.sleep(0.01)
        write_checkpoint(tmp_path / "ck.json", dsr=0.97)
        assert loaded.maybe_reload() is True

    def test_a_newer_but_failing_checkpoint_deactivates_the_bridge(
            self, loaded, tmp_path):
        """Adopting a rejected checkpoint must not leave the OLD one
        trading under a status line that says 'REJECTED'."""
        time.sleep(0.01)
        write_checkpoint(tmp_path / "ck.json", dsr=0.01)
        assert loaded.maybe_reload() is False
        assert loaded.active is False
        base = [0.3, 0.3, 0.3, 0.3]
        assert loaded.blend(KRONOS_NAMES, base) == base


class TestConfigConstruction:
    def _cfg(self, block):
        class C:
            def get(self, name, default=None):
                return block if name == "nightevolver" else default
        return C()

    def test_absent_block_yields_no_bridge(self):
        class C:
            def get(self, name, default=None):
                return default
        assert NightEvolverBridge.from_config(C()) is None

    def test_disabled_yields_no_bridge(self):
        assert NightEvolverBridge.from_config(
            self._cfg({"enabled": False})) is None

    def test_enabled_builds_with_the_configured_values(self):
        b = NightEvolverBridge.from_config(self._cfg({
            "enabled": True, "checkpoint_path": "/tmp/x.json",
            "blend_weight": 0.3, "history_days": 200, "require_gate": True,
        }))
        assert b is not None
        assert b.blend_weight == 0.3
        assert b.history_days == 200
        assert b.require_gate is True

    def test_the_shipped_config_has_it_off(self):
        """The default must stay observe-only: this project's own audit
        could not establish a directional edge, so an accidental
        enable-by-default would trade on a finding that does not exist."""
        import yaml
        cfg = yaml.safe_load(Path("kronos/config.yaml").read_text())
        ne = cfg.get("nightevolver")
        assert ne is not None, "config block missing"
        assert ne["enabled"] is False
        assert ne["blend_weight"] == 0.0
        assert ne["require_gate"] is True
        assert ne["history_days"] > 62, "history must exceed WARMUP_BARS + 2"

    def test_blend_weight_is_clamped(self, tmp_path):
        p = write_checkpoint(tmp_path / "ck.json")
        assert NightEvolverBridge(p, blend_weight=5.0).blend_weight == 1.0
        assert NightEvolverBridge(p, blend_weight=-2.0).blend_weight == 0.0
