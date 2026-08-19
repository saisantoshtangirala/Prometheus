"""
Tests for kronos/backtest_snn.py - the walk-forward backtest of the
ACTUAL production model (ReflexArc's SNN, trained via
prometheus.engine.PrometheusEngine's real pretrain->finetune->meta
procedure), as opposed to kronos/backtest.py's separate KronosStrategy.

Tiny config throughout (1 pretrain epoch, 1 meta epoch, 1 finetune epoch,
2-3 tickers, small windows) so this runs in reasonable CI time - this is
a wiring/correctness test, not a real research run. The real, full-scale
run happens on Hetzner via .github/workflows (real data, real compute
budget), not here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from kronos.backtest import SignalDiagnostic, synthetic_history
from kronos.backtest_snn import SNNTrainConfig, SNNWalkForwardBacktester
from kronos.backtest import WalkForwardConfig


TINY_TRAIN_CFG = SNNTrainConfig(
    seq_len=8, horizon=3, d_model=16, n_heads=2, n_layers=1,
    pretrain_epochs=1, finetune_epochs=1, meta_epochs=1,
    n_black_swans=10, batch_size=4, device="cpu", seed=1,
)


@pytest.fixture
def closes():
    return synthetic_history(["AAA", "BBB"], n_days=120, seed=3)


@pytest.fixture
def bt(closes):
    return SNNWalkForwardBacktester(
        closes, tickers=["AAA", "BBB"],
        config=WalkForwardConfig(train_window=40, test_window=15),
        train_cfg=TINY_TRAIN_CFG,
    )


class TestWindows:
    def test_windows_match_backtest_py_semantics(self, bt):
        """Same no-look-ahead index arithmetic as
        kronos/backtest.py's WalkForwardBacktester.windows()."""
        for (s, e, te) in bt.windows():
            assert s < e <= te

    def test_consecutive_windows_advance_by_test_window(self, bt):
        spans = bt.windows()
        for (s1, _, _), (s2, _, _) in zip(spans, spans[1:]):
            assert s2 - s1 == bt.cfg.test_window


class TestEndToEnd:
    def test_single_window_runs_and_returns_valid_diagnostic(self, bt):
        diag = bt.run_signal_diagnostic(max_windows=1)
        assert isinstance(diag, SignalDiagnostic)
        assert diag.strategy == "snn"
        assert diag.n_obs > 0
        assert 0.0 <= diag.hit_rate <= 1.0
        assert np.isfinite(diag.pearson_r)
        assert np.isfinite(diag.spearman_r)
        assert set(diag.per_ticker.keys()) == {"AAA", "BBB"}

    def test_two_windows_uses_a_fresh_finetune_each_time(self, bt):
        """Each window must independently start from the shared
        pretrain+meta baseline - not carry state from the previous
        window. Verified indirectly: running 2 windows must not raise
        and must produce roughly double the observations of 1 window."""
        diag_one = bt.run_signal_diagnostic(max_windows=1)
        diag_two = bt.run_signal_diagnostic(max_windows=2)
        assert diag_two.n_obs > diag_one.n_obs

    def test_evaluation_goes_through_a_real_reflex_arc(self, bt, monkeypatch):
        """The evaluation path must construct and call the actual
        kronos.reflex.ReflexArc.infer(), not a stand-in - spy on it."""
        from kronos import reflex as reflex_module
        calls = {"n": 0}
        real_infer = reflex_module.ReflexArc.infer

        def spying_infer(self, *args, **kwargs):
            calls["n"] += 1
            return real_infer(self, *args, **kwargs)

        monkeypatch.setattr(reflex_module.ReflexArc, "infer", spying_infer)
        bt.run_signal_diagnostic(max_windows=1)
        assert calls["n"] > 0

    def test_calibrate_size_scale_called_with_train_only_data(self, bt, monkeypatch):
        """Matches production's real post-adoption behavior
        (KronosOrchestrator.maybe_adopt_runpod_checkpoint()) - and must
        never see test-window data (no look-ahead)."""
        from kronos import reflex as reflex_module
        seen_shapes = []
        real_calibrate = reflex_module.ReflexArc.calibrate_size_scale

        def spying_calibrate(self, recent_returns):
            seen_shapes.append(recent_returns.shape[0])
            return real_calibrate(self, recent_returns)

        monkeypatch.setattr(reflex_module.ReflexArc, "calibrate_size_scale", spying_calibrate)
        bt.run_signal_diagnostic(max_windows=1)
        assert seen_shapes == [bt.cfg.train_window]

    def test_snn_weights_actually_change_after_finetune(self, bt):
        """A no-op finetune (e.g. broken data wiring) would leave the
        pretrain+meta baseline's weights untouched - catch that
        directly rather than trust the diagnostic's output alone."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            bt._build_baseline_checkpoint(d)
            from prometheus.engine import PrometheusEngine, PrometheusConfig
            baseline_engine = PrometheusEngine(PrometheusConfig(
                n_assets=2, seq_len=TINY_TRAIN_CFG.seq_len, horizon=TINY_TRAIN_CFG.horizon,
                d_model=TINY_TRAIN_CFG.d_model, n_heads=TINY_TRAIN_CFG.n_heads,
                n_layers=TINY_TRAIN_CFG.n_layers, device="cpu",
                snn_layer_sizes=[32, 16], snn_output_size=2,
            ))
            baseline_engine.load(d)
            baseline_snn_state = {k: v.clone() for k, v in baseline_engine.snn.state_dict().items()}

            rets = bt.returns.values
            s, e, te = bt.windows()[0]
            finetuned_state = bt._finetune_window(d, rets[s:e])

            changed = any(
                not torch.equal(baseline_snn_state[k], finetuned_state[k])
                for k in baseline_snn_state
            )
            assert changed, "finetune must actually update snn weights, not no-op"
