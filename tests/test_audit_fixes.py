"""
Regression tests for the deep-audit fixes (AUDIT-1A through AUDIT-3B).

Each test targets the specific mechanism the audit identified as a
concrete, verified bug - not the strategy's end-to-end performance (which
needs a real walk-forward backtest on real data to assess, not a unit
test). See the fixed files' own AUDIT-* comments for the full mechanism
and rationale of each fix.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from kronos.backtest import KronosStrategy, synthetic_history
from kronos.evolver import KronosEvolver
from kronos.nightmare_generator import NightmareBuffer
from prometheus.meta.maml_engine import MAMLMetaLearner


# ---------------------------------------------------------------------------
# AUDIT-1A: NEAT fitness function leaking the target into its own input
# ---------------------------------------------------------------------------

class TestNoTargetLeakage:
    def test_kronos_strategy_target_bar_not_inside_input_window(self):
        """kronos/backtest.py's KronosStrategy.fit() used to build X from a
        `horizon`-bar future and reuse THAT SAME future's own last bar as
        the fitness target y - so y was already sitting inside X. Fixed by
        bootstrapping one bar longer (horizon+1) and using the extra bar,
        which never enters X, as the target. This test re-derives X/y the
        same way fit() does and asserts the target is a genuinely
        different bar from anything X contains, not literally checking
        floating-point membership (which the old code, with jitter,
        would have made feel unbroken)."""
        strategy = KronosStrategy(
            horizon=3, population=4, generations=1, top_k=2,
            n_futures=8, seed=1,
        )
        rng = np.random.default_rng(0)
        train_returns = rng.normal(0, 0.01, size=(120, 2)).astype(np.float32)

        futures = strategy._bootstrap_futures(train_returns, length=strategy.horizon + 1)
        assert futures.shape[1] == strategy.horizon + 1, (
            "the NEAT fitness eval must bootstrap one bar MORE than the "
            "model's own input window (horizon), so a genuinely-unseen "
            "bar is available as the target"
        )

        X_val = futures[:, :strategy.horizon, :].reshape(futures.shape[0], -1)
        y_val = futures[:, strategy.horizon, :]

        n_assets = train_returns.shape[1]
        assert X_val.shape[1] == n_assets * strategy.horizon

        # The old bug's exact shape: X's own last bar (index horizon-1)
        # must NOT equal the target (index horizon) - they're different
        # historical blocks with independent bootstrap jitter, so an
        # exact/near match here would mean the target leaked back in.
        x_last_bar = futures[:, strategy.horizon - 1, :]
        assert not torch.allclose(x_last_bar, y_val, atol=1e-9), (
            "target bar must be disjoint from X's own last bar - if this "
            "fails, the leak is back"
        )

    def test_kronos_strategy_fit_produces_valid_model(self):
        """End-to-end: fit() must still run and produce a working model
        with the corrected (non-leaking) fitness data - this is a wiring
        check, not a performance claim."""
        strategy = KronosStrategy(
            horizon=3, population=4, generations=1, top_k=2,
            n_futures=8, seed=1,
        )
        rng = np.random.default_rng(0)
        train_returns = rng.normal(0, 0.01, size=(120, 2)).astype(np.float32)
        strategy.fit(train_returns)
        assert strategy.model is not None
        w = strategy.weights_for(train_returns[-10:])
        assert w.shape == (2,)
        assert np.all(np.isfinite(w))

    def test_kronos_evolver_input_dim_excludes_last_bar(self):
        """kronos/evolver.py's KronosEvolver._nightmare_val_data() had the
        identical leak: X = ALL of a `horizon`-bar future flattened, y =
        that future's own last bar. NightmareBuffer.futures can't be made
        one bar longer without touching the diffusion pipeline's seq_len
        (a much bigger change), so the fix instead holds out the buffer's
        last bar from X entirely: input_dim shrinks to (horizon-1)*n_assets
        and X only ever sees bars [0, T-2)."""
        from kronos.config import load_config
        cfg = load_config()
        cfg.override("data.tickers", ["A", "B"])
        cfg.override("nightmare.horizon_days", 4)
        evolver = KronosEvolver(cfg)

        n_assets = 2
        horizon = 4
        assert evolver.input_dim == n_assets * (horizon - 1), (
            "input_dim must be sized for (horizon-1) bars, not horizon, "
            "so the fitness eval can hold out a genuinely-unseen target bar"
        )

        n_futures = 6
        futures = torch.randn(n_futures, horizon, n_assets)
        buffer = NightmareBuffer(
            futures=futures,
            portfolio_pnl=torch.zeros(n_futures),
            condition_vector=torch.zeros(1),
        )
        X, y = evolver._nightmare_val_data(buffer)

        assert X.shape == (n_futures, n_assets * (horizon - 1))
        assert y.shape == (n_futures, n_assets)
        # X must be built ONLY from bars [0, horizon-2] (i.e. exclude the
        # buffer's last bar, index horizon-1, which is exactly `y`).
        expected_X = futures[:, :-1, :].reshape(n_futures, -1)
        assert torch.equal(X, expected_X)
        assert torch.equal(y, futures[:, -1, :])
        # The old bug's signature: y must not be reachable from X's own
        # last n_assets columns (that used to be the SAME bar as y).
        assert not torch.equal(X[:, -n_assets:], y)


# ---------------------------------------------------------------------------
# AUDIT-1B: MAML inner-loop adaptation was a complete no-op
# ---------------------------------------------------------------------------

class TestMamlAdaptationIsReal:
    def test_adapted_params_differ_from_pre_adaptation(self):
        """_forward_with_params() used to ignore the passed-in adapted
        parameter clones and call self.model(X) (the live weights)
        directly - so autograd.grad(loss, adapted_params) always returned
        None (zero-filled), and 'adapted' params were bit-identical to
        the pre-adaptation clones after every single inner step. Fixed
        via torch.func.functional_call so the loss's graph actually
        traces back to the passed-in params."""
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 2))
        learner = MAMLMetaLearner(model=model, inner_lr=0.1, n_inner_steps=3)

        X = torch.randn(12, 6)
        y = torch.randn(12, 2)
        pre = {n: p.clone() for n, p in model.named_parameters()}

        adapted, inner_losses = learner.adapt(
            (X, y), torch.nn.functional.mse_loss, return_adapted_model=True,
        )
        post = dict(adapted.named_parameters())

        changed = [n for n in pre if not torch.equal(pre[n], post[n])]
        assert changed == sorted(p[0] for p in model.named_parameters()) or len(changed) == len(pre), (
            f"expected every parameter to change after adaptation, only "
            f"{len(changed)}/{len(pre)} did: {changed}"
        )
        assert len(inner_losses) == 3
        assert inner_losses[-1] < inner_losses[0], (
            "3 real gradient steps on a fixed batch should reduce loss; "
            "a no-op adaptation would leave every step's loss identical"
        )

    def test_clipped_maml_warmup_pre_post_loss_can_differ(self):
        """kronos/warmer.py's ClippedMAML inherits _forward_with_params
        unchanged - the MAM-02 accept/reject gate (post_loss vs pre_loss)
        can only ever mean something if adaptation actually happened.
        Before the fix, pre_loss and post_loss were ALWAYS exactly equal
        (same weights, same forward pass) for any input."""
        from kronos.warmer import ClippedMAML

        torch.manual_seed(1)
        model = nn.Linear(4, 3)
        learner = ClippedMAML(model=model, inner_lr=0.2, n_inner_steps=3)
        X = torch.randn(10, 4)
        y = torch.randn(10, 3)
        loss_fn = torch.nn.functional.mse_loss

        with torch.no_grad():
            pre_loss = float(loss_fn(model(X), y).item())
        adapted, _ = learner.adapt((X, y), loss_fn, return_adapted_model=True)
        with torch.no_grad():
            post_loss = float(loss_fn(adapted(X), y).item())

        assert post_loss != pre_loss, (
            "pre/post loss must be able to differ - if they're always "
            "exactly equal, adaptation is a no-op again"
        )

    def test_meta_train_step_updates_model_parameters(self):
        """The outer loop (meta_train_step) must also move the model's
        real parameters via backprop through the (now-real) adapted
        params - this was already true even with the bug (outer loss
        used live params directly), but verify it still holds post-fix."""
        torch.manual_seed(2)
        model = nn.Linear(5, 2)
        learner = MAMLMetaLearner(model=model, inner_lr=0.05, n_inner_steps=2, outer_lr=0.1)
        pre = {n: p.clone() for n, p in model.named_parameters()}

        tasks = [
            ((torch.randn(8, 5), torch.randn(8, 2)),
             (torch.randn(8, 5), torch.randn(8, 2)))
            for _ in range(3)
        ]
        learner.meta_train_step(tasks, torch.nn.functional.mse_loss)

        post = dict(model.named_parameters())
        assert any(not torch.equal(pre[n], post[n]) for n in pre)


# ---------------------------------------------------------------------------
# AUDIT-2A: SNN input scale vs. fixed spike threshold
# ---------------------------------------------------------------------------

class TestSNNInputScale:
    def test_realistic_return_scale_input_no_longer_starves_membrane_potential(self):
        """Before AUDIT-2A, real per-bar returns (~1e-3) fed straight to
        LIFNeuron (fixed v_thresh=1.0, no input normalization anywhere)
        left membrane voltage ~1000x below threshold. InputScaleNorm now
        rescales to O(1) in train mode using the batch's own live scale -
        verify the normalized input actually lands near unit scale rather
        than staying at the raw 1e-3 magnitude."""
        from prometheus.neuro.spiking_network import InputScaleNorm

        torch.manual_seed(0)
        norm = InputScaleNorm(n_features=5)
        norm.train()
        x = torch.randn(8, 20, 5) * 0.003   # realistic real-return scale
        out = norm(x)
        assert out.abs().mean().item() > 0.1, (
            "normalized input should be within an order of magnitude of "
            "unit scale, not still at the raw ~1e-3 input scale"
        )

    def test_eval_mode_batch_size_one_does_not_crash(self):
        """Live production inference always calls this network with
        batch=1 (kronos/reflex.py). nn.BatchNorm1d would raise in train()
        mode at batch=1; InputScaleNorm must not, in either mode."""
        from prometheus.neuro.spiking_network import SpikingMarketEncoder

        torch.manual_seed(0)
        net = SpikingMarketEncoder(input_size=5, layer_sizes=[16, 8], output_size=5, n_timesteps=8)
        net.eval()
        x = torch.randn(1, 5, 5) * 0.005
        with torch.no_grad():
            out, meta = net(x)
        assert out.shape == (1, 5)
        assert torch.isfinite(out).all()

    def test_gradients_still_flow_through_input_norm(self):
        from prometheus.neuro.spiking_network import SpikingMarketEncoder

        torch.manual_seed(0)
        net = SpikingMarketEncoder(input_size=4, layer_sizes=[8], output_size=4, n_timesteps=8)
        net.train()
        x = torch.randn(3, 8, 4) * 0.003
        out, _ = net(x)
        out.pow(2).mean().backward()
        grads = [p.grad for p in net.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads)


# ---------------------------------------------------------------------------
# AUDIT-2B: live REFLEX ticks never saw intraday price movement
# ---------------------------------------------------------------------------

class TestReflexIntradayUpdates:
    def _memory(self):
        import pandas as pd
        from datetime import datetime, timezone
        from kronos.data_pipeline import DailyMemory

        dates = pd.date_range("2024-01-01", periods=10)
        prices = pd.DataFrame(
            {"A": np.linspace(100, 109, 10), "B": np.linspace(50, 59, 10)},
            index=dates,
        )
        returns = prices.pct_change().fillna(0.0)
        return DailyMemory(
            as_of=datetime.now(timezone.utc), prices=prices, volumes=prices * 0 + 1000,
            returns=returns, vix=pd.Series([20.0] * 10, index=dates),
            sentiment={}, macro={}, source_used="test",
        )

    def test_window_changes_tick_to_tick_with_live_prices(self):
        from kronos.orchestrator import KronosOrchestrator
        from kronos.config import load_config

        cfg = load_config()
        cfg.override("data.tickers", ["A", "B"])
        orch = KronosOrchestrator.__new__(KronosOrchestrator)
        orch.cfg = cfg
        orch._day_open_prices = {}
        orch.state = type("S", (), {"memory": self._memory()})()

        r1 = orch._recent_returns_with_intraday({"A": 109.0, "B": 59.0})
        r2 = orch._recent_returns_with_intraday({"A": 109.0 * 1.02, "B": 59.0})
        r3 = orch._recent_returns_with_intraday({"A": 109.0 * 0.98, "B": 59.0})

        assert not np.array_equal(r1[-1], r2[-1])
        assert not np.array_equal(r2[-1], r3[-1])
        assert r1.shape == r2.shape == r3.shape

    def test_no_bar_prices_falls_back_to_pure_historical_window(self):
        from kronos.orchestrator import KronosOrchestrator
        from kronos.config import load_config

        cfg = load_config()
        cfg.override("data.tickers", ["A", "B"])
        orch = KronosOrchestrator.__new__(KronosOrchestrator)
        orch.cfg = cfg
        orch._day_open_prices = {}
        memory = self._memory()
        orch.state = type("S", (), {"memory": memory})()

        r = orch._recent_returns_with_intraday(None)
        expected = memory.returns_window(cfg.nightmare.horizon_days)
        assert np.allclose(r, expected)


# ---------------------------------------------------------------------------
# AUDIT-2C: black-swan pretraining corpus from an untrained score network
# ---------------------------------------------------------------------------

class TestDiffusionPretrainFallback:
    def test_fetch_failure_falls_back_without_raising(self):
        from prometheus.engine import PrometheusEngine, PrometheusConfig

        torch.manual_seed(0)
        cfg = PrometheusConfig(
            n_assets=2, seq_len=8, horizon=2, d_model=8, n_heads=2,
            n_layers=1, device="cpu",
        )
        engine = PrometheusEngine(cfg)
        with patch("prometheus.engine.MarketDataFetcher") as MockFetcher:
            MockFetcher.return_value.fetch_all.side_effect = RuntimeError("no network")
            engine._pretrain_score_net(None, n_steps=2)   # must not raise

    def test_explicit_real_returns_trains_score_net(self):
        from prometheus.engine import PrometheusEngine, PrometheusConfig

        torch.manual_seed(0)
        cfg = PrometheusConfig(
            n_assets=2, seq_len=8, horizon=2, d_model=8, n_heads=2,
            n_layers=1, device="cpu",
        )
        engine = PrometheusEngine(cfg)
        pre = {k: v.clone() for k, v in engine.diffusion.score_net.state_dict().items()}

        rng = np.random.default_rng(0)
        real_returns = rng.standard_t(df=3, size=(60, 2)).astype(np.float32) * 0.01
        engine._pretrain_score_net(real_returns, n_steps=5)

        post = engine.diffusion.score_net.state_dict()
        assert any(not torch.equal(pre[k], post[k]) for k in pre), (
            "score_net weights must actually change when real data is "
            "available - a silent no-op would leave the network "
            "randomly-initialized exactly as before this fix"
        )

    def test_fat_tail_check_is_no_longer_a_near_tautology(self):
        """passes_fat_tail_check used to be `kurtosis > -1.0 OR
        coverage_pct >= min_tail_pct` - kurtosis > -1.0 is true for
        almost any distribution (including Gaussian, kurtosis=0), so the
        OR made the check pass almost regardless of coverage_pct. Fixed
        to AND with a real fat-tail threshold (kurtosis > 0). Exercises
        the actual method, with generate() patched to return a
        controlled near-Gaussian (thin-tailed) sample so the result is
        deterministic without running real diffusion sampling."""
        from prometheus.generative.diffusion_simulator import MarketDiffusionSimulator

        sim = MarketDiffusionSimulator(n_assets=2, seq_len=4, n_diffusion_steps=2, device="cpu")
        torch.manual_seed(0)
        gaussian_paths = torch.randn(500, 4, 2) * 0.01   # thin-tailed, kurtosis ~ 0

        with patch.object(sim, "generate", return_value=gaussian_paths):
            result = sim.validate_tail_coverage(n_scenarios=500, sigma_threshold=3.0, min_tail_pct=50.0)

        assert result["kurtosis"] < 1.0   # genuinely near-Gaussian
        assert result["coverage_pct"] < 50.0   # genuinely fails the coverage bar
        assert result["passes_fat_tail_check"] is False, (
            "a thin-tailed, low-coverage distribution must fail the check - "
            "the old OR-based logic would have passed it on kurtosis alone"
        )


# ---------------------------------------------------------------------------
# AUDIT-3B: asymmetric loss overestimate/underestimate penalty ratio
# ---------------------------------------------------------------------------

class TestAsymmetricLossBalance:
    def test_overestimate_penalty_ratio_meaningfully_larger_than_before(self):
        """The old calibration floor was a bare 1e-4 constant, disconnected
        from alpha - at alpha=0.5 that's a ~1:5000 ratio to the full
        underestimate penalty. Fixed to scale with alpha
        (OVERESTIMATE_PENALTY_FRAC=0.01), landing at ~1:100 - still
        respects the existing tested <1% invariant
        (test_calibration_gradient_much_smaller_than_underestimate_gradient
        in tests/unit/test_loss_kelly.py) but is no longer ~50x more
        negligible than that bound requires."""
        from prometheus.loss.asymmetric_loss import AsymmetricUtilityLoss

        loss_fn = AsymmetricUtilityLoss()
        pred_over = torch.tensor([[0.10]], requires_grad=True)
        target_over = torch.tensor([[0.05]])
        loss_fn(pred_over, target_over).backward()
        grad_over = pred_over.grad.abs().item()

        pred_under = torch.tensor([[0.05]], requires_grad=True)
        target_under = torch.tensor([[0.10]])
        loss_fn(pred_under, target_under).backward()
        grad_under = pred_under.grad.abs().item()

        ratio = grad_over / grad_under
        assert ratio > 0.005, (
            f"ratio {ratio:.5f} is still ~the old ~1:5000 floor - fix didn't take"
        )
        assert ratio < 0.01, (
            f"ratio {ratio:.5f} exceeds the tested <1% invariant"
        )
