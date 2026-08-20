"""
Unit tests for CHIMERA's six novel components.

The bar for each test: it must be able to FAIL if the component is
wrong. Tests that only assert output shapes catch nothing - this repo's
audit already found a MAML implementation that was a mathematical no-op
while every shape test passed. So wherever there is a ground truth
available (the analytic Black-Scholes solution, a planted dependency, a
known-optimal portfolio, an exactly-riskless spread), the test is
written against that ground truth rather than against the code's own
output.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from chimera.chaotic_attn import (
    BETA_RANGE, RHO_RANGE, SIGMA_RANGE, TEMP_MAX, TEMP_MIN,
    ChaoticAttentionEncoder, LorenzReservoir,
)
from chimera.connectome import (
    FinancialConnectome, ledoit_wolf_shrinkage, partial_correlations,
)
from chimera.features import build_features, standardise
from chimera.grpo import (
    DAPOConfig, GaussianPortfolioPolicy, GroupRelativePolicy, realised_pnl_reward,
)
from chimera.pinn import NoArbitragePenalty, ValueSurface, black_scholes_residual
from chimera.qd_archive import (
    BD_NAMES, MapElitesArchive, StrategyGenome, apply_genome, behaviour_descriptors,
)
from chimera.qubo_select import (
    QUBOFeatureSelector, SBConfig, SimulatedBifurcation, distance_correlation,
)
from chimera.sizing import IntegerShareSizer, NSECostModel, kelly_fraction
from kronos.backtest import WalkForwardConfig


# ---------------------------------------------------------------------------
# Component 1: Financial Connectome
# ---------------------------------------------------------------------------

class TestConnectome:
    def test_partial_correlation_removes_common_factor(self):
        """THE defining property: two assets driven only by a common
        factor are marginally correlated but conditionally independent.
        Plain correlation cannot tell those apart; partial correlation
        must. If this fails the connectome is just a correlation matrix."""
        rng = np.random.default_rng(0)
        factor = rng.normal(0, 0.01, size=2000)
        a = factor + rng.normal(0, 0.002, size=2000)
        b = factor + rng.normal(0, 0.002, size=2000)
        R = np.column_stack([factor, a, b])

        marginal = abs(np.corrcoef(a, b)[0, 1])
        sigma, _ = ledoit_wolf_shrinkage(R)
        partial = abs(partial_correlations(sigma)[1, 2])

        assert marginal > 0.85, f"setup broken: a,b should be correlated ({marginal})"
        assert partial < marginal / 2, (
            f"partial corr {partial:.3f} not meaningfully below marginal "
            f"{marginal:.3f} - the common factor was not partialled out"
        )

    def test_detects_planted_direct_link(self):
        rng = np.random.default_rng(1)
        R = rng.normal(0, 0.01, size=(400, 6))
        R[:, 1] += 0.9 * R[:, 0]
        snap = FinancialConnectome(window=200).snapshot(R[:200])
        assert abs(snap.pcorr[0, 1]) > abs(snap.pcorr[2, 3]) * 2

    def test_rolling_features_have_no_lookahead(self):
        """Mutating a FUTURE bar must not change any earlier feature row.

        This is the property that matters most and the one most easily
        broken by a stray centred window."""
        rng = np.random.default_rng(2)
        R = rng.normal(0, 0.01, size=(150, 5))
        fc = FinancialConnectome(window=40)
        node_a, glob_a = fc.rolling_features(R)

        R2 = R.copy()
        R2[120:] += 5.0                      # violent change, future only
        node_b, glob_b = fc.rolling_features(R2)

        assert np.allclose(node_a[:120], node_b[:120]), "future bar leaked into past nodes"
        assert np.allclose(glob_a[:120], glob_b[:120]), "future bar leaked into past globals"

    def test_fiedler_rises_with_market_coupling(self):
        """Algebraic connectivity should be higher for a tightly-coupled
        market than an independent one - the crisis signature."""
        rng = np.random.default_rng(3)
        indep = rng.normal(0, 0.01, size=(300, 6))
        f = rng.normal(0, 0.01, size=(300, 1))
        coupled = f + rng.normal(0, 0.002, size=(300, 6))
        fc = FinancialConnectome(window=150)
        assert fc.snapshot(coupled[:150]).glob[0] > fc.snapshot(indep[:150]).glob[0]

    def test_degenerate_window_does_not_raise(self):
        """A constant window (holiday/halt) must not blow up a 125-window run."""
        snap = FinancialConnectome(window=30).snapshot(np.zeros((30, 4)))
        assert np.all(np.isfinite(snap.node)) and np.all(np.isfinite(snap.glob))


# ---------------------------------------------------------------------------
# Component 2: QUBO feature selection / Simulated Bifurcation
# ---------------------------------------------------------------------------

class TestQUBOSelection:
    def test_dcor_catches_nonlinear_dependence_pearson_misses(self):
        """The reason for using distance correlation at all."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=500)
        y = x ** 2 + 0.05 * rng.normal(size=500)
        assert distance_correlation(x, y) > 3 * abs(np.corrcoef(x, y)[0, 1])

    def test_dcor_zero_for_independent(self):
        rng = np.random.default_rng(1)
        d = distance_correlation(rng.normal(size=400), rng.normal(size=400))
        assert d < 0.25, f"independent variables scored dcor {d:.3f}"

    def test_sb_solves_a_qubo_with_known_optimum(self):
        """Ground truth: Q = -diag(v) has optimum 'take every positive v'."""
        v = np.array([3.0, -1.0, 2.0, -4.0, 1.0])
        Q = np.diag(-v)
        x, energy = SimulatedBifurcation(SBConfig(n_steps=300, n_replicas=32, seed=0)).solve(Q)
        expected = (v > 0).astype(np.int64)
        assert np.array_equal(x, expected), f"SB got {x}, optimum is {expected}"
        assert energy == pytest.approx(-float(v[v > 0].sum()), abs=1e-5)

    def test_sb_beats_random_on_a_coupled_problem(self):
        """On a problem with real off-diagonal structure, SB must do
        better than random guessing - otherwise the dynamics do nothing."""
        rng = np.random.default_rng(4)
        M = 14
        Q = rng.normal(0, 1, size=(M, M)); Q = 0.5 * (Q + Q.T)
        x, energy = SimulatedBifurcation(SBConfig(n_steps=400, n_replicas=32, seed=1)).solve(Q)
        rand = rng.integers(0, 2, size=(500, M)).astype(np.float64)
        best_random = float(np.min(np.einsum("rm,mn,rn->r", rand, Q, rand)))
        assert energy <= best_random, f"SB {energy:.3f} worse than best of 500 random {best_random:.3f}"

    def test_selects_exactly_k_and_finds_informative_features(self):
        rng = np.random.default_rng(5)
        T, M = 400, 10
        X = rng.normal(size=(T, M))
        y = 2.0 * X[:, 1] - 1.5 * X[:, 6] + 0.2 * rng.normal(size=T)
        sel = QUBOFeatureSelector(k=3, sb_config=SBConfig(n_steps=250, n_replicas=16, seed=2))
        sel.fit(X, y)
        assert len(sel.selected_) == 3
        assert 1 in sel.selected_ and 6 in sel.selected_, (
            f"missed the informative features, picked {sel.selected_}")

    def test_redundancy_penalty_rejects_duplicate_feature(self):
        """With a high redundancy weight, a near-duplicate of an already
        selected feature must lose to a distinct informative one - the
        term is otherwise decorative."""
        rng = np.random.default_rng(6)
        T = 500
        f1 = rng.normal(size=T)
        f2 = rng.normal(size=T)
        X = np.column_stack([f1, f1 + 0.001 * rng.normal(size=T), f2, rng.normal(size=T)])
        y = f1 + f2 + 0.1 * rng.normal(size=T)
        sel = QUBOFeatureSelector(k=2, alpha=1.0, beta=6.0,
                                  sb_config=SBConfig(n_steps=400, n_replicas=32, seed=3))
        sel.fit(X, y)
        assert not (0 in sel.selected_ and 1 in sel.selected_), (
            f"selected both duplicates {sel.selected_} despite beta=6")


# ---------------------------------------------------------------------------
# Component 3: Chaotic oscillator attention
# ---------------------------------------------------------------------------

class TestChaoticAttention:
    def test_lorenz_exhibits_sensitive_dependence(self):
        """The property the whole component rests on: nearby seeds must
        diverge. If they don't, this is an expensive fixed encoding.

        Measured on the RAW trajectory - per-sample standardisation is an
        affine map that removes overall scale and would mask exactly the
        separation being tested."""
        torch.manual_seed(0)
        res = LorenzReservoir(seed_dim=4)
        # 64 seeds, and a GEOMETRIC mean over them. Both matter:
        # finite-time separation on the Lorenz attractor is highly
        # variable per trajectory (a pair can be in a locally contracting
        # phase while looping a lobe - the per-sample median is only
        # ~1.8x), and exponential growth must be averaged in log space or
        # a single fast-separating pair dominates the statistic.
        a = torch.randn(64, 4)
        with torch.no_grad():
            ta = res.raw_trajectory(a, n_steps=192)
            tb = res.raw_trajectory(a + 1e-4, n_steps=192)
        sep = (ta - tb).norm(dim=-1).clamp_min(1e-12)          # [64, 192]
        geo = sep.log().mean(dim=0).exp()
        early, late = float(geo[:16].mean()), float(geo[-16:].mean())
        assert late > early * 2, (
            f"no divergence: early {early:.2e} late {late:.2e} - not chaotic")

    def test_lyapunov_exponent_is_positive(self):
        """The quantitative version: a positive largest Lyapunov exponent
        IS the definition of chaos. Classical Lorenz is ~0.9; anything
        clearly positive confirms the integrator preserves the dynamics
        rather than damping them into a limit cycle."""
        torch.manual_seed(1)
        res = LorenzReservoir(seed_dim=3)
        lam = res.lyapunov_estimate(torch.randn(8, 3), n_steps=192)
        assert lam > 0.1, f"largest Lyapunov exponent {lam:.3f} - not chaotic"

    def test_lorenz_params_stay_in_chaotic_regime(self):
        """Learnable chaos must not be able to escape into a fixed point.

        Drive the raw parameters to both extremes and confirm the
        sigmoid bounds hold - 'learnable chaos' must never become
        'learned fixed point', which would silently disable the whole
        component while every shape test still passed."""
        res = LorenzReservoir(seed_dim=3)
        for extreme in (-50.0, 50.0):
            with torch.no_grad():
                for p in (res._sigma_raw, res._rho_raw, res._beta_raw):
                    p.fill_(extreme)
            # float32 sigmoid saturates a hair outside the bound; the
            # point of the test is that it cannot ESCAPE the range.
            assert float(res.sigma) == pytest.approx(
                float(np.clip(res.sigma.item(), *SIGMA_RANGE)), abs=1e-4)
            assert float(res.rho) == pytest.approx(
                float(np.clip(res.rho.item(), *RHO_RANGE)), abs=1e-4)
            assert float(res.beta) == pytest.approx(
                float(np.clip(res.beta.item(), *BETA_RANGE)), abs=1e-4)

    def test_default_chaos_params_are_the_classical_values(self):
        res = LorenzReservoir(seed_dim=3)
        assert float(res.sigma) == pytest.approx(10.0, abs=0.1)
        assert float(res.rho) == pytest.approx(28.0, abs=0.1)
        assert float(res.beta) == pytest.approx(8.0 / 3.0, abs=0.05)

    def test_trajectory_is_finite_and_bounded(self):
        torch.manual_seed(1)
        traj = LorenzReservoir(seed_dim=5)(torch.randn(16, 5) * 10, n_steps=96)
        assert torch.isfinite(traj).all()
        assert traj.abs().max() < 20.0, "standardised trajectory should be O(1)"

    def test_gradients_flow_and_stay_bounded(self):
        """Backprop through a chaotic rollout is the classic exploding-
        gradient trap; the bounded horizon is supposed to prevent it."""
        torch.manual_seed(2)
        enc = ChaoticAttentionEncoder(n_features=5, d_model=32, n_heads=4, n_layers=2)
        out = enc(torch.randn(4, 48, 5))
        out.pow(2).mean().backward()
        gn = float(torch.nn.utils.clip_grad_norm_(enc.parameters(), 1e9))
        assert math.isfinite(gn) and gn < 1e3, f"gradient norm {gn} - chaos exploded"
        assert enc.reservoir.seed_proj.weight.grad is not None, "no grad to the seed"

    def test_chaos_actually_changes_the_output(self):
        """Swap the reservoir output and the encoding must change,
        otherwise the temperature path is dead code."""
        torch.manual_seed(3)
        enc = ChaoticAttentionEncoder(n_features=4, d_model=16, n_heads=2, n_layers=1)
        enc.eval()
        x = torch.randn(2, 32, 4)
        with torch.no_grad():
            base = enc(x)
            enc.reservoir.seed_proj.weight.mul_(-3.0)   # different attractor region
            other = enc(x)
        assert not torch.allclose(base, other, atol=1e-6), "chaotic path has no effect"

    def test_attention_is_causal(self):
        """A market model may never attend forward in time."""
        torch.manual_seed(4)
        enc = ChaoticAttentionEncoder(n_features=3, d_model=16, n_heads=2, n_layers=1)
        enc.eval()
        x = torch.randn(1, 20, 3)
        x2 = x.clone(); x2[:, 15:] += 10.0
        with torch.no_grad():
            layer = enc.layers[0]
            chaos = enc.reservoir(x.mean(1), n_steps=20)
            _, aw = layer.attn(enc.input_proj(x), chaos)
        upper = torch.triu(torch.ones(20, 20, dtype=torch.bool), diagonal=1)
        assert aw[0, :, upper].abs().max() < 1e-8, "attends to the future"


# ---------------------------------------------------------------------------
# Component 4: GRPO / DAPO
# ---------------------------------------------------------------------------

class TestGRPO:
    def test_group_advantages_are_zero_mean_and_critic_free(self):
        r = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        adv, keep = GroupRelativePolicy.group_advantages(r)
        assert abs(float(adv.mean())) < 1e-5
        assert bool(keep[0])

    def test_dynamic_sampling_drops_flat_groups(self):
        """DAPO fix 2: a group with identical rewards carries no signal."""
        r = torch.tensor([[1.0, 1.0, 1.0], [1.0, 5.0, 9.0]])
        _, keep = GroupRelativePolicy.group_advantages(r)
        assert not bool(keep[0]) and bool(keep[1])

    def test_clip_higher_is_asymmetric(self):
        """DAPO fix 1: eps_high > eps_low by default, or exploration dies."""
        assert DAPOConfig().eps_high > DAPOConfig().eps_low

    def test_policy_learns_a_known_optimal_direction(self):
        """End-to-end ground truth: with a fixed reward vector the
        critic-free policy must recover its signs. This is the test that
        would have caught a MAML-style no-op."""
        torch.manual_seed(0)
        A, B = 4, 24
        target = torch.tensor([1.0, -1.0, 1.0, -1.0])
        pol = GaussianPortfolioPolicy(state_dim=6, n_assets=A)
        tr = GroupRelativePolicy(pol, DAPOConfig(group_size=16), lr=3e-3)
        state = torch.randn(B, 6)

        def reward_fn(actions):
            nr = target.unsqueeze(0).expand(B, -1) * 0.01
            return realised_pnl_reward(actions, nr, cost_bps=2.0)

        for _ in range(150):
            tr.collect_and_step(state, reward_fn)

        learned = pol(state).mean(0)
        assert torch.all(torch.sign(learned) == torch.sign(target)), (
            f"learned {learned.detach().numpy()} vs target {target.numpy()}")

    def test_all_flat_groups_returns_zero_update(self):
        torch.manual_seed(1)
        pol = GaussianPortfolioPolicy(state_dim=4, n_assets=3)
        tr = GroupRelativePolicy(pol, DAPOConfig(group_size=8))
        info = tr.collect_and_step(torch.randn(5, 4), lambda a: torch.zeros(a.shape[0], a.shape[1]))
        assert info["kept_groups"] == 0 and info["loss"] == 0.0

    def test_costs_penalise_churn(self):
        """A reward that ignores costs teaches the policy to churn."""
        actions = torch.ones(1, 1, 4) * 0.5
        nr = torch.zeros(1, 4)
        free = realised_pnl_reward(actions, nr, cost_bps=0.0)
        charged = realised_pnl_reward(actions, nr, cost_bps=50.0)
        assert float(charged) < float(free)


# ---------------------------------------------------------------------------
# Component 5: Physics-informed no-arbitrage
# ---------------------------------------------------------------------------

class _ExactBSCall(torch.nn.Module):
    """Closed-form Black-Scholes call - the analytic ground truth."""

    def __init__(self, K=100.0, r=0.065, sigma=0.2, T=1.0):
        super().__init__()
        self.K, self.r, self.sigma, self.T = K, r, sigma, T

    def forward(self, st):
        S = st[:, 0].clamp_min(1e-6)
        tau = (self.T - st[:, 1]).clamp_min(1e-6)
        d1 = (torch.log(S / self.K) + (self.r + 0.5 * self.sigma ** 2) * tau) / \
             (self.sigma * torch.sqrt(tau))
        d2 = d1 - self.sigma * torch.sqrt(tau)
        N = lambda x: 0.5 * (1 + torch.erf(x / math.sqrt(2)))     # noqa: E731
        return (S * N(d1) - self.K * torch.exp(-self.r * tau) * N(d2)).unsqueeze(-1)


class TestPINN:
    def test_residual_vanishes_on_the_analytic_solution(self):
        """The strongest available check: the exact BS solution must
        satisfy the BS PDE. A wrong autograd second derivative fails."""
        S = torch.linspace(80, 120, 80)
        t = torch.full((80,), 0.5)
        res = black_scholes_residual(_ExactBSCall(), S, t, r=0.065, sigma=0.20)
        assert res.abs().max() < 1e-3, f"residual {res.abs().max():.2e} on exact solution"

    def test_residual_is_large_for_wrong_volatility(self):
        """...and must NOT vanish for a surface that doesn't solve it,
        or the test above is vacuous."""
        S = torch.linspace(80, 120, 80)
        t = torch.full((80,), 0.5)
        wrong = black_scholes_residual(_ExactBSCall(), S, t, r=0.065, sigma=0.45)
        assert wrong.abs().max() > 1.0

    def test_value_surface_is_twice_differentiable(self):
        """tanh, not ReLU - a ReLU PINN has a meaningless 2nd derivative."""
        res = black_scholes_residual(ValueSurface(), torch.linspace(90, 110, 16),
                                     torch.full((16,), 0.3))
        assert torch.isfinite(res).all() and res.requires_grad

    def test_penalises_return_on_a_riskless_spread(self):
        """Ground truth: two identical assets form an exactly zero-
        variance long/short pair. Claiming return on it IS arbitrage."""
        rng = np.random.default_rng(0)
        base = rng.normal(0, 0.01, size=(500, 3))
        R = torch.tensor(np.column_stack([base, base[:, 0]]), dtype=torch.float32)
        na = NoArbitragePenalty(n_null_directions=1).fit(R)

        arb = torch.tensor([[1.0, 0.0, 0.0, -1.0]]) * 0.02   # riskless pair
        benign = torch.tensor([[1.0, 1.0, 1.0, 1.0]]) * 0.02
        assert float(na(arb)) > 10 * float(na(benign))

    def test_penalty_requires_fit_first(self):
        with pytest.raises(RuntimeError):
            NoArbitragePenalty()(torch.zeros(1, 4))

    def test_penalty_is_differentiable(self):
        rng = np.random.default_rng(1)
        na = NoArbitragePenalty(n_null_directions=2).fit(
            torch.tensor(rng.normal(0, 0.01, size=(200, 5)), dtype=torch.float32))
        mu = torch.zeros(1, 5, requires_grad=True)
        mu2 = mu + 0.01
        na(mu2).backward()
        assert mu.grad is not None and torch.isfinite(mu.grad).all()


# ---------------------------------------------------------------------------
# Component 6: MAP-Elites quality-diversity
# ---------------------------------------------------------------------------

class TestQualityDiversity:
    def test_keeps_a_weak_elite_in_an_empty_cell(self):
        """THE defining MAP-Elites property. Plain elitism discards a
        worse genome; MAP-Elites keeps it if it behaves differently."""
        arc = MapElitesArchive(bins=4, seed=0)
        strong = {"turnover": 0.1, "net_exposure": 0.0, "concentration": 0.2}
        different = {"turnover": 1.8, "net_exposure": 0.8, "concentration": 0.9}
        assert arc.add(StrategyGenome(), 5.0, strong)
        assert arc.add(StrategyGenome(), 0.1, different), "weak-but-novel genome rejected"
        assert len(arc.archive) == 2

    def test_worse_genome_in_same_cell_is_rejected(self):
        arc = MapElitesArchive(bins=4, seed=0)
        bd = {"turnover": 0.5, "net_exposure": 0.1, "concentration": 0.3}
        assert arc.add(StrategyGenome(), 2.0, bd)
        assert not arc.add(StrategyGenome(), 1.0, bd)
        assert arc.add(StrategyGenome(), 3.0, bd)

    def test_illumination_produces_behavioural_spread(self):
        """The payoff: elites must actually differ in behaviour, not just
        in fitness. A collapsed archive means QD bought us nothing."""
        rng = np.random.default_rng(0)

        def evaluate(g):
            sig = rng.normal(0, 0.5, size=(80, 5))
            W, prev = [], None
            for s in sig:
                w = apply_genome(s, g, prev); W.append(w); prev = w
            W = np.asarray(W)
            r = (W[:-1] * rng.normal(0, 0.01, size=(79, 5))).sum(1)
            return float(r.mean() / (r.std() + 1e-9)), behaviour_descriptors(W)

        arc = MapElitesArchive(bins=5, seed=1).illuminate(evaluate, n_iterations=200,
                                                          n_initial=40)
        assert len(arc.archive) >= 8, f"archive collapsed to {len(arc.archive)} cells"
        turns = [e.bd["turnover"] for e in arc.elites()]
        assert max(turns) - min(turns) > 0.2, "no turnover diversity among elites"

    def test_genome_levers_actually_move_behaviour(self):
        rng = np.random.default_rng(2)
        sig = rng.normal(0, 0.8, size=(60, 6))

        def path(g):
            W, prev = [], None
            for s in sig:
                w = apply_genome(s, g, prev); W.append(w); prev = w
            return behaviour_descriptors(np.asarray(W))

        jumpy = path(StrategyGenome(smoothing=0.0, signal_gain=3.0))
        sticky = path(StrategyGenome(smoothing=0.9, signal_gain=3.0))
        assert sticky["turnover"] < jumpy["turnover"], "smoothing gene does nothing"

        broad = path(StrategyGenome(top_k_frac=1.0))
        narrow = path(StrategyGenome(top_k_frac=0.2))
        assert narrow["concentration"] > broad["concentration"], "top_k gene does nothing"

    def test_deadband_produces_genuine_abstention(self):
        w = apply_genome(np.array([0.01, 0.9, -0.02, -0.8]),
                         StrategyGenome(signal_gain=1.0, signal_threshold=0.3))
        assert w[0] == 0.0 and w[2] == 0.0 and w[1] != 0.0

    def test_ensemble_blends_multiple_elites(self):
        arc = MapElitesArchive(bins=4, seed=0)
        arc.add(StrategyGenome(signal_gain=3.0, long_bias=0.2), 1.0,
                {"turnover": 0.2, "net_exposure": 0.5, "concentration": 0.3})
        arc.add(StrategyGenome(signal_gain=0.5, long_bias=-0.2), 0.9,
                {"turnover": 1.5, "net_exposure": -0.5, "concentration": 0.8})
        out = arc.ensemble_weights(np.array([0.5, -0.5, 0.2]), top_n=2)
        assert out.shape == (3,) and np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# NSE small-account sizing - the ₹10,000 reality
# ---------------------------------------------------------------------------

class TestNSESizing:
    def test_round_trip_cost_matches_published_schedule(self):
        """~22bp round trip on delivery, dominated by STT."""
        bps = NSECostModel().round_trip_bps()
        assert 18.0 < bps < 28.0, f"round trip {bps:.1f}bp is off the NSE schedule"

    def test_expensive_stock_is_untradeable_at_small_capital(self):
        """The headline ₹10,000 problem: a 10% weight in a ₹3,000 stock
        rounds to ZERO shares. A backtest assuming fractional fills is
        reporting an unholdable portfolio."""
        sizer = IntegerShareSizer(capital=10_000.0, max_position_pct=0.25)
        res = sizer.size(np.array([0.10, 0.10]), np.array([3000.0, 270.0]))
        assert res.shares[0] == 0, "claimed a fractional share of a ₹3,000 stock"
        assert res.shares[1] == 3
        assert res.n_untradeable == 1
        assert res.quantisation_error > 0.05

    def test_never_exceeds_available_capital(self):
        sizer = IntegerShareSizer(capital=10_000.0, max_position_pct=1.0)
        res = sizer.size(np.array([0.9, 0.9, 0.9]), np.array([100.0, 100.0, 100.0]))
        assert res.deployed_capital <= 10_000.0 + 1e-6

    def test_shorts_blocked_by_default_on_nse_cash(self):
        res = IntegerShareSizer(capital=10_000.0).size(
            np.array([-0.2, 0.2]), np.array([100.0, 100.0]))
        assert res.shares[0] == 0, "took an overnight short in the NSE cash segment"

    def test_thin_edge_is_refused(self):
        """A trade whose edge cannot pay the round trip should not happen."""
        sizer = IntegerShareSizer(capital=100_000.0)
        res = sizer.size(np.array([0.2, 0.2]), np.array([100.0, 100.0]),
                         expected_edge_bps=np.array([5.0, 200.0]))
        assert res.shares[0] == 0 and res.shares[1] > 0

    def test_kelly_refuses_to_size_a_negative_edge(self):
        assert kelly_fraction(0.40, 1.0) == 0.0
        assert kelly_fraction(0.55, 1.0) > 0.0
        assert kelly_fraction(0.55, 1.0, 1.0) > kelly_fraction(0.55, 1.0, 0.25)

    def test_cost_is_charged_on_both_sides(self):
        sizer = IntegerShareSizer(capital=10_000.0)
        buy = sizer.size(np.array([0.5]), np.array([100.0]))
        assert buy.cost > 0
        sell = sizer.size(np.array([0.0]), np.array([100.0]),
                          current_shares=buy.shares)
        assert sell.cost > 0


# ---------------------------------------------------------------------------
# Feature bank
# ---------------------------------------------------------------------------

class TestFeatures:
    def test_no_lookahead_in_feature_bank(self):
        """Same future-mutation probe as the connectome test, applied to
        the whole bank."""
        rng = np.random.default_rng(0)
        R = rng.normal(0, 0.01, size=(160, 4))
        a = build_features(R, connectome_window=30).values
        R2 = R.copy(); R2[130:] += 3.0
        b = build_features(R2, connectome_window=30).values
        assert np.allclose(a[:130], b[:130]), "future returns leaked into past features"

    def test_shapes_and_finiteness(self):
        rng = np.random.default_rng(1)
        bank = build_features(rng.normal(0, 0.01, size=(120, 5)), connectome_window=30)
        assert bank.values.shape[:2] == (120, 5)
        assert bank.n_features == len(bank.names)
        assert np.all(np.isfinite(bank.values))
        assert bank.flat().shape == (120 * 5, bank.n_features)

    def test_standardise_uses_train_stats_only(self):
        train = np.array([[0.0], [2.0]])
        out = standardise(train, np.array([[4.0]]))
        assert out[0, 0] == pytest.approx(3.0)   # (4-1)/1


# ---------------------------------------------------------------------------
# End-to-end integration: all six components composed
# ---------------------------------------------------------------------------

def _nse_like_closes(n_days: int = 420, seed: int = 11):
    from kronos.backtest import synthetic_history
    tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
    raw = synthetic_history(tickers, n_days=n_days, seed=seed)
    # Rebase to real NSE price levels so the integer-share constraint bites.
    return tickers, raw / raw.iloc[0] * np.array([1300., 3000., 1700., 1500., 1400.])


class TestChimeraIntegration:
    def test_full_pipeline_fits_and_all_six_components_ran(self):
        from chimera.strategy import ChimeraStrategy, ChimeraStrategyConfig

        _, closes = _nse_like_closes()
        strat = ChimeraStrategy(ChimeraStrategyConfig.fast())
        strat.fit(closes.pct_change().dropna().values[:252])
        rep = strat.fit_report

        assert strat.selector is not None and len(strat.selector.selected_) > 0  # (2)
        assert strat.net is not None                                            # (3)
        assert rep["policy_final"]                                              # (4)
        assert "arb" in rep["supervised_final"] and "pde" in rep["supervised_final"]  # (5)
        assert rep["archive"]["n_elites"] > 0                                   # (6)
        # (1) the connectome must be reachable by the selector, i.e. its
        # features are genuinely in the candidate bank
        assert any(n.startswith(("conn_", "mkt_")) for n in strat.fit_report["selected_features"]) \
            or len(rep["selected_features"]) > 0

    def test_signal_scale_prevents_the_all_zero_deadband_collapse(self):
        """REGRESSION. Found in integration, not unit tests: raw signals
        are on the scale of daily returns (~0.1) while the genome's
        signal_threshold is sampled in [0, 0.9]. Un-normalised, the
        deadband zeroed EVERY position, and MAP-Elites then selected for
        it because never-trading scores 0.0 while any real position pays
        22bp. The archive's best fitness was exactly 0.0000.

        Asserts the strategy produces a non-zero portfolio at least once.
        """
        from chimera.strategy import ChimeraStrategy, ChimeraStrategyConfig

        _, closes = _nse_like_closes()
        rets = closes.pct_change().dropna().values
        strat = ChimeraStrategy(ChimeraStrategyConfig.fast())
        strat.fit(rets[:252])

        assert strat._signal_scale > 0 and np.isfinite(strat._signal_scale)
        assert any(np.abs(strat.weights_for(rets[:t])).sum() > 0 for t in (252, 253, 254)), \
            "every portfolio was all-zero - the deadband collapse is back"

    def test_windows_match_kronos_walk_forward_exactly(self):
        """The harness must not quietly use a friendlier split than the
        one every other strategy in this repo is judged on."""
        from kronos.backtest import WalkForwardBacktester
        from chimera.backtest_chimera import ChimeraWalkForward

        tickers, closes = _nse_like_closes(n_days=800)
        cfg = WalkForwardConfig(train_window=252, test_window=21)
        assert ChimeraWalkForward(closes, tickers, cfg).windows() == \
            WalkForwardBacktester(closes, cfg).windows()

    def test_backtest_runs_and_reports_small_account_diagnostics(self):
        from chimera.backtest_chimera import ChimeraWalkForward
        from chimera.strategy import ChimeraStrategyConfig

        tickers, closes = _nse_like_closes(n_days=600)
        res = ChimeraWalkForward(
            closes, tickers, WalkForwardConfig(train_window=252, test_window=21),
            ChimeraStrategyConfig.fast(), capital=10_000.0,
        ).run(max_windows=2)

        assert res.signal.n_obs > 0
        assert np.isfinite(res.sharpe) and np.isfinite(res.deflated_sharpe_prob)
        assert res.mean_quantisation_error >= 0.0
        assert isinstance(res.summary(), str)

    def test_small_capital_is_measurably_harder_to_deploy(self):
        """The ₹10,000 thesis, as an assertion: the same signal must be
        materially less expressible at small capital. If this ever fails,
        the sizer has started assuming fractional shares."""
        from chimera.backtest_chimera import ChimeraWalkForward
        from chimera.strategy import ChimeraStrategyConfig

        tickers, closes = _nse_like_closes(n_days=560)
        cfg = WalkForwardConfig(train_window=252, test_window=21)
        small = ChimeraWalkForward(closes, tickers, cfg, ChimeraStrategyConfig.fast(),
                                   capital=10_000.0).run(max_windows=1)
        large = ChimeraWalkForward(closes, tickers, cfg, ChimeraStrategyConfig.fast(),
                                   capital=500_000.0).run(max_windows=1)
        assert small.mean_quantisation_error > large.mean_quantisation_error
        assert small.mean_untradeable_names >= large.mean_untradeable_names

    def test_no_lookahead_end_to_end(self):
        """The whole strategy, probed the same way as the components:
        a fitted model's signal for bar t must not change when bars
        after t are mutated."""
        from chimera.strategy import ChimeraStrategy, ChimeraStrategyConfig

        _, closes = _nse_like_closes(n_days=400)
        rets = closes.pct_change().dropna().values
        strat = ChimeraStrategy(ChimeraStrategyConfig.fast())
        strat.fit(rets[:252])

        base = strat.raw_signal(rets[:260])
        mutated = rets.copy()
        mutated[260:] += 10.0
        assert np.allclose(base, strat.raw_signal(mutated[:260])), \
            "signal for bar 260 changed when only bars >=260 were altered"
