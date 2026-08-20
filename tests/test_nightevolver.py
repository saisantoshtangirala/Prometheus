"""
Tests for NightEvolver.

Same bar as the rest of this repo: every test must be able to FAIL if
its component is wrong. The headline test is
`test_ga_overfits_pure_noise_and_the_controls_catch_it` - it asserts the
central claim of the whole design, on data that is unpredictable by
construction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from kronos.backtest import WalkForwardConfig
from nightevolver.data_loader import (
    WARMUP_BARS, build_market_data, compute_indicators,
)
from nightevolver.ga_engine import (
    GAConfig, GeneticEvolver, expected_max_sharpe_from_noise, fitness, simulate,
)
from nightevolver.genome import (
    GENOME_LENGTH, INDICATOR_NAMES, N_INDICATORS, crossover, decode, mutate,
    random_genome, score_matrix,
)
from nightevolver.rl_trainer import N_ACTIONS, QConfig, build_states, train_q_learning
from nightevolver.saver import (
    MIN_DEFLATED_SHARPE_PROB, load_checkpoint, save_checkpoint,
)
from nightevolver.strategy_decoder import EvolvedStrategy


def _random_walk(n_days: int = 500, n_assets: int = 6, seed: int = 0,
                 drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.012, size=(n_days, n_assets))
    px = 100 * np.cumprod(1 + rets, axis=0)
    return pd.DataFrame(px, index=pd.bdate_range("2021-01-01", periods=n_days),
                        columns=[f"T{i}.NS" for i in range(n_assets)])


# ---------------------------------------------------------------------------
# Genome
# ---------------------------------------------------------------------------

class TestGenome:
    def test_decode_is_deterministic_and_in_range(self):
        rng = np.random.default_rng(0)
        g = random_genome(rng)
        a, b = decode(g), decode(g)
        assert np.array_equal(a.indicator_weights, b.indicator_weights)
        assert a.indicator_weights.sum() == pytest.approx(1.0)
        assert 5 <= a.hold_days <= 90
        assert 0.02 <= a.trailing_stop <= 0.20
        assert 0.0 <= a.kelly_fraction <= 1.0
        assert 0.0 <= a.vol_target <= 0.30

    def test_hold_floor_excludes_the_horizons_costs_close_off(self):
        """A 1-day hold needs rho=0.342 against forward returns just to
        clear 22bp; the best measured directional feature was rho=0.035.
        Those horizons are arithmetically closed and must not be in the
        search space, where the GA would only find noise in them."""
        rng = np.random.default_rng(3)
        for _ in range(200):
            assert decode(random_genome(rng)).hold_days >= 5

    def test_genome_length_matches_indicator_count(self):
        """Guards the layout: 3 per-indicator blocks + 6 scalars
        (hold, trailing stop, kelly, regime, conviction, vol_target)."""
        assert GENOME_LENGTH == 3 * N_INDICATORS + 6
        assert len(INDICATOR_NAMES) == N_INDICATORS

    def test_wrong_length_genome_is_rejected(self):
        with pytest.raises(ValueError):
            decode(np.zeros(GENOME_LENGTH - 1))

    def test_all_zero_weights_falls_back_to_equal_weighting(self):
        """A degenerate genome must not divide by zero mid-GA."""
        g = np.zeros(GENOME_LENGTH)
        w = decode(g).indicator_weights
        assert w.sum() == pytest.approx(1.0)
        assert np.allclose(w, w[0])

    def test_operators_stay_in_the_unit_hypercube(self):
        rng = np.random.default_rng(1)
        a, b = random_genome(rng), random_genome(rng)
        c1, c2 = crossover(a, b, rng)
        m = mutate(a, rng, rate=1.0, sigma=5.0)     # extreme, must still clip
        for v in (c1, c2, m):
            assert v.shape == (GENOME_LENGTH,)
            assert v.min() >= 0.0 and v.max() <= 1.0

    def test_crossover_preserves_the_gene_pool(self):
        """Every child gene must come from one of its parents."""
        rng = np.random.default_rng(2)
        a, b = random_genome(rng), random_genome(rng)
        c1, c2 = crossover(a, b, rng)
        assert np.all((c1 == a) | (c1 == b))
        assert np.all(np.isclose(c1 + c2, a + b))   # uniform swap conserves the pair

    def test_score_respects_thresholds(self):
        """A genome with maximal entry thresholds must never vote long."""
        g = np.zeros(GENOME_LENGTH)
        g[:N_INDICATORS] = 1.0                       # equal weights
        g[N_INDICATORS:2 * N_INDICATORS] = 1.0       # entry threshold = 1.0
        g[2 * N_INDICATORS:3 * N_INDICATORS] = 1.0   # exit threshold = 1.0
        strat = decode(g)
        X = np.full((5, 3, N_INDICATORS), 0.9)       # strong but below 1.0
        assert np.allclose(score_matrix(X, strat), 0.0)


# ---------------------------------------------------------------------------
# Indicators / data loader
# ---------------------------------------------------------------------------

class TestIndicators:
    def test_channel_count_matches_genome(self):
        """compute_indicators emits the TECHNICAL block only; the flow
        channels are appended by build_market_data, so the genome's full
        channel count is technical + flow."""
        from nightevolver.genome import N_FLOW, N_TECHNICAL
        close = _random_walk(200, 4)
        vol = pd.DataFrame(1e6, index=close.index, columns=close.columns)
        ind = compute_indicators(close, close, close, vol)
        assert ind.shape == (200, 4, N_TECHNICAL)
        assert np.all(np.isfinite(ind))
        assert N_TECHNICAL + N_FLOW == N_INDICATORS

    def test_build_market_data_emits_the_full_genome_channel_set(self):
        close = _random_walk(200, 4)
        md = build_market_data(close)
        assert md.indicators.shape[2] == N_INDICATORS
        # Flow channels are inert (zero) when no flow data is supplied,
        # rather than absent - the layout must not change with the data.
        from nightevolver.genome import N_TECHNICAL
        assert np.allclose(md.indicators[:, :, N_TECHNICAL:], 0.0)

    def test_indicators_are_bounded(self):
        """tanh-squashed, so thresholds in [0,1] are on a comparable scale."""
        close = _random_walk(300, 5, seed=3)
        ind = compute_indicators(close, close * 1.01, close * 0.99,
                                 pd.DataFrame(1e6, index=close.index, columns=close.columns))
        assert ind.min() >= -1.0 and ind.max() <= 1.0

    def test_no_lookahead_in_indicators(self):
        """Mutating FUTURE bars must not change any earlier indicator row.

        The expanding z-score is the risk here: a full-sample z-score
        would leak, and this is what catches it."""
        close = _random_walk(300, 4, seed=4)
        vol = pd.DataFrame(1e6, index=close.index, columns=close.columns)
        a = compute_indicators(close, close, close, vol)

        c2 = close.copy()
        c2.iloc[250:] *= 3.0
        b = compute_indicators(c2, c2, c2, vol)
        assert np.allclose(a[:250], b[:250], atol=1e-10), "future bars leaked backwards"

    def test_inf_price_does_not_poison_the_target(self):
        """FOUND BY AUDIT. A single Inf close produced a forward return of
        1.8e308 - float max - because np.nan_to_num maps +inf to FLOAT
        MAX, not to 0. The indicators looked clean (tanh-squashed), so
        only the TARGET was poisoned, and one 1.8e308 return dominates
        any mean, Sharpe or fitness it touches."""
        close = _random_walk(300, 5, seed=41)
        clean = build_market_data(close)
        d = close.copy()
        d.iloc[100:105, 1] = np.inf
        md = build_market_data(d)
        assert np.isfinite(md.forward_returns).all()
        assert np.abs(md.forward_returns).max() < 1.0, \
            f"target poisoned: max|fwd| = {np.abs(md.forward_returns).max():.3g}"
        assert np.abs(md.forward_returns).max() == pytest.approx(
            np.abs(clean.forward_returns).max(), rel=0.5)

    def test_zero_and_negative_prices_are_rejected(self):
        """A zero close divides by zero into the same 1.8e308; a negative
        close is not a price at all and was silently accepted."""
        close = _random_walk(300, 5, seed=42)
        for bad in (0.0, -5.0):
            d = close.copy()
            d.iloc[200, 3] = bad
            md = build_market_data(d)
            assert np.isfinite(md.forward_returns).all()
            assert np.abs(md.forward_returns).max() < 1.0, \
                f"price {bad} poisoned the target"

    def test_all_nan_column_raises_instead_of_emptying_the_panel(self):
        """dropna(how='any') silently deleted every row and returned a
        MarketData with 0 bars, so downstream code computed statistics on
        empty arrays instead of failing."""
        close = _random_walk(300, 5, seed=43)
        close["T2"] = np.nan
        with pytest.raises(ValueError, match="no usable price rows"):
            build_market_data(close)

    def test_forward_returns_are_correctly_aligned(self):
        """forward_returns[t] must be the t -> t+1 move, never t-1 -> t."""
        close = _random_walk(200, 3, seed=5)
        md = build_market_data(close)
        raw = close.to_numpy()
        expected = raw[WARMUP_BARS + 1] / raw[WARMUP_BARS] - 1.0
        assert np.allclose(md.forward_returns[0], expected)

    def test_warmup_rows_are_dropped(self):
        close = _random_walk(200, 3, seed=6)
        md = build_market_data(close)
        assert md.n_bars == 200 - WARMUP_BARS - 1


# ---------------------------------------------------------------------------
# GA engine - the headline tests
# ---------------------------------------------------------------------------

class TestGAEngine:
    def test_noise_benchmark_grows_with_search_budget(self):
        """sqrt(2 ln N): more searching finds a better maximum in noise."""
        b10 = expected_max_sharpe_from_noise(10, 250)
        b1000 = expected_max_sharpe_from_noise(1000, 250)
        assert 0 < b10 < b1000
        assert b1000 > 2.5, "best-of-1000 on 250 bars should exceed Sharpe 2.5 in pure noise"

    def test_search_budget_counts_every_evaluation(self):
        cfg = GAConfig(population_size=50, n_generations=20)
        assert cfg.search_budget == 50 * 21

    def test_ga_overfits_pure_noise_and_the_controls_catch_it(self):
        """THE test. On a random walk - unpredictable by construction -
        the GA WILL report a strong in-sample Sharpe. That is not a bug;
        it is what searching 300+ strategies on one window does. What
        matters is that the controls flag it:

          - deflated P(SR>0) must fail the 0.95 gate
          - the winner must not beat the best-of-N noise benchmark

        If this ever passes the gate on pure noise, the statistics are
        broken and every downstream result is worthless."""
        close = _random_walk(600, 8, seed=7)
        md = build_market_data(close)
        split = int(md.n_bars * 0.7)
        res = GeneticEvolver(GAConfig(population_size=20, n_generations=8, seed=1)) \
            .evolve(md.slice(0, split), md.slice(split, md.n_bars))

        assert res.deflated_sharpe_prob < MIN_DEFLATED_SHARPE_PROB, (
            f"pure noise passed the deflated-Sharpe gate at "
            f"{res.deflated_sharpe_prob:.3f} - the control is not working")
        assert not res.beats_noise or res.in_sample.sharpe < res.noise_benchmark_sharpe * 1.5

    def test_simulate_is_causal(self):
        """A strategy's returns must not change when FUTURE bars move."""
        close = _random_walk(300, 5, seed=8)
        md = build_market_data(close)
        strat = decode(random_genome(np.random.default_rng(3)))
        base = simulate(md.slice(0, 100), strat)

        c2 = close.copy(); c2.iloc[250:] *= 2.0
        md2 = build_market_data(c2)
        after = simulate(md2.slice(0, 100), strat)
        assert np.allclose(base.daily_returns, after.daily_returns, atol=1e-10)

    def test_costs_reduce_returns(self):
        close = _random_walk(300, 5, seed=9)
        md = build_market_data(close)
        strat = decode(random_genome(np.random.default_rng(4)))
        free = simulate(md, strat, cost_bps=0.0)
        charged = simulate(md, strat, cost_bps=100.0)
        if free.n_trades > 0:
            assert charged.total_return < free.total_return

    def test_position_cap_is_enforced(self):
        close = _random_walk(300, 5, seed=10)
        md = build_market_data(close)
        g = random_genome(np.random.default_rng(5))
        g[-3] = 1.0                                  # max kelly fraction
        stats = simulate(md, decode(g), max_position=0.10)
        assert stats.avg_turnover <= 5 * 0.10 * 2 + 1e-9

    def test_fitness_penalises_bad_risk(self):
        from nightevolver.ga_engine import BacktestStats
        good = BacktestStats(sharpe=2.0, total_return=0.2, max_drawdown=0.05,
                             win_rate=0.60, n_trades=50, avg_turnover=0.1)
        deep_dd = BacktestStats(sharpe=2.0, total_return=0.2, max_drawdown=0.35,
                                win_rate=0.60, n_trades=50, avg_turnover=0.1)
        assert fitness(deep_dd) < fitness(good) * 0.5

    def test_abstention_beats_losing_but_does_not_beat_winning(self):
        """With no edge, not trading IS better than trading badly, so
        abstention must outrank a loser. But it must NOT be exactly 0.0:
        that used to beat every negative score, making never-trading the
        global optimum the moment a search started failing. Measured: 3
        of 8 seeds abstained and their 0.00s pulled an arm's mean from
        -1.45 up to -0.90, making the worse arm look better."""
        from nightevolver.ga_engine import ABSTENTION_FITNESS, BacktestStats
        none = BacktestStats(0.0, 0.0, 0.0, 0.0, 0, 0.0)
        loser = BacktestStats(-1.5, -0.1, 0.08, 0.30, 60, 0.1)
        winner = BacktestStats(1.5, 0.2, 0.05, 0.55, 60, 0.1)
        assert fitness(none) == ABSTENTION_FITNESS
        assert ABSTENTION_FITNESS < 0.0, "0.0 would dominate every negative"
        assert fitness(loser) < fitness(none) < fitness(winner)
        assert none.abstained and not loser.abstained

    def test_elites_are_carried_over_unmutated(self):
        """Elitism must PRESERVE the top N, not just seed them. If the
        elites were mutated, best-fitness could go backwards between
        generations."""
        close = _random_walk(300, 5, seed=31, drift=0.0004)
        md = build_market_data(close)
        res = GeneticEvolver(GAConfig(population_size=14, n_generations=10,
                                      elitism=2, seed=7)).evolve(md)
        best = [h["best_fitness"] for h in res.history]
        assert all(b2 >= b1 - 1e-12 for b1, b2 in zip(best, best[1:])), \
            f"best fitness regressed across generations: {best}"

    def test_top_strategies_are_not_near_duplicates(self):
        """Diversity check. If the top individuals are the same rule with
        cosmetic differences, the population has collapsed and the search
        is no longer exploring."""
        close = _random_walk(300, 5, seed=32, drift=0.0004)
        md = build_market_data(close)
        ev = GeneticEvolver(GAConfig(population_size=24, n_generations=8, seed=9))
        res = ev.evolve(md)
        assert res.history[-1]["mean_fitness"] != res.history[0]["mean_fitness"]
        # The evolved strategy must actually use its weight budget rather
        # than collapsing onto a single indicator.
        w = res.best_strategy.indicator_weights
        assert w.max() < 0.9, f"one indicator carries {w.max():.2f} of the vote"
        assert (w > 0.01).sum() >= 3, "fewer than 3 indicators carry any weight"

    def test_validation_window_scales_with_the_holding_period(self):
        """A 63-bar window cannot evaluate a 49-day hold, and nothing
        errors when you try - you just get a confident Sharpe on almost
        no trades. Measured: after the hold floor rose to 5 days the GA
        settled at a 49-day median hold, and the unchanged 63-bar window
        produced a median of TWO out-of-sample trades over 8 seeds."""
        from nightevolver.ga_engine import (
            MIN_TRADES_FOR_A_CLAIM, required_validation_bars,
        )
        # 20 trades, 10 assets, 90-day hold, duty cycle 0.25 -> 720 bars.
        # The duty-cycle factor is measured, not assumed: without it this
        # returned 180 bars claiming >=20 trades, and the run delivered 9.
        assert required_validation_bars(90, 10) == 720
        assert required_validation_bars(90, 10, duty_cycle=1.0) == 180
        # longer holds must demand strictly longer windows
        assert required_validation_bars(90, 10) > required_validation_bars(49, 10)
        # more assets amortise the requirement
        assert required_validation_bars(90, 20) < required_validation_bars(90, 10)
        # and the window it returns really does clear the bar
        for hold in (10, 49, 90):
            w = required_validation_bars(hold, 10)
            from nightevolver.ga_engine import TYPICAL_DUTY_CYCLE
            assert 10 * w / hold * TYPICAL_DUTY_CYCLE >= MIN_TRADES_FOR_A_CLAIM

    def test_universe_size_is_the_cheap_lever_on_window_length(self):
        """Trade count scales linearly with the universe, and bhavcopy is
        one file per session containing every symbol - so more names cost
        no extra fetching, while more history is bounded by the archive
        (UDiFF starts 2024)."""
        from nightevolver.ga_engine import required_validation_bars
        assert required_validation_bars(90, 50) < required_validation_bars(90, 10)
        assert (required_validation_bars(90, 10)
                == pytest.approx(required_validation_bars(90, 50) * 5, rel=0.02))

    def test_required_validation_bars_rejects_nonsense(self):
        from nightevolver.ga_engine import required_validation_bars
        with pytest.raises(ValueError):
            required_validation_bars(0, 10)
        with pytest.raises(ValueError):
            required_validation_bars(30, 0)
        with pytest.raises(ValueError):
            required_validation_bars(30, 10, duty_cycle=0.0)
        with pytest.raises(ValueError):
            required_validation_bars(30, 10, duty_cycle=1.5)

    def test_vol_targeting_matches_an_explicit_rolling_std(self):
        """The forecast is a vectorised rolling std. The readable version
        is a Python loop over T, but vol_hat depends only on prices, not
        the strategy, so that loop would re-run identically for all ~1050
        evaluations in one GA run. Pin the two against each other."""
        import pandas as pd

        from nightevolver.ga_engine import TRADING_DAYS, VOL_LOOKBACK
        close = _random_walk(300, 5, seed=21)
        md = build_market_data(close)
        c = md.close
        rets = np.zeros_like(c)
        rets[1:] = c[1:] / c[:-1] - 1.0

        loop = np.full(c.shape, np.nan)
        for t in range(VOL_LOOKBACK - 1, len(c)):
            loop[t] = rets[t - VOL_LOOKBACK + 1:t + 1].std(axis=0, ddof=1) \
                * np.sqrt(TRADING_DAYS)
        vec = (pd.DataFrame(rets).rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK)
               .std(ddof=1).to_numpy() * np.sqrt(TRADING_DAYS))
        m = np.isfinite(loop) & np.isfinite(vec)
        assert m.sum() > 1000
        np.testing.assert_allclose(loop[m], vec[m], atol=1e-12)

    def test_vol_targeting_does_not_look_ahead(self):
        """Sizing at t may use returns up to t and no further."""
        close = _random_walk(300, 4, seed=22)
        g = random_genome(np.random.default_rng(5))
        from nightevolver.genome import IDX_VOL_TARGET
        g[IDX_VOL_TARGET] = 0.8                       # force vol targeting on
        strat = decode(g)
        assert strat.vol_target > 0.0

        cut_raw = 220
        a = simulate(build_market_data(close), strat)
        c2 = close.copy()
        c2.iloc[cut_raw:] *= 2.0
        b = simulate(build_market_data(c2), strat)

        # Index mapping matters here and is easy to get wrong.
        # build_market_data drops WARMUP_BARS rows, so raw bar `cut_raw`
        # is md row cut_raw - WARMUP_BARS. And the row BEFORE that also
        # changes legitimately: its forward_return spans into the mutated
        # bar. That is the target moving, not a feature looking ahead.
        safe = cut_raw - WARMUP_BARS - 1
        assert safe > 50
        np.testing.assert_allclose(a.daily_returns[:safe], b.daily_returns[:safe],
                                   atol=1e-10)
        assert not np.allclose(a.daily_returns[safe:], b.daily_returns[safe:]), \
            "mutating the future changed nothing - the sizing is inert"

    def test_vol_scale_is_capped(self):
        """An unclamped target/estimate ratio diverges as the estimate
        goes to zero - which is what a halted or stale name produces."""
        from nightevolver.ga_engine import VOL_SCALE_CAP
        assert VOL_SCALE_CAP < np.inf and VOL_SCALE_CAP > 1.0
        flat = _random_walk(300, 3, seed=23)
        flat.iloc[:] = 100.0                          # zero volatility
        g = random_genome(np.random.default_rng(6))
        from nightevolver.genome import IDX_VOL_TARGET
        g[IDX_VOL_TARGET] = 1.0
        stats = simulate(build_market_data(flat), decode(g), max_position=0.10)
        assert np.isfinite(stats.sharpe)
        assert np.all(np.abs(stats.daily_returns) < 1.0)

    def test_profit_factor_reflects_asymmetric_payoffs(self):
        """The number win rate cannot show: few big wins beating many
        small losses."""
        from nightevolver.ga_engine import BacktestStats
        s = BacktestStats(0.0, 0.0, 0.0, 0.0, 0, 0.0)
        assert s.profit_factor == 0.0                 # default, no trades

    def test_fitness_ignores_win_rate(self):
        """Win rate is not the objective. Break-even at a 1-day hold is
        61.1% after 22bp, so the old WinRate/0.5 multiplier optimised
        hard for a number that loses money at the value being targeted -
        and it scored a 45%/2:1 strategy at 0.90x against a 60%/1:1
        loser at 1.20x. Sharpe already prices asymmetry."""
        from nightevolver.ga_engine import BacktestStats
        low_wr = BacktestStats(1.2, 0.15, 0.06, 0.45, 60, 0.1)
        high_wr = BacktestStats(1.2, 0.15, 0.06, 0.65, 60, 0.1)
        assert fitness(low_wr) == pytest.approx(fitness(high_wr))

    def test_few_trades_are_shrunk_toward_zero(self):
        """The measured artefact was an out-of-sample Sharpe of +1.18 on
        THREE trades. It must not outrank a real track record."""
        from nightevolver.ga_engine import BacktestStats
        lucky = BacktestStats(1.18, 0.02, 0.02, 0.667, 3, 0.02)
        real = BacktestStats(0.60, 0.10, 0.05, 0.52, 120, 0.1)
        assert fitness(lucky) < fitness(real)

    def test_ga_improves_fitness_over_generations(self):
        """Sanity: selection pressure must actually select."""
        close = _random_walk(400, 6, seed=11, drift=0.0006)
        md = build_market_data(close)
        res = GeneticEvolver(GAConfig(population_size=16, n_generations=8, seed=2)) \
            .evolve(md)
        first = res.history[0]["best_fitness"]
        last = res.history[-1]["best_fitness"]
        assert last >= first, f"best fitness went backwards: {first} -> {last}"


# ---------------------------------------------------------------------------
# Checkpoint contract
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def _result(self, seed: int = 1):
        close = _random_walk(400, 5, seed=seed)
        md = build_market_data(close)
        split = int(md.n_bars * 0.75)
        return GeneticEvolver(GAConfig(population_size=12, n_generations=4, seed=seed)) \
            .evolve(md.slice(0, split), md.slice(split, md.n_bars)), close.columns

    def test_roundtrip_preserves_the_strategy(self, tmp_path):
        res, tickers = self._result()
        save_checkpoint(res, list(tickers), tmp_path)
        ck = load_checkpoint(tmp_path / "nightevolver_best.json", require_gate=False)
        assert np.allclose(ck["genome"], res.best_genome)
        assert ck["decoded"].hold_days == res.best_strategy.hold_days
        assert np.allclose(ck["decoded"].indicator_weights,
                           res.best_strategy.indicator_weights)

    def test_checkpoint_is_json_not_pickle(self, tmp_path):
        """A pickle fetched from an ephemeral pod onto the trading box is
        arbitrary code execution on load."""
        res, tickers = self._result()
        p = save_checkpoint(res, list(tickers), tmp_path)
        with open(p) as f:
            json.load(f)                    # must parse as plain JSON

    def test_version_mismatch_is_refused(self, tmp_path):
        res, tickers = self._result()
        p = save_checkpoint(res, list(tickers), tmp_path)
        payload = json.loads(p.read_text())
        payload["genome_version"] = 999
        p.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="genome_version"):
            load_checkpoint(p, require_gate=False)

    def test_indicator_reorder_is_refused(self, tmp_path):
        """Reordered channels would attach evolved weights to the wrong
        indicators - silently trading a different strategy."""
        res, tickers = self._result()
        p = save_checkpoint(res, list(tickers), tmp_path)
        payload = json.loads(p.read_text())
        payload["indicator_names"] = list(reversed(payload["indicator_names"]))
        p.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="indicator ordering"):
            load_checkpoint(p, require_gate=False)

    def test_gate_refuses_an_underpowered_strategy(self, tmp_path):
        """The mechanism that stops an overfit nightly run from reaching
        the trading account merely by being the newest file."""
        res, tickers = self._result()
        p = save_checkpoint(res, list(tickers), tmp_path)
        payload = json.loads(p.read_text())
        payload["metrics"]["deflated_sharpe_prob"] = 0.10
        p.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="below the"):
            load_checkpoint(p, require_gate=True)
        load_checkpoint(p, require_gate=False)       # inspection still allowed


# ---------------------------------------------------------------------------
# Hetzner-side decoder
# ---------------------------------------------------------------------------

class TestDecoder:
    def test_signals_match_the_training_time_scorer(self, tmp_path):
        """What executes must equal what was evolved. This is the exact
        property whose absence broke the previous project."""
        close = _random_walk(400, 5, seed=12)
        md = build_market_data(close)
        res = GeneticEvolver(GAConfig(population_size=10, n_generations=3, seed=3)).evolve(md)
        save_checkpoint(res, list(close.columns), tmp_path)

        ev = EvolvedStrategy.from_checkpoint(tmp_path / "nightevolver_best.json",
                                             require_gate=False)
        live = ev.signal(close)
        expected = score_matrix(md.indicators[-1:], res.best_strategy)[0]
        assert np.allclose([s.score for s in live], expected, atol=1e-9)

    def test_short_history_stays_flat(self, tmp_path):
        close = _random_walk(400, 4, seed=13)
        md = build_market_data(close)
        res = GeneticEvolver(GAConfig(population_size=8, n_generations=2, seed=4)).evolve(md)
        save_checkpoint(res, list(close.columns), tmp_path)
        ev = EvolvedStrategy.from_checkpoint(tmp_path / "nightevolver_best.json",
                                             require_gate=False)
        out = ev.signal(close.iloc[:10])
        assert all(s.direction == 0 and s.target_weight == 0.0 for s in out)

    def test_latency_under_100ms(self, tmp_path):
        """Acceptance criterion 5."""
        close = _random_walk(400, 10, seed=14)
        md = build_market_data(close)
        res = GeneticEvolver(GAConfig(population_size=8, n_generations=2, seed=5)).evolve(md)
        save_checkpoint(res, list(close.columns), tmp_path)
        ev = EvolvedStrategy.from_checkpoint(tmp_path / "nightevolver_best.json",
                                             require_gate=False)
        ev.signal(close)                                    # warm caches
        live = ev.signal(close)
        assert live[0].latency_ms < 100.0, f"{live[0].latency_ms:.1f}ms exceeds budget"

    def test_position_respects_cap(self, tmp_path):
        close = _random_walk(400, 5, seed=15)
        md = build_market_data(close)
        res = GeneticEvolver(GAConfig(population_size=8, n_generations=2, seed=6)).evolve(md)
        save_checkpoint(res, list(close.columns), tmp_path)
        ev = EvolvedStrategy.from_checkpoint(tmp_path / "nightevolver_best.json",
                                             require_gate=False, max_position=0.10)
        assert all(abs(s.target_weight) <= 0.10 + 1e-9 for s in ev.signal(close))


# ---------------------------------------------------------------------------
# Q-learning
# ---------------------------------------------------------------------------

class TestQLearning:
    def test_learns_and_reports_out_of_sample(self):
        close = _random_walk(400, 5, seed=16)
        md = build_market_data(close)
        split = int(md.n_bars * 0.7)
        res = train_q_learning(md.slice(0, split), md.slice(split, md.n_bars),
                               QConfig(episodes=40, seed=1))
        assert res.q_table.shape[-1] == N_ACTIONS
        assert np.isfinite(res.in_sample_sharpe)
        assert res.out_of_sample_sharpe is not None
        assert 0.0 <= res.state_coverage <= 1.0
        assert res.q_table.any(), "Q-table never updated"

    def test_state_bins_are_fixed_not_data_derived(self):
        """Data-derived quantile edges would leak and would make a state
        mean different things in training vs execution."""
        a = build_states(build_market_data(_random_walk(300, 4, seed=17)))
        b = build_states(build_market_data(_random_walk(300, 4, seed=18) * 100))
        assert a.min() >= 0 and b.min() >= 0
        assert a.max() < 5 and b.max() < 5


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

class TestWalkForward:
    def test_windows_match_kronos(self):
        from kronos.backtest import WalkForwardBacktester
        from nightevolver.backtest_evolved import EvolvedWalkForward

        close = _random_walk(900, 5, seed=19)
        md = build_market_data(close)
        cfg = WalkForwardConfig(train_window=252, test_window=21)
        mine = EvolvedWalkForward(md, cfg).windows()
        theirs = WalkForwardBacktester(close.iloc[WARMUP_BARS + 1:], cfg).windows()
        assert len(mine) == len(theirs) and mine[0] == theirs[0]

    def test_run_reports_overfitting_gap_and_honest_trial_count(self):
        from nightevolver.backtest_evolved import EvolvedWalkForward

        close = _random_walk(700, 5, seed=20)
        md = build_market_data(close)
        res = EvolvedWalkForward(
            md, WalkForwardConfig(train_window=252, test_window=21),
            GAConfig(population_size=8, n_generations=3, seed=7),
        ).run(max_windows=2)

        assert res.signal.n_obs > 0
        assert np.isfinite(res.mean_overfitting_gap)
        # n_trials must count EVERY genome evaluated across ALL windows.
        assert res.total_trials == 2 * GAConfig(population_size=8, n_generations=3).search_budget
        assert isinstance(res.summary(), str)
