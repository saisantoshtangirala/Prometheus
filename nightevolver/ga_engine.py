"""
Genetic algorithm over strategy genomes, with the multiple-testing
control the naive version is missing.

THE FLAW THIS MODULE EXISTS TO FIX
----------------------------------
A GA with population 50 over 20 generations evaluates ~1000 strategies
on one training window and returns the best. That is 1000 trials, and
the expected maximum Sharpe of N INDEPENDENT PURE-NOISE strategies grows
like sqrt(2 ln N):

    N=10    -> ~2.1 standard errors
    N=1000  -> ~3.7 standard errors
    N=10000 -> ~4.3 standard errors

On a 252-bar window the standard error of an annualised Sharpe is
roughly 1.0. So searching 1000 random strategies over one year of data
is EXPECTED to surface an in-sample Sharpe near 3 even if every strategy
is worthless. Reporting that number as the strategy's Sharpe is not a
small optimism - it is the entire result.

This is almost certainly the mechanism behind the "Sharpe 2.53" and
"55.5% win rate" figures quoted for GA trading systems. They are the
maximum of a large search, reported as if they were an estimate of a
single strategy's skill.

Two controls, both mandatory here:

1. `deflated_sharpe_prob` is computed with n_trials = the ACTUAL search
   budget (population x generations), not a token 3. Bailey & Lopez de
   Prado's deflation exists precisely to discount a maximum-of-N.

2. `EvolutionResult` reports IN-SAMPLE and OUT-OF-SAMPLE Sharpe side by
   side. The GAP between them is the overfitting measure, and it is the
   number to look at first. A large in-sample Sharpe with an OOS Sharpe
   near zero is the signature of a search that found noise, and it is
   invisible if you only print the winner's fitness.

Implementation notes: the GA is written directly in numpy rather than
via DEAP. DEAP is in requirements.txt but not installed in this
environment (so DEAP-based code could not be tested here at all), and
its real value is flexible representations - variable-length genomes,
GP trees. This genome is a fixed-length real vector in [0,1]^65, for
which DEAP's machinery reduces to a few lines of array indexing that
are cheaper to write, faster to run, and testable right now.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nightevolver.data_loader import MarketData
from nightevolver.genome import (
    DecodedStrategy, GENOME_LENGTH, crossover, decode, mutate, random_genome,
    score_matrix,
)

logger = logging.getLogger("nightevolver.ga")

TRADING_DAYS = 252

# Fitness penalty gates, from the spec.
# MIN_WIN_RATE is deliberately gone - see fitness(). Win rate is not the
# objective and gating on it penalised asymmetric-payoff strategies,
# which are the ones that actually clear costs.
MAX_DRAWDOWN = 0.20
PENALTY_MULTIPLIER = 0.1

# Evidence shrinkage: a strategy keeps n/(n+K) of its score. K=20 means
# 3 trades keep 13%, 20 keep 50%, 100 keep 83%. Chosen so the measured
# artefact (Sharpe +1.18 on three trades) cannot win a tournament
# against a strategy with a real track record.
TRADE_SHRINKAGE_K = 20.0

# Vol targeting: trailing window for the forecast, and a cap on the
# resulting size multiplier. The cap matters - an unclamped
# target/estimate ratio goes to infinity as the estimate goes to zero,
# which is exactly what a halted or stale name produces.
VOL_LOOKBACK = 20
VOL_SCALE_CAP = 3.0

# Below this many out-of-sample trades, a Sharpe is not a measurement
# and the summary says so out loud.
MIN_TRADES_FOR_A_CLAIM = 20

# Not exactly 0.0. Zero used to beat every negative score, which made
# never-trading the global optimum as soon as a search started failing -
# measured: 3 of 8 seeds abstained and their 0.00s pulled an arm's mean
# ABOVE the arm that actually traded. Small and negative keeps
# abstention better than losing without letting it dominate.
ABSTENTION_FITNESS = -0.05


@dataclass
class GAConfig:
    population_size: int = 50
    n_generations: int = 20
    tournament_size: int = 3
    crossover_prob: float = 0.7
    mutation_rate: float = 0.1
    mutation_sigma: float = 0.05
    elitism: int = 2
    cost_bps: float = 22.0        # NSE delivery round trip, STT-dominated
    max_position: float = 0.10    # 10% of capital per position (spec)
    seed: int = 42

    @property
    def search_budget(self) -> int:
        """Total genome evaluations - the N in 'best of N'.

        This is the number that must be handed to the deflated Sharpe
        calculation. Under-reporting it is how a search result gets
        mistaken for a skill estimate.
        """
        return self.population_size * (self.n_generations + 1)


@dataclass
class BacktestStats:
    """Performance of ONE genome over one window."""

    sharpe: float
    total_return: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    avg_turnover: float
    daily_returns: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    profit_factor: float = 0.0

    @property
    def abstained(self) -> bool:
        """No trades placed. Distinct from 'traded and made nothing'.

        These were conflated: a zero-trade run reported Sharpe 0.00 and
        got averaged in with real results, which pulled one arm's mean
        from -1.45 to -0.90 and made it look like the better arm. Any
        reporting path must branch on this rather than on sharpe == 0.
        """
        return self.n_trades == 0


def simulate(md: MarketData, strat: DecodedStrategy, cost_bps: float = 22.0,
             max_position: float = 0.10) -> BacktestStats:
    """Run one decoded strategy over a MarketData window.

    Vectorised across ASSETS, looped over TIME. The time loop is not
    laziness: the trailing stop is genuinely path-dependent, and
    replacing it with a vectorised approximation would mean the backtest
    measures a different strategy from the one that trades - the exact
    failure this project already made once. ~250 iterations of 10-wide
    numpy ops costs ~2ms, which is affordable at 1000 evaluations.

    CAUSALITY: the position held over bar t is decided from indicators
    at t and earns forward_returns[t] (the t -> t+1 move). No future
    information enters the decision.
    """
    scores = score_matrix(md.indicators, strat)       # [T, A]
    fwd = md.forward_returns                          # [T, A]
    close = md.close
    T, A = scores.shape

    # Trailing realised volatility, annualised, for vol targeting.
    # CAUSAL: vol_hat[t] uses returns up to and including t, and is used
    # to size the position held over t -> t+1. Row 0 has no history, so
    # it falls back to the target itself (a neutral 1.0x multiplier).
    vol_hat = None
    if strat.vol_target > 0.0:
        with np.errstate(divide="ignore", invalid="ignore"):
            rets = np.zeros_like(close)
            rets[1:] = close[1:] / np.maximum(close[:-1], 1e-12) - 1.0
        # Vectorised rolling std. The obvious version is a Python loop
        # over T, but vol_hat depends only on PRICES, not on the strategy
        # - so that loop would re-run identically for all ~1050
        # evaluations in a single GA run. pandas' rolling is C-level and
        # numerically careful, and this is called once per simulate().
        import pandas as pd
        sd = (pd.DataFrame(rets).rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK)
              .std(ddof=1).to_numpy() * np.sqrt(TRADING_DAYS))
        vol_hat = np.where(np.isfinite(sd) & (sd > 1e-6), sd, strat.vol_target)

    pos = np.zeros(A)               # signed position weight per asset
    entry_px = np.zeros(A)
    peak_px = np.zeros(A)           # best price since entry, for trailing stop
    held = np.zeros(A, dtype=np.int64)

    positions = np.zeros((T, A))
    trades = 0

    for t in range(T):
        px = close[t]
        active = pos != 0.0

        if active.any():
            held[active] += 1
            long_a = active & (pos > 0)
            short_a = active & (pos < 0)
            # Track the most favourable price seen since entry.
            peak_px[long_a] = np.maximum(peak_px[long_a], px[long_a])
            peak_px[short_a] = np.minimum(peak_px[short_a], px[short_a])

            with np.errstate(divide="ignore", invalid="ignore"):
                draw_long = np.where(peak_px > 0, (peak_px - px) / peak_px, 0.0)
                draw_short = np.where(px > 0, (px - peak_px) / np.maximum(px, 1e-12), 0.0)
            stop_hit = ((long_a & (draw_long > strat.trailing_stop))
                        | (short_a & (draw_short > strat.trailing_stop)))
            expired = active & (held >= strat.hold_days)
            # Exit if the vote flips against the open position.
            flipped = active & (np.sign(scores[t]) == -np.sign(pos)) \
                & (np.abs(scores[t]) > strat.conviction_floor)

            close_now = stop_hit | expired | flipped
            if close_now.any():
                pos[close_now] = 0.0
                held[close_now] = 0
                entry_px[close_now] = 0.0
                peak_px[close_now] = 0.0

        # Entries: only into flat names, only on sufficient conviction.
        flat = pos == 0.0
        want = flat & (np.abs(scores[t]) > strat.conviction_floor)
        if want.any():
            size = np.abs(scores[t][want]) * strat.kelly_fraction
            if vol_hat is not None:
                # Scale toward a constant risk contribution: a quiet name
                # gets more capital than a wild one for equal conviction.
                # Capped at VOL_SCALE_CAP so a near-zero vol estimate
                # cannot demand an enormous position - the classic way a
                # vol-targeting rule blows up on a stale or halted name.
                scale = np.clip(strat.vol_target / vol_hat[t][want],
                                0.0, VOL_SCALE_CAP)
                size = size * scale
            size = np.clip(size, 0.0, max_position)
            pos[want] = np.sign(scores[t][want]) * size
            entry_px[want] = px[want]
            peak_px[want] = px[want]
            held[want] = 0
            trades += int(want.sum())

        positions[t] = pos

    gross = (positions * fwd).sum(axis=1)
    turnover = np.abs(np.diff(positions, axis=0, prepend=np.zeros((1, A)))).sum(axis=1)
    net = gross - turnover * (cost_bps / 10_000.0)

    sd = float(net.std())
    sharpe = float(net.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 1e-12 else 0.0

    equity = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(equity)
    mdd = float(np.abs(((equity - peak) / np.maximum(peak, 1e-12)).min())) if T else 0.0

    traded = net[turnover > 0]
    win_rate = float((traded > 0).mean()) if traded.size else 0.0

    # Profit factor: gross gains / gross losses. Reported, not optimised.
    # It is the number that shows an asymmetric payoff, which win rate
    # cannot: a 45%-hit-rate strategy at 2:1 has PF 1.6 and makes money,
    # while a 60%-hit-rate strategy at 0.5:1 has PF 0.75 and does not.
    gains = float(net[net > 0].sum())
    losses = float(-net[net < 0].sum())
    profit_factor = (gains / losses) if losses > 1e-12 else (
        float("inf") if gains > 0 else 0.0)

    return BacktestStats(
        sharpe=sharpe, total_return=float(equity[-1] - 1.0) if T else 0.0,
        max_drawdown=mdd, win_rate=win_rate, n_trades=trades,
        avg_turnover=float(turnover.mean()) if T else 0.0, daily_returns=net,
        profit_factor=profit_factor,
    )


def fitness(stats: BacktestStats) -> float:
    """Net-of-cost Sharpe x (1 - MaxDD) x evidence shrinkage.

    REDEFINED. The previous objective was
    `Sharpe x (1 - MaxDD) x (WinRate / 0.5)` with a MIN_WIN_RATE gate,
    and both win-rate terms are now gone. Three measured reasons:

    1. WIN RATE IS THE WRONG OBJECTIVE, and provably so on this data.
       Break-even win rate after 22bp round-trip costs at a 1-day hold
       is 61.1%; the spec's 55% target would lose 12.1bp per round trip.
       Multiplying fitness by WinRate/0.5 optimised hard for a number
       that, at the value being aimed at, loses money.

    2. IT PENALISED THE PAYOFF PROFILE THAT ACTUALLY WORKS. A 45% hit
       rate at 2:1 beats 60% at 1:1, but the old term scored the first
       at 0.90x and the second at 1.20x. Sharpe already prices
       asymmetry correctly - it depends on mean and dispersion, not on
       hit count - so removing the multiplier does not lose the
       information, it stops double-counting the wrong half of it.

    3. EVIDENCE SHRINKAGE replaces it. The measured pathology was an
       out-of-sample Sharpe of +1.18 on THREE trades - two winners and
       a loser - which a naive report would have called a 66.7% win
       rate. Sharpe over a window where most days are flat has a tiny
       denominator, so a handful of lucky trades produces a large
       number. `n / (n + K)` shrinks that toward zero: 3 trades keep
       13% of their score, 100 trades keep 83%.

    Abstention (`n_trades == 0`) scores ABSTENTION_FITNESS, a small
    negative. It stays BETTER than actively losing money, because with
    no edge not trading genuinely is optimal - but it is no longer
    exactly 0.0, which used to beat every negative and made
    never-trading the global optimum the moment a search started
    failing. `stats.abstained` carries the distinction to the reporting
    layer so an abstention is never averaged in as a zero-Sharpe
    result, which is exactly what inflated the flow-arm mean from
    -1.45 to -0.90 in the last A/B.
    """
    if stats.n_trades == 0:
        return ABSTENTION_FITNESS
    base = stats.sharpe * (1.0 - stats.max_drawdown)
    base *= stats.n_trades / (stats.n_trades + TRADE_SHRINKAGE_K)
    if stats.max_drawdown > MAX_DRAWDOWN:
        base *= PENALTY_MULTIPLIER
    return float(base) if np.isfinite(base) else ABSTENTION_FITNESS


def expected_max_sharpe_from_noise(n_trials: int, n_obs: int) -> float:
    """Expected best annualised Sharpe from `n_trials` WORTHLESS strategies.

    The benchmark any GA result must beat to mean anything. Uses the
    standard extreme-value approximation for the maximum of n_trials
    standard normals, scaled by the standard error of a Sharpe estimate
    over n_obs bars.
    """
    if n_trials < 2 or n_obs < 2:
        return 0.0
    e = 0.5772156649                                   # Euler-Mascheroni
    from scipy.stats import norm
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
    max_z = z1 * (1 - e) + z2 * e                      # E[max of n_trials N(0,1)]
    return float(max_z * np.sqrt(TRADING_DAYS / n_obs))


@dataclass
class EvolutionResult:
    """Outcome of one evolution run - reported so overfitting is visible."""

    best_genome: np.ndarray
    best_strategy: DecodedStrategy
    in_sample: BacktestStats
    out_of_sample: Optional[BacktestStats]
    search_budget: int
    noise_benchmark_sharpe: float
    deflated_sharpe_prob: float
    generations_run: int
    elapsed_seconds: float
    history: List[Dict] = field(default_factory=list)

    @property
    def overfitting_gap(self) -> float:
        """In-sample minus out-of-sample Sharpe. THE number to read first."""
        if self.out_of_sample is None:
            return float("nan")
        return self.in_sample.sharpe - self.out_of_sample.sharpe

    @property
    def beats_noise(self) -> bool:
        """Did the winner beat what pure noise would have produced?"""
        return self.in_sample.sharpe > self.noise_benchmark_sharpe

    @staticmethod
    def _stat_line(label: str, s: BacktestStats) -> str:
        # An abstention is NOT a Sharpe of 0.00. Printing it as one is
        # how three zero-trade seeds ended up averaged into an arm's
        # mean and made it look like the better arm.
        if s.abstained:
            return (f"  {label:11s} ABSTAINED - placed no trades "
                    f"(not a zero-Sharpe result; nothing was risked)")
        pf = "inf" if not np.isfinite(s.profit_factor) else f"{s.profit_factor:.2f}"
        return (f"  {label:11s} Sharpe {s.sharpe:+.2f}  PF {pf}  "
                f"win {s.win_rate:.1%}  maxDD {s.max_drawdown:.1%}  "
                f"trades {s.n_trades}")

    def summary(self) -> str:
        oos = self.out_of_sample
        lines = [
            f"GA: {self.generations_run} generations, budget {self.search_budget} "
            f"evaluations, {self.elapsed_seconds:.1f}s",
            self._stat_line("IN-SAMPLE", self.in_sample),
        ]
        if oos is not None:
            lines.append(self._stat_line("OUT-SAMPLE", oos))
            if not (oos.abstained or self.in_sample.abstained):
                lines.append(f"  OVERFITTING GAP  {self.overfitting_gap:+.2f} Sharpe "
                             f"(in-sample minus out-of-sample)")
            if not oos.abstained and oos.n_trades < MIN_TRADES_FOR_A_CLAIM:
                lines.append(
                    f"  !! only {oos.n_trades} out-of-sample trades. A Sharpe on "
                    f"fewer than {MIN_TRADES_FOR_A_CLAIM} trades is not a "
                    f"measurement - a +1.18 on 3 trades has already appeared "
                    f"in this project and did not survive 8 seeds.")
        lines.append(
            f"  NOISE BENCHMARK  best-of-{self.search_budget} worthless strategies "
            f"would score Sharpe ~{self.noise_benchmark_sharpe:+.2f}  "
            f"-> winner {'BEATS' if self.beats_noise else 'DOES NOT BEAT'} noise")
        lines.append(f"  deflated P(SR>0) = {self.deflated_sharpe_prob:.3f} "
                     f"(n_trials={self.search_budget}, gate is 0.95)")
        return "\n".join(lines)


class GeneticEvolver:
    """Tournament GA over strategy genomes."""

    def __init__(self, config: Optional[GAConfig] = None):
        self.cfg = config or GAConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    def _evaluate(self, genome: np.ndarray, md: MarketData) -> Tuple[float, BacktestStats]:
        stats = simulate(md, decode(genome), self.cfg.cost_bps, self.cfg.max_position)
        return fitness(stats), stats

    def _tournament(self, pop: np.ndarray, fits: np.ndarray) -> np.ndarray:
        idx = self.rng.integers(0, len(pop), size=self.cfg.tournament_size)
        return pop[idx[int(np.argmax(fits[idx]))]].copy()

    def evolve(self, train: MarketData,
               validation: Optional[MarketData] = None,
               on_generation: Optional[Callable[[int, float], None]] = None
               ) -> EvolutionResult:
        """Run the GA on `train`, then score the winner on `validation`.

        `validation` is strictly out-of-sample and is used ONLY to
        report - never to select. Selecting on it would make it
        in-sample too, which is what "pick the best of the top 3 on the
        21-day OOS window" quietly does.
        """
        cfg = self.cfg
        t0 = time.time()

        pop = np.stack([random_genome(self.rng) for _ in range(cfg.population_size)])
        fits = np.zeros(cfg.population_size)
        stats_cache: List[Optional[BacktestStats]] = [None] * cfg.population_size
        for i, g in enumerate(pop):
            fits[i], stats_cache[i] = self._evaluate(g, train)

        history: List[Dict] = []
        for gen in range(cfg.n_generations):
            order = np.argsort(-fits)
            new_pop = [pop[i].copy() for i in order[: cfg.elitism]]     # elitism

            while len(new_pop) < cfg.population_size:
                p1 = self._tournament(pop, fits)
                p2 = self._tournament(pop, fits)
                if self.rng.random() < cfg.crossover_prob:
                    c1, c2 = crossover(p1, p2, self.rng)
                else:
                    c1, c2 = p1, p2
                new_pop.append(mutate(c1, self.rng, cfg.mutation_rate, cfg.mutation_sigma))
                if len(new_pop) < cfg.population_size:
                    new_pop.append(mutate(c2, self.rng, cfg.mutation_rate, cfg.mutation_sigma))

            pop = np.stack(new_pop)
            for i, g in enumerate(pop):
                fits[i], stats_cache[i] = self._evaluate(g, train)

            best = float(fits.max())
            history.append({"generation": gen + 1, "best_fitness": best,
                            "mean_fitness": float(fits.mean()),
                            "best_sharpe": stats_cache[int(np.argmax(fits))].sharpe})
            if on_generation:
                on_generation(gen + 1, best)
            logger.info("[ga] gen %d/%d best_fitness=%.4f mean=%.4f",
                        gen + 1, cfg.n_generations, best, float(fits.mean()))

        bi = int(np.argmax(fits))
        best_genome = pop[bi].copy()
        in_sample = stats_cache[bi]
        oos = simulate(validation, decode(best_genome), cfg.cost_bps,
                       cfg.max_position) if validation is not None else None

        n_obs = max(len(in_sample.daily_returns), 2)
        noise = expected_max_sharpe_from_noise(cfg.search_budget, n_obs)

        # Deflate against the REAL search budget. Using the default
        # n_trials=3 here would overstate significance by orders of
        # magnitude - a 1000-evaluation search is not three tries.
        from kronos.backtest import deflated_sharpe
        scored = oos if oos is not None else in_sample
        dsr = deflated_sharpe(scored.sharpe, max(len(scored.daily_returns), 3),
                              n_trials=cfg.search_budget)

        return EvolutionResult(
            best_genome=best_genome, best_strategy=decode(best_genome),
            in_sample=in_sample, out_of_sample=oos,
            search_budget=cfg.search_budget, noise_benchmark_sharpe=noise,
            deflated_sharpe_prob=dsr, generations_run=cfg.n_generations,
            elapsed_seconds=time.time() - t0, history=history,
        )
