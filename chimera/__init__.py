"""
CHIMERA - a six-component hybrid trading system.

Deliberately a SEPARATE package from kronos/ and prometheus/: this is an
experimental research platform, and the one thing this project's history
makes clear is that entangling a new idea with the live paper-trading
stack makes it impossible to attribute a result to any one change.
chimera/ reuses kronos/backtest.py's honest walk-forward machinery
(WalkForwardConfig, SignalDiagnostic, deflated_sharpe) and nothing else.

The six components, and what each is actually doing (not the marketing
name):

  1. connectome.py   - dynamic market network. Rolling shrinkage precision
                       matrix -> partial-correlation graph -> per-asset
                       centrality/clustering + global spectral features
                       (Fiedler value, von Neumann entropy). Connectivity
                       state is a regime signal, not just a feature.

  2. qubo_select.py  - feature selection as QUBO (max relevance, min
                       redundancy, cardinality-penalised), solved with
                       ballistic Simulated Bifurcation - a real
                       quantum-inspired solver (Goto et al.), not a
                       rebranded greedy filter.

  3. chaotic_attn.py - a Lorenz oscillator seeded FROM the market state
                       drives per-head attention temperature. The point
                       is sensitive dependence: nearby market states
                       diverge into separable trajectories, giving a
                       chaotic-reservoir basis expansion inside attention.

  4. grpo.py         - critic-free policy optimisation. GRPO's group-
                       relative advantage (no value network) plus DAPO's
                       clip-higher, dynamic sampling, and asset-level
                       (token-level analogue) loss aggregation.

  5. pinn.py         - no-arbitrage as a differentiable penalty: (a) a
                       Black-Scholes PDE residual on an auxiliary value
                       head via autograd, and (b) the cross-sectional
                       condition that a (near-)zero-variance portfolio
                       must have (near-)zero predicted return.

  6. qd_archive.py   - MAP-Elites over strategy genomes. Behaviour
                       descriptors are turnover / net exposure /
                       concentration, so the archive yields a portfolio
                       of BEHAVIOURALLY distinct strategies to ensemble,
                       not N copies of one local optimum.

Composition is a pipeline, not a bag of parts:

    OHLCV -> connectome ---> feature bank -> QUBO subset
          -> chaotic-attention encoder -> { PINN value head (constraint)
                                          , GRPO policy head (action) }
          -> MAP-Elites ensemble -> NSE integer-share Kelly sizing
"""

from chimera.connectome import ConnectomeFeatures, FinancialConnectome
from chimera.qubo_select import QUBOFeatureSelector, SimulatedBifurcation
from chimera.chaotic_attn import ChaoticAttentionEncoder, LorenzReservoir
from chimera.grpo import DAPOConfig, GroupRelativePolicy
from chimera.pinn import NoArbitragePenalty, black_scholes_residual
from chimera.qd_archive import MapElitesArchive, StrategyGenome

__all__ = [
    "FinancialConnectome", "ConnectomeFeatures",
    "SimulatedBifurcation", "QUBOFeatureSelector",
    "LorenzReservoir", "ChaoticAttentionEncoder",
    "GroupRelativePolicy", "DAPOConfig",
    "NoArbitragePenalty", "black_scholes_residual",
    "MapElitesArchive", "StrategyGenome",
]
