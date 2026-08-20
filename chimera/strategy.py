"""
ChimeraStrategy - all six components wired into kronos/backtest.py's
Strategy interface (fit / weights_for), so it drops straight into the
existing walk-forward harness and is judged by the same honest machinery
(deflated Sharpe, signal-direction diagnostic) as everything else.

Per walk-forward window, fit() runs the full pipeline:

  1. build the causal feature bank            (features + component 1)
  2. QUBO-select k features on TRAIN only     (component 2)
  3. fit the no-arbitrage null space on TRAIN (component 5A)
  4. train the chaotic-attention trunk with
     supervised + no-arb + PDE loss           (components 3, 5)
  5. train the policy head with GRPO/DAPO     (component 4)
  6. illuminate the MAP-Elites archive by
     evaluating genomes on TRAIN              (component 6)

weights_for() then encodes the recent window, gets the model signal, and
blends the archive's behaviourally-diverse elites into a final weight
vector.

LOOK-AHEAD DISCIPLINE: fit() sees only train_returns. Feature
standardisation, QUBO selection, the no-arbitrage basis, network weights
and the elite archive are all fitted inside fit(). weights_for() is
strictly read-only w.r.t. fitted state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from chimera.features import build_features, standardise
from chimera.model import ChimeraConfig, ChimeraNet, ChimeraTrainer
from chimera.qd_archive import (
    MapElitesArchive, StrategyGenome, apply_genome, behaviour_descriptors,
)
from chimera.qubo_select import QUBOFeatureSelector, SBConfig
from chimera.sizing import NSECostModel


@dataclass
class ChimeraStrategyConfig:
    """Everything tunable, in one place.

    `fast` collapses the expensive stages (SB replicas, epochs,
    MAP-Elites iterations) for tests and smoke runs. It exists because a
    125-window backtest that takes 8 hours cannot be iterated on, and a
    harness nobody runs is a harness that rots - a lesson this repo has
    already paid for.
    """

    model: ChimeraConfig = field(default_factory=ChimeraConfig)
    n_features_selected: int = 12
    qubo_alpha: float = 1.0
    qubo_beta: float = 0.8
    connectome_window: int = 60

    policy_epochs: int = 20
    qd_iterations: int = 120
    qd_initial: int = 30
    qd_bins: int = 5
    qd_ensemble_top_n: int = 8

    cost_bps: float = 22.0          # NSE delivery round trip (see sizing.py)
    max_weight: float = 0.25
    allow_short: bool = False       # NSE cash segment reality
    seed: int = 42

    @classmethod
    def fast(cls) -> "ChimeraStrategyConfig":
        m = ChimeraConfig(seq_len=16, d_model=32, n_heads=2, n_layers=1, epochs=3)
        return cls(model=m, n_features_selected=8, policy_epochs=3,
                   qd_iterations=25, qd_initial=10, qd_bins=4, connectome_window=20)


class ChimeraStrategy:
    """The six-component system, as a walk-forward Strategy."""

    name = "chimera"

    def __init__(self, config: Optional[ChimeraStrategyConfig] = None):
        self.cfg = config or ChimeraStrategyConfig()
        self.net: Optional[ChimeraNet] = None
        self.trainer: Optional[ChimeraTrainer] = None
        self.selector: Optional[QUBOFeatureSelector] = None
        self.archive: Optional[MapElitesArchive] = None
        self._feat_mu: Optional[np.ndarray] = None
        self._feat_sd: Optional[np.ndarray] = None
        self._signal_scale: float = 1.0
        self._n_assets: int = 0
        self._prev_weights: Optional[np.ndarray] = None
        self.fit_report: Dict[str, object] = {}

    # -- windowing helpers -------------------------------------------------

    def _sequences(self, feats: np.ndarray, targets: Optional[np.ndarray],
                   seq_len: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """[T, A, F] -> X [S, A, seq_len, F], y [S, A].

        Sample s ends at bar t and predicts bar t's return from features
        over [t-seq_len, t). Because build_features() rows are already
        strictly backward-looking, this cannot leak.
        """
        T, A, F = feats.shape
        xs, ys = [], []
        for t in range(seq_len, T):
            xs.append(feats[t - seq_len : t].transpose(1, 0, 2))   # [A, seq, F]
            if targets is not None:
                ys.append(targets[t])
        if not xs:
            raise ValueError(f"not enough bars ({T}) for seq_len {seq_len}")
        X = torch.tensor(np.stack(xs), dtype=torch.float32)
        y = torch.tensor(np.stack(ys), dtype=torch.float32) if targets is not None else None
        return X, y

    # -- Strategy interface ------------------------------------------------

    def fit(self, train_returns: np.ndarray) -> None:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        R = np.asarray(train_returns, dtype=np.float64)
        T, A = R.shape
        self._n_assets = A

        # 1. causal feature bank (includes component 1's connectome)
        bank = build_features(R, connectome_window=cfg.connectome_window)
        feats = bank.values[bank.warmup :]          # drop unusable warmup rows
        rets = R[bank.warmup :]
        if feats.shape[0] < cfg.model.seq_len + 8:
            raise ValueError(
                f"train window too short: {feats.shape[0]} usable bars after "
                f"{bank.warmup}-bar warmup, need >= {cfg.model.seq_len + 8}"
            )

        # 2. QUBO feature selection, on TRAIN only
        flat = feats.reshape(-1, feats.shape[2])
        flat_y = rets.reshape(-1)
        self.selector = QUBOFeatureSelector(
            k=min(cfg.n_features_selected, flat.shape[1]),
            alpha=cfg.qubo_alpha, beta=cfg.qubo_beta,
            sb_config=SBConfig(seed=cfg.seed),
        ).fit(flat, flat_y)
        sel = self.selector.selected_
        feats = feats[:, :, sel]

        # standardisation statistics: TRAIN only, stored for inference
        f2 = feats.reshape(-1, feats.shape[2])
        self._feat_mu = f2.mean(axis=0, keepdims=True)
        self._feat_sd = np.where(f2.std(axis=0, keepdims=True) > 1e-12,
                                 f2.std(axis=0, keepdims=True), 1.0)
        feats = (feats - self._feat_mu) / self._feat_sd

        X, y = self._sequences(feats, rets, cfg.model.seq_len)

        # 3-5. build and train the network
        self.net = ChimeraNet(n_features=len(sel), n_assets=A, cfg=cfg.model)
        self.trainer = ChimeraTrainer(self.net, cfg.model)
        self.trainer.fit_constraints(rets)

        sup_hist = []
        for _ in range(cfg.model.epochs):
            sup_hist.append(self.trainer.train_epoch(X, y))

        pol_hist = []
        for _ in range(cfg.policy_epochs):
            pol_hist.append(self.trainer.train_policy_epoch(X, y, cost_bps=cfg.cost_bps))

        # 6. MAP-Elites over deployment genomes, scored on TRAIN
        self.net.eval()
        with torch.no_grad():
            signals = self.net.predict_returns(X).numpy()        # [S, A]
        realised = y.numpy()                                      # [S, A]

        # SIGNAL NORMALISATION - load-bearing, and the source of a real
        # bug caught in integration testing. The return head predicts on
        # the scale of daily returns (~1e-3..1e-1), but the genome's
        # signal_threshold gene is sampled in [0, 0.9] and compared
        # against tanh(gain * signal). With a raw signal of 0.17 and
        # gain 0.7 the squashed value is ~0.12, under almost every
        # sampled threshold - so EVERY genome produced an all-zero
        # portfolio, and MAP-Elites then *selected for* that, because
        # "never trade" scores exactly 0.0 while any real position loses
        # money to 22bp costs. The archive's best fitness was 0.0000:
        # a perfectly rational conclusion, reached for the wrong reason,
        # that made the entire search space unreachable.
        #
        # Dividing by the train-set signal scale puts the signal in
        # units of its own standard deviation, so the gain and threshold
        # genes address the distribution they were designed for. This is
        # the same failure mode as the SNN size_scale collapse found in
        # this repo's audit: a threshold compared against a quantity on
        # a different scale.
        self._signal_scale = float(np.std(signals))
        if not np.isfinite(self._signal_scale) or self._signal_scale < 1e-9:
            self._signal_scale = 1.0
        signals = signals / self._signal_scale

        def evaluate(g: StrategyGenome) -> Tuple[float, Dict[str, float]]:
            W, prev = [], None
            for s in signals:
                w = apply_genome(s, g, prev)
                if not cfg.allow_short:
                    w = np.clip(w, 0.0, None)
                W.append(w); prev = w
            W = np.asarray(W)
            # PnL is w_t applied to the return realised at t, minus the
            # cost of getting into w_t. Cost is charged inside fitness so
            # the archive cannot reward a churning strategy.
            pnl = (W * realised).sum(axis=1)
            turn = np.abs(np.diff(W, axis=0, prepend=np.zeros((1, W.shape[1])))).sum(axis=1)
            net = pnl - turn * (cfg.cost_bps / 10_000.0)
            sd = net.std()
            fit = float(net.mean() / sd * np.sqrt(252)) if sd > 1e-12 else 0.0
            return fit, behaviour_descriptors(W)

        self.archive = MapElitesArchive(bins=cfg.qd_bins, seed=cfg.seed).illuminate(
            evaluate, n_iterations=cfg.qd_iterations, n_initial=cfg.qd_initial,
        )

        self._prev_weights = None
        self.fit_report = {
            "selected_features": [bank.names[i] for i in sel],
            "n_train_sequences": int(X.shape[0]),
            "supervised_final": sup_hist[-1] if sup_hist else {},
            "policy_final": pol_hist[-1] if pol_hist else {},
            "archive": self.archive.summary(),
            "arbitrage_score": self.trainer.no_arb.arbitrage_score(
                torch.tensor(signals, dtype=torch.float32)),
        }

    def raw_signal(self, recent_returns: np.ndarray) -> np.ndarray:
        """Model signal for the most recent bar - no genome, no sizing.

        Exposed separately so the signal-direction diagnostic can measure
        the MODEL, not the model plus deployment heuristics. That
        separation is what made the earlier no-edge result interpretable
        rather than ambiguous.
        """
        if self.net is None:
            return np.zeros(self._n_assets or np.asarray(recent_returns).shape[1])
        cfg = self.cfg
        R = np.asarray(recent_returns, dtype=np.float64)

        bank = build_features(R, connectome_window=cfg.connectome_window)
        feats = bank.values[:, :, self.selector.selected_]
        feats = (feats - self._feat_mu) / self._feat_sd

        need = cfg.model.seq_len
        if feats.shape[0] < need:
            pad = np.zeros((need - feats.shape[0], feats.shape[1], feats.shape[2]))
            feats = np.concatenate([pad, feats], axis=0)
        window = feats[-need:].transpose(1, 0, 2)[None]           # [1, A, seq, F]

        self.net.eval()
        with torch.no_grad():
            return self.net.predict_returns(
                torch.tensor(window, dtype=torch.float32)).numpy()[0]

    def weights_for(self, recent_returns: np.ndarray) -> np.ndarray:
        """Target portfolio weights from the QD ensemble."""
        if self.net is None or self.archive is None:
            return np.zeros(np.asarray(recent_returns).shape[1])
        # Normalise by the TRAIN-set signal scale before the genome sees
        # it - the genome's gain/threshold genes were evolved against
        # normalised signals (see fit()). Positive scaling, so the
        # direction the diagnostic measures is unchanged.
        sig = self.raw_signal(recent_returns) / self._signal_scale
        w = self.archive.ensemble_weights(
            sig, top_n=self.cfg.qd_ensemble_top_n, prev_weights=self._prev_weights)
        if not self.cfg.allow_short:
            w = np.clip(w, 0.0, None)
        w = np.clip(w, -self.cfg.max_weight, self.cfg.max_weight)
        self._prev_weights = w
        return w
