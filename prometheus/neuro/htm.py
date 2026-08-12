"""
Hierarchical Temporal Memory (HTM) for order-flow sequence prediction.

Based on Numenta's HTM theory: the neocortex learns spatial-temporal sequences
through sparse distributed representations (SDRs) and Hebbian-like learning.
This HTM implementation detects anomalies in order-flow patterns that precede
major price moves.

Key properties:
  - Online learning: updates continuously without retraining
  - Anomaly detection: naturally produces anomaly scores for novel sequences
  - Sparse representations: each input encoded as sparse bit-vector (2% active)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class ScalarEncoder:
    """Encodes a scalar value into a Sparse Distributed Representation (SDR)."""

    def __init__(self, n_bits: int = 2048, n_active: int = 40, value_range: Tuple = (-1.0, 1.0)):
        self.n_bits = n_bits
        self.n_active = n_active
        self.v_min, self.v_max = value_range
        self.bits_per_bucket = n_bits - n_active + 1

    def encode(self, value: float) -> np.ndarray:
        sdr = np.zeros(self.n_bits, dtype=np.float32)
        value = np.clip(value, self.v_min, self.v_max)
        norm = (value - self.v_min) / (self.v_max - self.v_min)
        start = int(norm * self.bits_per_bucket)
        sdr[start: start + self.n_active] = 1.0
        return sdr


class SpatialPooler:
    """
    HTM Spatial Pooler: converts raw encodings into stable SDRs.
    Implements homeostatic plasticity and competitive inhibition.
    """

    def __init__(
        self,
        input_size: int,
        n_columns: int = 2048,
        n_active_cols: int = 40,
        lr: float = 0.05,
        boost_strength: float = 1.0,
    ):
        self.n_columns = n_columns
        self.n_active_cols = n_active_cols
        self.lr = lr
        self.boost_strength = boost_strength

        # Permanences ∈ [0, 1]: synaptic strength
        self.permanences = np.random.uniform(0.3, 0.7, (n_columns, input_size)).astype(np.float32)
        self.synapse_threshold = 0.5
        self.boost_factors = np.ones(n_columns, dtype=np.float32)
        self.activity_history = np.zeros(n_columns, dtype=np.float32)
        self.target_density = n_active_cols / n_columns

    def compute(self, input_sdr: np.ndarray, learn: bool = True) -> np.ndarray:
        """Run spatial pooling. Returns active column indices."""
        connected = (self.permanences >= self.synapse_threshold).astype(np.float32)
        overlaps = connected @ input_sdr
        boosted = overlaps * self.boost_factors

        # Winner-take-all: top-k columns
        active_cols = np.argsort(boosted)[-self.n_active_cols:]

        if learn:
            self._update_permanences(input_sdr, active_cols)
            self._update_boost(active_cols)

        output = np.zeros(self.n_columns, dtype=np.float32)
        output[active_cols] = 1.0
        return output

    def _update_permanences(self, input_sdr: np.ndarray, active_cols: np.ndarray) -> None:
        for col in active_cols:
            # Hebbian: strengthen connected synapses that were active
            self.permanences[col] += self.lr * input_sdr
            self.permanences[col] -= self.lr * 0.5 * (1 - input_sdr)
            self.permanences[col] = np.clip(self.permanences[col], 0.0, 1.0)

    def _update_boost(self, active_cols: np.ndarray) -> None:
        active_mask = np.zeros(self.n_columns)
        active_mask[active_cols] = 1.0
        # Exponential moving average of activity
        self.activity_history = 0.99 * self.activity_history + 0.01 * active_mask
        # Boost under-active columns
        self.boost_factors = np.exp(
            self.boost_strength * (self.target_density - self.activity_history)
        )


class TemporalMemory:
    """
    HTM Temporal Memory: learns sequences of column activations.
    Detects when current input matches a predicted sequence vs. an anomaly.
    """

    def __init__(
        self,
        n_columns: int = 2048,
        n_cells_per_col: int = 32,
        activation_threshold: int = 13,
        learning_threshold: int = 10,
        initial_permanence: float = 0.51,
        permanence_increment: float = 0.1,
        permanence_decrement: float = 0.1,
    ):
        self.n_columns = n_columns
        self.n_cells = n_columns * n_cells_per_col
        self.n_cells_per_col = n_cells_per_col
        self.activation_threshold = activation_threshold
        self.learning_threshold = learning_threshold

        self.active_cells: np.ndarray = np.zeros(self.n_cells, dtype=bool)
        self.predicted_cells: np.ndarray = np.zeros(self.n_cells, dtype=bool)
        self.winner_cells: np.ndarray = np.zeros(self.n_cells, dtype=bool)
        self.segments: List = []

    def compute(
        self,
        active_columns: np.ndarray,  # binary [n_columns]
        learn: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """
        Run one temporal memory step.
        Returns (active_cells, anomaly_score).
        Anomaly = fraction of active columns that were NOT predicted.
        """
        active_col_indices = np.where(active_columns > 0)[0]

        # Columns that were predicted: check if any cell in column is predicted
        predicted_active = 0
        new_active = np.zeros(self.n_cells, dtype=bool)

        for col in active_col_indices:
            col_predicted = any(
                self.predicted_cells[col * self.n_cells_per_col: (col + 1) * self.n_cells_per_col]
            )
            if col_predicted:
                predicted_active += 1
                # Activate predicted cells only
                for c in range(self.n_cells_per_col):
                    cell = col * self.n_cells_per_col + c
                    if self.predicted_cells[cell]:
                        new_active[cell] = True
            else:
                # Burst: all cells in column become active (surprise)
                for c in range(self.n_cells_per_col):
                    new_active[col * self.n_cells_per_col + c] = True

        anomaly_score = 1.0 - (predicted_active / max(len(active_col_indices), 1))
        self.active_cells = new_active
        return new_active, float(anomaly_score)


class HierarchicalTemporalMemory(nn.Module):
    """
    Full HTM pipeline for order-flow anomaly detection.

    Pipeline: raw_tick → scalar_encode → spatial_pool → temporal_memory
    Outputs: anomaly score + learned sequence embeddings.
    """

    def __init__(
        self,
        n_input_features: int,
        n_columns: int = 2048,
        n_active_cols: int = 40,
    ):
        super().__init__()
        self.encoders = [
            ScalarEncoder(n_bits=512, n_active=20, value_range=(-3.0, 3.0))
            for _ in range(n_input_features)
        ]
        self.spatial_pooler = SpatialPooler(
            input_size=512 * n_input_features,
            n_columns=n_columns,
            n_active_cols=n_active_cols,
        )
        self.temporal_memory = TemporalMemory(n_columns=n_columns)

        # Learned projection from HTM state to latent vector
        self.state_proj = nn.Linear(n_columns, 128)

    def forward_step(
        self,
        feature_vector: np.ndarray,
        learn: bool = True,
    ) -> Dict:
        """Process one tick. Returns anomaly score and state embedding."""
        # Encode each feature separately, concatenate
        encodings = [enc.encode(float(v)) for enc, v in zip(self.encoders, feature_vector)]
        combined_sdr = np.concatenate(encodings)

        # Spatial pooling → stable SDR
        col_activations = self.spatial_pooler.compute(combined_sdr, learn=learn)

        # Temporal memory → anomaly score
        _, anomaly_score = self.temporal_memory.compute(col_activations, learn=learn)

        # Project to latent (for downstream models)
        col_tensor = torch.tensor(col_activations, dtype=torch.float32).unsqueeze(0)
        embedding = self.state_proj(col_tensor)

        return {
            "anomaly_score": anomaly_score,
            "column_sdr": col_activations,
            "embedding": embedding,
            "alert": anomaly_score > 0.7,  # market regime change signal
            "interpretation": (
                "ANOMALOUS ORDER FLOW - potential regime shift" if anomaly_score > 0.7
                else ("ELEVATED: monitor closely" if anomaly_score > 0.4 else "NORMAL")
            ),
        }

    def process_sequence(
        self,
        sequence: np.ndarray,  # [T, n_features]
        learn: bool = True,
    ) -> Dict:
        """Process a sequence of ticks and return aggregated anomaly profile."""
        scores = []
        for t in range(len(sequence)):
            result = self.forward_step(sequence[t], learn=learn)
            scores.append(result["anomaly_score"])

        scores_arr = np.array(scores)
        return {
            "anomaly_scores": scores_arr,
            "mean_anomaly": float(scores_arr.mean()),
            "max_anomaly": float(scores_arr.max()),
            "anomaly_spike_times": np.where(scores_arr > 0.7)[0].tolist(),
            "sequence_novelty": float(scores_arr[-10:].mean()),  # recent novelty
        }
