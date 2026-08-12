"""
Scenario Library – manages, stores, and retrieves synthetic market scenarios.
Provides stratified sampling for training (ensuring coverage of all severity levels).
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class ScenarioDataset(Dataset):
    """PyTorch Dataset wrapping the black-swan scenario library."""

    def __init__(
        self,
        scenarios: List[Dict],
        seq_len: int,
        n_assets: int,
        normalize: bool = True,
    ):
        self.scenarios = scenarios
        self.seq_len = seq_len
        self.n_assets = n_assets
        self.normalize = normalize
        self._precompute_stats()

    def _precompute_stats(self) -> None:
        all_returns = np.stack([s["return_path"] for s in self.scenarios])
        self.global_mean = float(all_returns.mean())
        self.global_std = float(all_returns.std()) + 1e-8

    def __len__(self) -> int:
        return len(self.scenarios)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.scenarios[idx]
        path = s["return_path"][:self.seq_len, :self.n_assets]

        # Pad if shorter
        if path.shape[0] < self.seq_len:
            pad = np.zeros((self.seq_len - path.shape[0], self.n_assets))
            path = np.vstack([path, pad])

        if self.normalize:
            path = (path - self.global_mean) / self.global_std

        return {
            "returns": torch.tensor(path, dtype=torch.float32),
            "severity": torch.tensor(s["severity_score"], dtype=torch.float32),
            "max_drawdown": torch.tensor(s["max_drawdown"], dtype=torch.float32),
            "sigma_multiplier": torch.tensor(s["sigma_multiplier"], dtype=torch.float32),
        }


class ScenarioLibrary:
    """
    Persistent storage and retrieval manager for synthetic scenarios.

    Supports:
      - Save/load to disk
      - Stratified sampling by severity
      - Curriculum training (easy → hard progression)
      - Scenario augmentation via time-warping and amplitude scaling
    """

    def __init__(self, storage_dir: str = "data/scenarios"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.scenarios: List[Dict] = []

    def add_scenarios(self, scenarios: List[Dict]) -> None:
        self.scenarios.extend(scenarios)
        logger.info("Library now contains %d scenarios", len(self.scenarios))

    def save(self, name: str = "black_swan_library") -> str:
        path = self.storage_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self.scenarios, f)
        # Save summary JSON alongside
        summary = {
            "n_scenarios": len(self.scenarios),
            "templates": list(set(s["template"] for s in self.scenarios)),
        }
        with open(self.storage_dir / f"{name}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved %d scenarios to %s", len(self.scenarios), path)
        return str(path)

    def load(self, name: str = "black_swan_library") -> None:
        path = self.storage_dir / f"{name}.pkl"
        with open(path, "rb") as f:
            self.scenarios = pickle.load(f)
        logger.info("Loaded %d scenarios from %s", len(self.scenarios), path)

    def get_curriculum_loader(
        self,
        n_assets: int,
        seq_len: int,
        batch_size: int = 32,
        stage: str = "easy",  # "easy" | "medium" | "hard" | "all"
    ) -> DataLoader:
        """
        Returns a DataLoader filtered by difficulty stage.
        Curriculum: train on easy first, gradually introduce doomsday scenarios.
        """
        severities = np.array([s["severity_score"] for s in self.scenarios])
        p33, p66 = np.percentile(severities, 33), np.percentile(severities, 66)

        if stage == "easy":
            filtered = [s for s in self.scenarios if s["severity_score"] <= p33]
        elif stage == "medium":
            filtered = [s for s in self.scenarios if p33 < s["severity_score"] <= p66]
        elif stage == "hard":
            filtered = [s for s in self.scenarios if s["severity_score"] > p66]
        else:
            filtered = self.scenarios

        dataset = ScenarioDataset(filtered, seq_len, n_assets)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    def stratified_sample(self, n: int, n_strata: int = 5) -> List[Dict]:
        """
        Stratified sample ensuring all severity levels are represented.
        Critical for training: model must see both mild and catastrophic events.
        """
        severities = np.array([s["severity_score"] for s in self.scenarios])
        strata_bounds = np.percentile(severities, np.linspace(0, 100, n_strata + 1))
        n_per_stratum = n // n_strata
        sampled = []

        for i in range(n_strata):
            lo, hi = strata_bounds[i], strata_bounds[i + 1]
            stratum = [s for s in self.scenarios
                       if lo <= s["severity_score"] <= hi]
            if stratum:
                k = min(n_per_stratum, len(stratum))
                idx = np.random.choice(len(stratum), k, replace=False)
                sampled.extend([stratum[j] for j in idx])

        return sampled

    def augment(self, scenario: Dict, n_augments: int = 5) -> List[Dict]:
        """
        Data augmentation via:
          - Time-warping: stretch/compress temporal dimension
          - Amplitude scaling: ±20% shock magnitude
          - Phase shift: roll the sequence in time
        """
        augmented = []
        path = scenario["return_path"]

        for i in range(n_augments):
            scale = np.random.uniform(0.8, 1.2)
            roll = np.random.randint(0, max(path.shape[0] // 4, 1))
            aug_path = np.roll(path * scale, roll, axis=0)
            new_s = dict(scenario)
            new_s["return_path"] = aug_path
            new_s["id"] = f"{scenario['id']}_aug{i}"
            new_s["severity_score"] = scenario["severity_score"] * scale
            augmented.append(new_s)

        return augmented
