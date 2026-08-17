"""
Tests for kronos/runpod_trigger.py - loading RunPod-trained checkpoints
into Kronos's live trading path.

Pod orchestration (create/train/pull/delete) no longer lives in this
module or runs from Hetzner at all - it's entirely owned by
.github/workflows/train-runpod.yml on a schedule, which scp's the result
onto this box. What's left here is just load_runpod_checkpoint(), tested
against real torch state dicts (no network, no subprocess - there's
nothing left to mock).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from kronos import runpod_trigger
from kronos.runpod_trigger import load_runpod_checkpoint


class TestLoadRunpodCheckpoint:
    def test_missing_file_returns_false_and_leaves_model_unchanged(self, tmp_path):
        model = nn.Linear(4, 4)
        before = model.weight.clone()
        loaded = load_runpod_checkpoint(model, checkpoint_dir=tmp_path)
        assert loaded is False
        assert torch.equal(model.weight, before)

    def test_valid_checkpoint_loads_in_place(self, tmp_path):
        source = nn.Linear(4, 4)
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        torch.save(source.state_dict(), meta_dir / "snn.pt")

        target = nn.Linear(4, 4)
        loaded = load_runpod_checkpoint(target, checkpoint_dir=tmp_path)
        assert loaded is True
        assert torch.equal(target.weight, source.weight)

    def test_shape_mismatch_returns_false_not_raise(self, tmp_path):
        """Regression for the real PrometheusEngine-vs-ReflexArc SNN
        output_size mismatch (n_assets//2 vs n_assets) - must degrade to
        'no checkpoint', never crash the caller."""
        source = nn.Linear(4, 4)
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        torch.save(source.state_dict(), meta_dir / "snn.pt")

        target = nn.Linear(4, 2)   # incompatible shape
        before = target.weight.clone()
        loaded = load_runpod_checkpoint(target, checkpoint_dir=tmp_path)
        assert loaded is False
        assert torch.equal(target.weight, before)

    def test_corrupt_file_returns_false_not_raise(self, tmp_path):
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "snn.pt").write_bytes(b"not a torch checkpoint at all")

        target = nn.Linear(4, 4)
        before = target.weight.clone()
        loaded = load_runpod_checkpoint(target, checkpoint_dir=tmp_path)
        assert loaded is False
        assert torch.equal(target.weight, before)

    def test_default_checkpoint_dir_is_checkpoints_runpod(self):
        assert runpod_trigger.CHECKPOINT_DIR == Path("checkpoints/runpod")
