"""Tests for per-run training artifacts (LAB-34) — ``policy.run_artifacts``.

A run folder must end up with the four files (checkpoint, metadata, history json +
png), the checkpoint must reload into a deployable ``LearnedResidual``, and the
metadata must carry the params/results needed to reconstruct and compare runs.
"""

from __future__ import annotations

import json

import torch

from ai_teleop.data.dataset import NormStats
from ai_teleop.policy import (
    LearnedResidual,
    LossConfig,
    PolicyConfig,
    ResidualPolicy,
    TrainConfig,
    build_metadata,
    write_run_artifacts,
)
from ai_teleop.policy.run_artifacts import (
    CHECKPOINT_NAME,
    HISTORY_NAME,
    HISTORY_PLOT_NAME,
    METADATA_NAME,
    summarize_history,
)


def _stats() -> NormStats:
    return NormStats(
        mean={
            "command": torch.zeros(9),
            "force_torque": torch.zeros(6),
            "proprioception": torch.zeros(24),
        },
        std={
            "command": torch.ones(9),
            "force_torque": torch.ones(6),
            "proprioception": torch.ones(24),
        },
    )


_HISTORY = {"train_loss": [1.0, 0.4, 0.25], "val_loss": [1.2, 0.5, 0.3]}


def test_summarize_history_picks_the_minimum():
    summary = summarize_history(_HISTORY)
    assert summary["epochs_run"] == 3
    assert summary["best_val_loss"] == 0.3
    assert summary["best_epoch"] == 2
    assert summary["final_train_loss"] == 0.25


def test_build_metadata_carries_configs_and_results():
    metadata = build_metadata(
        config=PolicyConfig(hidden_size=64),
        loss_config=LossConfig(),
        train_config=TrainConfig(epochs=5),
        history=_HISTORY,
        dataset={"dir": "data/dataset_1", "n_train_episodes": 8},
        extra={"run_name": "demo", "device": "cpu"},
    )
    assert metadata["model_config"]["hidden_size"] == 64
    assert metadata["train_config"]["epochs"] == 5
    assert metadata["dataset"]["n_train_episodes"] == 8
    assert metadata["results"]["best_val_loss"] == 0.3
    assert metadata["run_name"] == "demo"


def test_write_run_artifacts_produces_loadable_run(tmp_path):
    config = PolicyConfig(hidden_size=16, num_layers=1)
    torch.manual_seed(0)
    model = ResidualPolicy(config).eval()
    stats = _stats()
    metadata = build_metadata(
        config=config,
        loss_config=LossConfig(),
        train_config=TrainConfig(),
        history=_HISTORY,
        dataset={"dir": "data/dataset_1"},
        extra={"run_name": "run0"},
    )

    run_dir = write_run_artifacts(
        tmp_path / "run0",
        model=model,
        config=config,
        norm_stats=stats,
        loss_config=LossConfig(),
        history=_HISTORY,
        metadata=metadata,
    )

    # All four artifacts exist and the plot is non-empty.
    for name in (CHECKPOINT_NAME, METADATA_NAME, HISTORY_NAME, HISTORY_PLOT_NAME):
        assert (run_dir / name).exists(), name
    assert (run_dir / HISTORY_PLOT_NAME).stat().st_size > 0

    # metadata + history round-trip as JSON.
    on_disk = json.loads((run_dir / METADATA_NAME).read_text(encoding="utf-8"))
    assert on_disk["results"]["best_val_loss"] == 0.3
    assert json.loads((run_dir / HISTORY_NAME).read_text(encoding="utf-8")) == _HISTORY

    # The checkpoint reloads into a deployable provider.
    provider = LearnedResidual.from_checkpoint(run_dir / CHECKPOINT_NAME)
    assert provider is not None
