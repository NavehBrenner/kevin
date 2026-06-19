"""Tests for the BC training loop (LAB-34) — ``scripts/train_policy.train``.

The training core takes prebuilt ``DataLoader``s, so it is exercised on a small
synthetic corpus with a *learnable* signal (each step's Δ is a fixed linear map of
that step's inputs) — no sim and no corpus on disk. The acceptance check is the
spec's "sane curves": both train and validation loss fall over a handful of epochs.
``scripts/`` is not importable as a package, so it is loaded by file path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ai_teleop.data.dataset import Episode, collate_episodes
from ai_teleop.policy import PolicyConfig, save_checkpoint
from ai_teleop.policy.residual_policy import load_checkpoint

_TRAIN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_policy.py"


def _load_train_module():
    spec = importlib.util.spec_from_file_location("train_policy", _TRAIN_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_policy"] = module
    spec.loader.exec_module(module)
    return module


train_policy = _load_train_module()


def _linear_episode(index: int, length: int, weight: torch.Tensor, *, seed: int = 0) -> Episode:
    """An episode whose per-step Δ is a fixed linear function of its inputs."""
    generator = torch.Generator().manual_seed(seed + index)
    command = torch.randn(length, 9, generator=generator)
    force_torque = torch.randn(length, 6, generator=generator)
    proprioception = torch.randn(length, 24, generator=generator)
    inputs = torch.cat([command, force_torque, proprioception], dim=-1)  # (length, 39)
    delta = 0.1 * (inputs @ weight.T)  # deterministic, learnable target
    return Episode(
        episode_index=index,
        command=command,
        force_torque=force_torque,
        proprioception=proprioception,
        delta=delta,
    )


def _loader(weight: torch.Tensor, *, n_episodes: int, base_index: int, seed: int) -> DataLoader:
    episodes = [
        _linear_episode(base_index + i, length=12 + i, weight=weight, seed=seed)
        for i in range(n_episodes)
    ]
    return DataLoader(episodes, batch_size=4, shuffle=True, collate_fn=collate_episodes)


def test_train_drives_train_and_val_loss_down():
    weight = torch.randn(7, 39, generator=torch.Generator().manual_seed(99))
    train_loader = _loader(weight, n_episodes=8, base_index=0, seed=1)
    val_loader = _loader(weight, n_episodes=3, base_index=100, seed=2)

    _, history = train_policy.train(
        train_loader,
        val_loader,
        config=PolicyConfig(hidden_size=32, num_layers=1),
        train_config=train_policy.TrainConfig(epochs=25, learning_rate=1e-2, patience=25),
    )

    assert len(history["train_loss"]) >= 5
    # Sane curves: both losses end well below where they started.
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert history["val_loss"][-1] < history["val_loss"][0]


def test_checkpoint_persists_training_history(tmp_path):
    from ai_teleop.data.dataset import NormStats
    from ai_teleop.policy import ResidualPolicy

    config = PolicyConfig(hidden_size=8, num_layers=1)
    stats = NormStats(
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
    history = {"train_loss": [1.0, 0.5], "val_loss": [1.1, 0.6]}
    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path, model=ResidualPolicy(config), config=config, norm_stats=stats, train_history=history
    )

    loaded = load_checkpoint(path)
    assert loaded.train_history == history
    assert loaded.policy_checkpoint_version != "unknown"
