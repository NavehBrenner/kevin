"""Smoke-probe the OfflineResidualBCDataset on the small dataset_0 corpus.

Checks: construction, stream shapes, train-only normalization (train means ≈ 0,
std ≈ 1), shared stats across the val split, and a disjoint episode-level split.
Run: uv run python scripts/dev/probe_loader.py
"""

from __future__ import annotations

import torch

from ai_teleop.data.dataset import OfflineResidualBCDataset

DATASET_DIR = "data/dataset_0"


def main() -> None:
    train = OfflineResidualBCDataset(DATASET_DIR, download=False, train=True)
    val = OfflineResidualBCDataset(
        DATASET_DIR, download=False, train=False, norm_stats=train.norm_stats
    )

    print(f"train episodes: {len(train)} | val episodes: {len(val)}")

    sample = train[0]
    print(
        "shapes  command:",
        tuple(sample.command.shape),
        "| force_torque:",
        tuple(sample.force_torque.shape),
        "| proprioception:",
        tuple(sample.proprioception.shape),
        "| delta:",
        tuple(sample.delta.shape),
    )

    # Train-split normalization: pooled per-channel mean ≈ 0, std ≈ 1.
    pooled = torch.cat([episode.command for episode in train.episodes], dim=0)
    print(f"train 'command' pooled mean≈{pooled.mean():+.3f}  std≈{pooled.std():.3f}")

    # Val uses the *train* stats (not its own) — so val's pooled stats need NOT be 0/1.
    print("val shares train stats:", val.norm_stats is train.norm_stats)

    # Episode-level split is disjoint.
    train_idx = {episode.episode_index for episode in train.episodes}
    val_idx = {episode.episode_index for episode in val.episodes}
    print(
        f"split disjoint: {train_idx.isdisjoint(val_idx)} | union covers all: {len(train_idx | val_idx)}"
    )


if __name__ == "__main__":
    main()
