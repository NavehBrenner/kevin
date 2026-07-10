"""LAB-106: WHERE does the trained GRU's position error live — near or far field?

Normalization already exists (inputs are z-scored end-to-end; targets are left
raw). Yet the GRU scores ~7.6 mm held-out vs a linear probe's 2.36 mm. This splits
the trained model's teacher-forced position error by distance-to-hole, so we can
tell a *gating* failure (emits junk where the expert target is 0, far field) from a
*correction* failure (near field), and compare to the zero-Δ baseline per bucket.

Reuses the exact val split + feature assembly + normalization the checkpoint trained
under. Run from kevin/:  uv run python scripts/dev/lab106_error_decomp.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ai_teleop.common.log import configure_logging, get_logger
from ai_teleop.data.dataset import (
    extract_training_episode,
    normalize_episode,
    split_episodes,
)
from ai_teleop.data.trajectory import load_episode
from ai_teleop.policy.residual_policy import load_checkpoint

log = get_logger("lab106decomp")

DATASET = Path("data/dataset_vision")
D_FAR = 0.15
RUNS = ["ftonly_baseline_lab82", "ftonly_wpos10_wd", "ftonly_gate_wpos10_wd"]


def _val_summaries(seed: int = 0, val_fraction: float = 0.2):
    meta = json.loads((DATASET / "metadata.json").read_text())
    _, val = split_episodes(meta["episodes"], val_fraction=val_fraction, seed=seed)
    return val


def _report_run(run: str, val_summaries) -> None:
    loaded = load_checkpoint(Path("outputs/policy/runs") / run / "checkpoint.pt")
    model = loaded.model.eval()

    # buckets: (label, mask_fn) → accumulate error and |pred|, |target|.
    buckets = {"far d>=0.15": [], "near 0.05-0.15": [], "close d<0.05": []}
    pred_mag = {k: [] for k in buckets}
    zero_err = {k: [] for k in buckets}

    for summary in val_summaries:
        raw = load_episode(DATASET / summary["file"])
        columns, _ = raw
        episode = normalize_episode(
            extract_training_episode(raw, command_ee_delta=loaded.config.use_command_ee_delta),
            loaded.norm_stats,
        )
        with torch.no_grad():
            predicted, _ = model.forward(
                episode.command.unsqueeze(0),
                episode.force_torque.unsqueeze(0),
                episode.proprioception.unsqueeze(0),
                lengths=torch.tensor([episode.command.shape[0]]),
            )
        pred_pos = predicted[0, :, :3].numpy()  # raw-space delta (model trained on raw Δ)
        tgt_pos = columns["delta_position"]
        dist = columns["distance"]
        err = np.linalg.norm(pred_pos - tgt_pos, axis=1)
        pmag = np.linalg.norm(pred_pos, axis=1)
        tmag = np.linalg.norm(tgt_pos, axis=1)

        masks = {
            "far d>=0.15": dist >= D_FAR,
            "near 0.05-0.15": (dist >= 0.05) & (dist < D_FAR),
            "close d<0.05": dist < 0.05,
        }
        for k, m in masks.items():
            if m.any():
                buckets[k].append(err[m])
                pred_mag[k].append(pmag[m])
                zero_err[k].append(tmag[m])  # zero-Δ error == |target|

    log.info("── %s  (command_ee_delta=%s) ──", run, loaded.config.use_command_ee_delta)
    all_e, all_z = [], []
    for k in buckets:
        e = np.concatenate(buckets[k])
        p = np.concatenate(pred_mag[k])
        z = np.concatenate(zero_err[k])
        all_e.append(e)
        all_z.append(z)
        log.info(
            "  %-16s n=%7d │ GRU |err| %5.2f mm │ zero-Δ %5.2f mm │ GRU |pred| %5.2f mm",
            k,
            len(e),
            1e3 * e.mean(),
            1e3 * z.mean(),
            1e3 * p.mean(),
        )
    e, z = np.concatenate(all_e), np.concatenate(all_z)
    log.info(
        "  %-16s n=%7d │ GRU |err| %5.2f mm │ zero-Δ %5.2f mm  ← offline pos",
        "ALL steps",
        len(e),
        1e3 * e.mean(),
        1e3 * z.mean(),
    )


def main() -> int:
    configure_logging()
    val = _val_summaries()
    log.info("val episodes: %d (seed 0, val_fraction 0.2)", len(val))
    for run in RUNS:
        _report_run(run, val)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
