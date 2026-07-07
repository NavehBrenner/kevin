"""LAB-81 — run the trained vision policy on the (re-rendered) recorded corpus.

`data/recorded/` is 54 **real human vision-teleop** episodes recorded `noassist`
(no correction applied → logged Δ ≈ 0, so there is no BC target to score against).
After rendering wrist frames for them (replay-render, `run_episode.py --record
images`), this script feeds each episode's **actual rendered frames** + F/T +
proprioception + command to the trained policies and reports the **correction the
policy would inject** per channel — the first look at the vision policy consuming
real vision input.

It is **open-loop** (the policy sees the recorded human states, does not drive the
arm — closed-loop vision deploy is unbuilt, LAB-83) and the checkpoint is the weak
50-episode synthetic bring-up model, so this is a **pipeline / behavior** check, not
a performance claim. `data/recorded/` has no `metadata.json`, so episodes are loaded
directly (not via the dataset loader).

Run from ``kevin/`` after rendering the recorded frames::

    uv run python scripts/dev/lab81_recorded_vision_eval.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ai_teleop.common.log import configure_logging, get_logger
from ai_teleop.data.dataset import Episode, extract_training_episode, normalize_episode
from ai_teleop.data.trajectory import episode_imgs_dir, load_episode
from ai_teleop.policy.residual_policy import LoadedCheckpoint, load_checkpoint

log = get_logger("lab81rec")

_RECORDED = Path("data/recorded")
_RUNS = _RECORDED / "runs"
_BRINGUP = Path("outputs/policy/bringup")

# Deployment clamp bounds (docs/design/policy-model.md) — for context on saturation.
_CLAMP_POS_M = 0.03
_CLAMP_ORI_DEG = 10.0


def _episode_indices() -> list[int]:
    indices = []
    for path in sorted(_RUNS.glob("episode_*")):
        match = re.fullmatch(r"episode_(\d+)", path.name)
        if match and (path / "imgs").is_dir() and any((path / "imgs").glob("step_*.jpg")):
            indices.append(int(match.group(1)))
    return indices


def _load_recorded_episode(index: int, stats) -> Episode:
    """Load one recorded episode (npz + rendered frames) and normalize with train stats."""
    episode_dir = _RUNS / f"episode_{index:05d}"
    columns, metadata = load_episode(episode_dir / "episode.npz")
    # The recorded corpus's per-episode metadata omits `episode_index` (a different
    # schema than the M4 generator); extract_training_episode only needs it as a label.
    metadata = {**metadata, "episode_index": index}
    raw = extract_training_episode((columns, metadata), imgs_dir=episode_imgs_dir(_RUNS, index))
    return normalize_episode(raw, stats)


def _predict(model: torch.nn.Module, episode: Episode, *, vision: bool) -> Tensor:
    """Forward one episode (batch 1) → predicted per-step Δ ``(T, 7)`` (raw, unclamped)."""
    with torch.no_grad():
        predicted, _ = model.forward(
            episode.command[None],
            episode.force_torque[None],
            episode.proprioception[None],
            images=episode.images[None] if vision else None,
            image_frame_index=episode.image_frame_index[None] if vision else None,
        )
    return predicted[0]


def _summarize(deltas: list[Tensor]) -> dict[str, float]:
    """Aggregate per-step correction magnitudes across all episodes."""
    stacked = torch.cat(deltas)  # (sum_T, 7)
    pos_mm = 1e3 * torch.linalg.norm(stacked[:, 0:3], dim=-1)
    ori_deg = torch.rad2deg(torch.linalg.norm(stacked[:, 3:6], dim=-1))
    grip_n = stacked[:, 6].abs()
    clamp_hit = (pos_mm > 1e3 * _CLAMP_POS_M).float().mean().item()
    return {
        "pos_mm_mean": pos_mm.mean().item(),
        "pos_mm_p90": torch.quantile(pos_mm, 0.9).item(),
        "ori_deg_mean": ori_deg.mean().item(),
        "grip_n_mean": grip_n.mean().item(),
        "pos_clamp_frac": clamp_hit,
    }


def _run(name: str, loaded: LoadedCheckpoint, indices: list[int], *, vision: bool) -> None:
    deltas = [
        _predict(loaded.model.eval(), _load_recorded_episode(i, loaded.norm_stats), vision=vision)
        for i in indices
    ]
    summary = _summarize(deltas)
    log.info(
        "%-9s │ Δpos %.2f mm (p90 %.2f, clamp>%.0fmm %.1f%%) │ Δori %.2f° │ Δgrip %.3f N",
        name,
        summary["pos_mm_mean"],
        summary["pos_mm_p90"],
        1e3 * _CLAMP_POS_M,
        1e2 * summary["pos_clamp_frac"],
        summary["ori_deg_mean"],
        summary["grip_n_mean"],
    )


def _logged_delta_magnitude(indices: list[int]) -> float:
    """Mean |Δpos| the corpus itself logged (noassist ⇒ ≈0, the honest baseline)."""
    total, count = 0.0, 0
    for index in indices:
        columns, _ = load_episode(_RUNS / f"episode_{index:05d}" / "episode.npz")
        magnitude = np.linalg.norm(columns["delta_position"], axis=-1)
        total += float(magnitude.sum())
        count += magnitude.shape[0]
    return 1e3 * total / max(count, 1)


def main() -> int:
    configure_logging()
    indices = _episode_indices()
    if not indices:
        log.error("no rendered recorded episodes under %s — render frames first", _RUNS)
        return 2
    log.info("running trained policies on %d rendered recorded episodes (open-loop)", len(indices))
    log.info(
        "corpus logged Δpos (noassist baseline): %.3f mm/step", _logged_delta_magnitude(indices)
    )
    _run(
        "ft_only", load_checkpoint(_BRINGUP / "ft_bringup" / "checkpoint.pt"), indices, vision=False
    )
    _run(
        "vision",
        load_checkpoint(_BRINGUP / "vision_bringup" / "checkpoint.pt"),
        indices,
        vision=True,
    )
    log.info(
        "open-loop over real human-teleop states; weak bring-up checkpoint — behavior check, not a KPI"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
