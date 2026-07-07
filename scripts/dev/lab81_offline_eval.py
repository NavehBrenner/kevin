"""LAB-81 offline held-out eval — trained policy vs expert on unseen episodes.

Closed-loop vision deployment isn't wired yet (``Observation`` carries no wrist
image — that's LAB-83), and the ``data/recorded/`` corpus has no rendered frames,
so the vision policy can't be run in the sim loop or on those recordings. What we
*can* test now is the honest offline metric: run each trained checkpoint over the
**held-out val split** (episodes excluded from training) and measure how well its
per-step Δ reproduces the expert's logged Δ.

This is teacher-forced / open-loop (the GRU is fed the logged observations, not its
own rollout), so it measures BC accuracy, not closed-loop success — but on genuinely
unseen episodes. Reported per interpretable channel (position mm, orientation deg,
grip N), against a zero-Δ baseline so "did it learn anything" is unambiguous.

Run from ``kevin/``::

    uv run python scripts/dev/lab81_offline_eval.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import Tensor

from ai_teleop.common.log import configure_logging, get_logger
from ai_teleop.data import build_dataloaders
from ai_teleop.policy.losses import _GRIP, _ORI, _POS, geodesic_angle
from ai_teleop.policy.residual_policy import load_checkpoint

log = get_logger("lab81eval")

_DATASET = Path("data/dataset_vision_bringup")
_BRINGUP = Path("outputs/policy/bringup")


def _channel_errors(predicted: Tensor, target: Tensor, mask: Tensor) -> dict[str, float]:
    """Masked mean per-channel error: position (mm), orientation (deg), grip (N)."""
    valid = mask.bool()
    position_mm = (
        1e3
        * torch.linalg.norm((predicted[..., _POS] - target[..., _POS])[valid], dim=-1).mean().item()
    )
    orientation_deg = (
        torch.rad2deg(geodesic_angle(predicted[..., _ORI], target[..., _ORI])[valid]).mean().item()
    )
    grip_n = (predicted[..., _GRIP] - target[..., _GRIP])[valid].abs().mean().item()
    return {"pos_mm": position_mm, "ori_deg": orientation_deg, "grip_N": grip_n}


def _evaluate(
    dataset_dir: Path, checkpoint_dir: Path, *, vision: bool
) -> tuple[dict[str, float], dict[str, float]]:
    """Run a checkpoint over the held-out val split; return (policy_err, zero_baseline_err)."""
    loaded = load_checkpoint(checkpoint_dir / "checkpoint.pt")
    model = loaded.model.eval()

    # Same split (seed 0, val_fraction 0.2) the run trained under → the true held-out
    # episodes; load_images matches the checkpoint's modality.
    _, val_loader, _ = build_dataloaders(
        dataset_dir, batch_size=4, val_fraction=0.2, seed=0, load_images=vision, download=False
    )

    predicted_all, target_all, mask_all = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            max_steps = batch.command.shape[1]
            steps = torch.arange(max_steps)
            mask = (steps[None, :] < batch.lengths[:, None]).float()  # (B, T)
            predicted, _ = model.forward(
                batch.command,
                batch.force_torque,
                batch.proprioception,
                images=batch.images if vision else None,
                image_frame_index=batch.image_frame_index if vision else None,
                lengths=batch.lengths,
            )
            predicted_all.append(predicted)
            target_all.append(batch.delta)
            mask_all.append(mask)

    predicted = torch.cat(predicted_all)
    target = torch.cat(target_all)
    mask = torch.cat(mask_all)
    policy = _channel_errors(predicted, target, mask)
    zero_baseline = _channel_errors(torch.zeros_like(target), target, mask)
    return policy, zero_baseline


def _report(name: str, dataset_dir: Path, checkpoint_dir: Path, *, vision: bool) -> None:
    policy, zero = _evaluate(dataset_dir, checkpoint_dir, vision=vision)
    log.info(
        "%-9s │ pos %.2f mm (zeroΔ %.2f) │ ori %.2f° (zeroΔ %.2f) │ grip %.3f N (zeroΔ %.3f)",
        name,
        policy["pos_mm"],
        zero["pos_mm"],
        policy["ori_deg"],
        zero["ori_deg"],
        policy["grip_N"],
        zero["grip_N"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=_DATASET, help="M4 dataset dir.")
    parser.add_argument("--ft-run", type=Path, default=_BRINGUP / "ft_bringup", help="F/T run dir.")
    parser.add_argument(
        "--vision-run", type=Path, default=_BRINGUP / "vision_bringup", help="Vision run dir."
    )
    args = parser.parse_args()

    configure_logging()
    log.info(
        "LAB-81 offline held-out eval on %s val split (policy Δ̂ vs expert Δ*)", args.dataset.name
    )
    _report("ft_only", args.dataset, args.ft_run, vision=False)
    _report("vision", args.dataset, args.vision_run, vision=True)
    log.info("(zeroΔ = error of predicting no correction — the far-field prior to beat)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
