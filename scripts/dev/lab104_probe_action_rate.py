"""LAB-104 probe: measure the action-rate penalty's natural magnitude.

Loads the existing LAB-82 vision checkpoint, runs it teacher-forced over a few
val-split episodes of ``data/dataset_vision``, and reports the imitation loss vs
the raw action-rate term (at ``weight_action_rate=1``). The ratio tells us what
``--action-rate-weight`` makes the penalty comparable to the imitation loss, so
the first real training run isn't blindly scaled.

Read-only: loads a checkpoint + corpus, computes losses, prints. Run from kevin/:

    uv run python scripts/dev/lab104_probe_action_rate.py [checkpoint.pt]

Auto-detects the checkpoint's ``use_vision`` (works for F/T-only checkpoints too).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ai_teleop.data import build_dataloaders  # noqa: E402
from ai_teleop.policy import LossConfig, residual_bc_loss  # noqa: E402
from ai_teleop.policy.residual_policy import load_checkpoint  # noqa: E402

DATASET = Path("data/dataset_vision")
DEFAULT_CHECKPOINT = Path("outputs/policy/runs/vision_frozen_lab82/checkpoint.pt")
N_BATCHES = 4


def _step_mask(lengths: torch.Tensor, max_steps: int) -> torch.Tensor:
    indices = torch.arange(max_steps, device=lengths.device)
    return (indices[None, :] < lengths[:, None]).to(torch.float32)


def main() -> int:
    checkpoint = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CHECKPOINT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = load_checkpoint(checkpoint, map_location=device)
    model = loaded.model.to(device).eval()
    use_vision = loaded.config.use_vision
    print(f"checkpoint: {checkpoint}  (use_vision={use_vision})")

    _, val_loader, _ = build_dataloaders(
        DATASET, batch_size=2, val_fraction=0.2, seed=0, load_images=use_vision, num_workers=0
    )

    imitation_total = 0.0
    rate_total = 0.0
    steps_total = 0.0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= N_BATCHES:
                break
            command = batch.command.to(device)
            force_torque = batch.force_torque.to(device)
            proprioception = batch.proprioception.to(device)
            delta = batch.delta.to(device)
            lengths = batch.lengths.to(device)
            mask = _step_mask(lengths, command.shape[1])

            embedding = None
            if use_vision:
                embedding = model.per_step_image_embedding(
                    batch.images.to(device), batch.image_frame_index.to(device)
                )
            predicted, _ = model.forward(
                command, force_torque, proprioception, image_embedding=embedding, hidden=None
            )

            imitation = residual_bc_loss(predicted, delta, mask, config=LossConfig())
            with_rate = residual_bc_loss(
                predicted, delta, mask, config=LossConfig(weight_action_rate=1.0)
            )
            valid = float(mask.sum())
            imitation_total += float(imitation) * valid
            rate_total += float(with_rate - imitation) * valid  # rate term at weight 1
            steps_total += valid

    imitation_mean = imitation_total / steps_total
    rate_mean = rate_total / steps_total
    print(f"imitation loss (mean over {int(steps_total)} steps): {imitation_mean:.6e}")
    print(f"action-rate term @ weight=1:                         {rate_mean:.6e}")
    ratio = imitation_mean / rate_mean
    print(f"ratio imitation/rate:                                {ratio:.3g}")
    print(f"  → weight_action_rate ≈ {ratio:.3g} makes the penalty ~= the imitation loss")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
