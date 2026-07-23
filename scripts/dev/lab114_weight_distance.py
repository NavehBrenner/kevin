"""LAB-114 H-C: is CPU-vs-GPU a *different model*, or the same model at float noise?

The 2026-07-07 headline was CPU-trained; every retrain has been on GPU. H-C asks whether
that alone can move closed-loop success. Training the same seed on both devices gives the
same epoch count and the same `best_val_loss` to six decimals but a different checkpoint
hash — so the question is one of *scale*: how far apart does the device put the weights,
compared with how far apart two training **seeds** put them (a distance already known to be
worth up to 18 pp of closed-loop success)?

Read-only. Run: `uv run python scripts/dev/lab114_weight_distance.py`
"""

from __future__ import annotations

from pathlib import Path

import torch

RUNS = Path("outputs/policy/runs")


def weights(name: str) -> dict[str, torch.Tensor]:
    payload = torch.load(RUNS / name / "checkpoint.pt", map_location="cpu", weights_only=False)
    return payload["model_state_dict"]


def distance(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> tuple[float, float]:
    """(max |Δ| over all parameters, ‖Δ‖₂ / ‖left‖₂) — an absolute and a relative view."""
    max_abs = max(float((left[k] - right[k]).abs().max()) for k in left)
    delta = torch.cat([(left[k] - right[k]).flatten() for k in left])
    base = torch.cat([left[k].flatten() for k in left])
    return max_abs, float(delta.norm() / base.norm())


def main() -> None:
    gpu0, cpu0, gpu1 = weights("lab114_seed0"), weights("lab114_cpu_seed0"), weights("lab114_seed1")

    for label, other in (("device (CPU vs GPU, seed 0)", cpu0), ("seed (0 vs 1, both GPU)", gpu1)):
        max_abs, relative = distance(gpu0, other)
        print(f"{label:<32} max|Δw| = {max_abs:.3e}   ‖Δw‖/‖w‖ = {relative:.3e}")


if __name__ == "__main__":
    main()
