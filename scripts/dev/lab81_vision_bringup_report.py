"""LAB-81 bring-up report — parameter counts + single-step inference timing.

Loads the two bring-up checkpoints (F/T-only and vision-conditioned) trained on
``data/dataset_vision_bringup`` and reports, for the M7 model acceptance:

- total / trainable parameter counts (the vision branch's cost), and
- mean single-step ``model.step`` latency against the ~10 ms (100 Hz) control
  budget — the F/T recurrent path (O(1) per tick) vs the vision path (which also
  encodes one wrist frame).

Timing here is **CPU** (this box has no CUDA); deployment targets GPU, and frames
are decimated (the CNN runs once per *new* frame, not every tick — see
``docs/design/policy-model.md`` latency budget), so these are conservative
upper bounds, not the deployed per-tick cost.

Run from ``kevin/``::

    uv run python scripts/dev/lab81_vision_bringup_report.py
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from ai_teleop.common.log import configure_logging, get_logger
from ai_teleop.policy.residual_policy import load_checkpoint

log = get_logger("lab81")

_BRINGUP = Path("outputs/policy/bringup")
_WARMUP = 20
_ITERS = 200


def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _time_step(model: torch.nn.Module, *, vision: bool) -> float:
    """Mean ms per ``model.step`` over ``_ITERS`` calls (batch 1), after warmup."""
    command = torch.randn(1, 9)
    force_torque = torch.randn(1, 6)
    proprioception = torch.randn(1, 24)
    image = torch.randn(1, 3, 224, 224) if vision else None

    def one_step(hidden: torch.Tensor | None) -> torch.Tensor:
        with torch.no_grad():
            if vision:
                _, hidden = model.step(command, force_torque, proprioception, image, hidden)
            else:
                _, hidden = model.step(command, force_torque, proprioception, hidden=hidden)
        return hidden

    hidden = None
    for _ in range(_WARMUP):
        hidden = one_step(hidden)

    started = time.perf_counter()
    for _ in range(_ITERS):
        hidden = one_step(hidden)
    elapsed = time.perf_counter() - started
    return 1e3 * elapsed / _ITERS


def _report(name: str, checkpoint_dir: Path, *, vision: bool) -> None:
    loaded = load_checkpoint(checkpoint_dir / "checkpoint.pt")
    model = loaded.model
    total, trainable = _count_parameters(model)
    best_val = min(loaded.train_history["val_loss"]) if loaded.train_history else float("nan")
    ms_per_step = _time_step(model, vision=vision)
    log.info(
        "%-12s │ params %8d (%8d trainable) │ best val %.5f │ %.3f ms/step %s",
        name,
        total,
        trainable,
        best_val,
        ms_per_step,
        "(≤10ms budget ✓)" if ms_per_step <= 10.0 else "(> 10ms budget)",
    )


def main() -> int:
    configure_logging()
    torch.manual_seed(0)
    log.info("LAB-81 bring-up report (CPU; batch 1; %d timed iters)", _ITERS)
    _report("ft_only", _BRINGUP / "ft_bringup", vision=False)
    _report("vision", _BRINGUP / "vision_bringup", vision=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
