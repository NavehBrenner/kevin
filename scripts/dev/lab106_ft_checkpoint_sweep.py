"""LAB-106: does the action-rate penalty explain F/T "worse than zero"?

The delta-target audit showed a *linear* probe on F/T observables gets ~2.4 mm
held-out (beating the 4.9 mm zero-Δ baseline), while the trained F/T policy was
reported at 7.74 mm (worse than zero). That gap is a fitting failure, not an
unlearnable target. Prime suspect: the LAB-104 action-rate (smoothness) penalty.

This runs the held-out offline eval (reusing lab81's evaluator) across the F/T
checkpoints that differ only in that penalty, and prints each one's configured
weight. Run from kevin/:  uv run python scripts/dev/lab106_ft_checkpoint_sweep.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from ai_teleop.common.log import configure_logging, get_logger
from ai_teleop.policy.residual_policy import load_checkpoint

# Reuse the exact held-out evaluator (same split seed/fraction, same metric).
_spec = importlib.util.spec_from_file_location(
    "lab81_offline_eval", Path(__file__).with_name("lab81_offline_eval.py")
)
_lab81 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lab81)  # type: ignore[union-attr]
_evaluate = _lab81._evaluate

log = get_logger("lab106sweep")

DATASET = Path("data/dataset_vision")
RUNS = ["ftonly_baseline_lab82", "ftonly_wpos10_wd"]


def _action_rate_weight(checkpoint_dir: Path) -> object:
    loaded = load_checkpoint(checkpoint_dir / "checkpoint.pt")
    for attr in ("loss_config", "config"):
        obj = getattr(loaded, attr, None)
        if obj is not None:
            w = getattr(obj, "weight_action_rate", None)
            if w is not None:
                return w
    # Fall back to the raw checkpoint dict.
    raw = torch.load(checkpoint_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    return raw.get("loss_config", raw.get("config", {}))


def main() -> int:
    configure_logging()
    log.info("F/T offline held-out eval on %s (Δ̂ pos vs expert; zero-Δ ≈ 4.9 mm)", DATASET.name)
    for run in RUNS:
        cdir = Path("outputs/policy/runs") / run
        policy, zero = _evaluate(DATASET, cdir, vision=False)
        log.info(
            "%-22s arw=%-6s │ pos %.2f mm (zeroΔ %.2f) │ ori %.2f° │ grip %.3f N",
            run,
            _action_rate_weight(cdir),
            policy["pos_mm"],
            zero["pos_mm"],
            policy["ori_deg"],
            policy["grip_N"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
