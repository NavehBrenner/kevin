"""Real-time latency check for the vision deploy path (LAB-83).

Measures the per-step inference cost of the deployed residual policy against the
~10 ms / 100 Hz control budget, on the *actual* deploy path (`LearnedResidual.
get_delta` over a real `Observation` carrying a real wrist frame), so the number is
the honest one — not the LAB-81 bring-up estimate taken before the CUDA wheel and
before the deploy path existed.

Three costs, because the env is the frame-rate limiter (`SimEnv.enable_wrist_capture`):

* **F/T-only** `get_delta` — the recurrent O(1) baseline.
* **vision** `get_delta` — normalize + CNN encode + GRU, run **every** tick (the env
  hands the policy the held frame; the policy re-encodes it — the honest per-tick
  compute cost).
* **render** — `render_wrist_camera` alone, which fires only 1/`render_every` ticks,
  so it amortizes.

Reported: steady-state per-tick (vision compute), worst-case tick (compute + a fresh
render), and the amortized per-tick (compute + render/`render_every`).

    uv run python scripts/dev/lab83_latency.py --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ai_teleop.common.command import Command  # noqa: E402
from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)
from ai_teleop.policy import LearnedResidual  # noqa: E402
from ai_teleop.sim.config import EnvConfig  # noqa: E402
from ai_teleop.sim.env_setup import make_env  # noqa: E402

log = get_logger("lab83-latency")

_RUNS = Path(__file__).resolve().parents[2] / "outputs" / "policy" / "runs"


def _command_from(observation) -> Command:
    """A base command aimed at the current EE pose (contents don't affect timing)."""
    return Command(
        target_position=observation.ee_pose[:3].copy(),
        target_quaternion=observation.ee_pose[3:7].copy(),
    )


def _time_get_delta(
    policy: LearnedResidual, observation, command, *, iters: int, cuda: bool
) -> float:
    """Mean ms per `get_delta` over `iters` calls (10-call warmup, cuda-synced)."""
    for _ in range(10):
        policy.get_delta(observation, command)
    if cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        policy.get_delta(observation, command)
    if cuda:
        torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - start) / iters


def _time_render(environment, *, iters: int) -> float:
    """Mean ms per offscreen wrist render (the cost the env amortizes)."""
    for _ in range(3):
        environment.render_wrist_camera()
    start = time.perf_counter()
    for _ in range(iters):
        environment.render_wrist_camera()
    return 1e3 * (time.perf_counter() - start) / iters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vision-checkpoint", default=str(_RUNS / "vision_frozen_lab82" / "checkpoint.pt")
    )
    parser.add_argument(
        "--ftonly-checkpoint", default=str(_RUNS / "ftonly_baseline_lab82" / "checkpoint.pt")
    )
    parser.add_argument("--device", default="cuda", help="Torch device (cuda default).")
    parser.add_argument("--iters", type=int, default=300, help="Timed get_delta calls per config.")
    parser.add_argument("--render-iters", type=int, default=30, help="Timed render calls.")
    parser.add_argument(
        "--render-every", type=int, default=20, help="Deploy frame decimation (env rate limit)."
    )
    parser.add_argument("--budget-ms", type=float, default=10.0, help="Control budget (100 Hz).")
    add_logging_arguments(parser)
    args = parser.parse_args()
    configure_from_args(args)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        log.warning("cuda unavailable — falling back to cpu (numbers will be worse than deploy)")
        device = "cpu"
    cuda = device.startswith("cuda")

    # A real scene → a real wrist frame + realistic proprioception for the policy input.
    environment = make_env(EnvConfig(wall_seed=None), render_mode="headless")
    try:
        environment.enable_wrist_capture(args.render_every)
        observation = environment.reset()
        command = _command_from(observation)
        assert observation.wrist_image is not None, "capture enabled but no frame stamped"
        log.info("wrist frame %s on device %s", observation.wrist_image.shape, device)

        ftonly = LearnedResidual.from_checkpoint(args.ftonly_checkpoint, device=device)
        vision = LearnedResidual.from_checkpoint(args.vision_checkpoint, device=device)
        assert not ftonly.use_vision and vision.use_vision, "checkpoint use_vision flags unexpected"

        ftonly_ms = _time_get_delta(ftonly, observation, command, iters=args.iters, cuda=cuda)
        vision_ms = _time_get_delta(vision, observation, command, iters=args.iters, cuda=cuda)
        render_ms = _time_render(environment, iters=args.render_iters)
    finally:
        environment.close()

    amortized_ms = vision_ms + render_ms / args.render_every
    worstcase_ms = vision_ms + render_ms

    log.info("=== LAB-83 real-time latency (device=%s, budget=%.1f ms) ===", device, args.budget_ms)
    log.info("F/T-only get_delta          : %6.3f ms/step", ftonly_ms)
    log.info("vision get_delta (encode/tick): %6.3f ms/step", vision_ms)
    log.info("wrist render (1/%d ticks)    : %6.3f ms", args.render_every, render_ms)
    log.info(
        "vision amortized per tick     : %6.3f ms/step  (compute + render/%d)",
        amortized_ms,
        args.render_every,
    )
    log.info(
        "vision worst-case tick        : %6.3f ms/step  (compute + fresh render)", worstcase_ms
    )
    verdict = "WITHIN" if amortized_ms <= args.budget_ms else "OVER"
    log.info("amortized vs budget: %s (%.3f vs %.1f ms)", verdict, amortized_ms, args.budget_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
