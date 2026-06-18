"""Record the Phase-1 residual policy's size + per-step inference latency (LAB-33).

The acceptance criterion needs two numbers on record: the parameter count, and
the per-control-tick latency of the O(1) ``step`` path against the ~10 ms budget
(see LAB-34). This probes the *deployment* path — ``model.step`` with ``B=1`` on
CPU, ``eval`` + ``no_grad`` — which is how the seam adapter will call the policy
at run time, not the (batched, sequence) training forward.

Run: uv run python scripts/dev/policy_latency.py
"""

from __future__ import annotations

import statistics
import time

import torch

from ai_teleop.common.log import configure_logging, get_logger
from ai_teleop.policy.config import PolicyConfig
from ai_teleop.policy.model import ResidualPolicy

log = get_logger("policy-latency")

CONTROL_BUDGET_MS = 10.0  # one ~100 Hz control tick
WARMUP_STEPS = 50
TIMED_STEPS = 1000


def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def _measure_step_latency_ms(model: ResidualPolicy, config: PolicyConfig) -> list[float]:
    """Time ``model.step`` for a single agent (B=1) over many isolated ticks.

    Each tick carries the hidden state forward, matching real deployment; only
    the forward compute is timed (input tensors are pre-built).
    """
    command = torch.randn(1, config.command_dim)
    force_torque = torch.randn(1, config.force_torque_dim)
    proprioception = torch.randn(1, config.proprioception_dim)

    samples: list[float] = []
    hidden: torch.Tensor | None = None
    with torch.no_grad():
        for tick in range(WARMUP_STEPS + TIMED_STEPS):
            start = time.perf_counter()
            delta, hidden = model.step(command, force_torque, proprioception, hidden=hidden)
            elapsed_ms = (time.perf_counter() - start) * 1e3
            if tick >= WARMUP_STEPS:
                samples.append(elapsed_ms)
            del delta
    return samples


def main() -> None:
    configure_logging(level="INFO")
    torch.set_num_threads(1)  # single-thread: closest to the real-time control budget

    config = PolicyConfig()
    model = ResidualPolicy(config).eval()

    total, trainable = _count_parameters(model)
    log.info(
        "ResidualPolicy: input_dim=%d hidden=%d layers=%d head=%s | params total=%s trainable=%s",
        config.input_dim,
        config.hidden_size,
        config.num_layers,
        config.head_hidden,
        f"{total:,}",
        f"{trainable:,}",
    )

    samples = _measure_step_latency_ms(model, config)
    median_ms = statistics.median(samples)
    p95_ms = sorted(samples)[int(0.95 * len(samples)) - 1]
    log.info(
        "step latency (B=1, CPU, 1 thread): median=%.3f ms  p95=%.3f ms  budget=%.1f ms",
        median_ms,
        p95_ms,
        CONTROL_BUDGET_MS,
    )
    verdict = "WITHIN" if p95_ms < CONTROL_BUDGET_MS else "OVER"
    log.info("verdict: %s the %.1f ms control budget", verdict, CONTROL_BUDGET_MS)


if __name__ == "__main__":
    main()
