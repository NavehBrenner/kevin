"""LAB-42/H-8: is the new 100-seed run's disagreement policy, harness, or sampling?

The published Phase-1 band run (30 seeds) and the 2026-07-22 re-run (100 seeds) share a
master seed, so seeds 0..29 are the *same trials*. That makes the causes separable:

* **human_only** depends on no checkpoint at all. If its seeds 0..29 outcomes differ, the
  harness or the sim drifted between 2026-07-07 and today — a finding about the eval path.
* **residual** additionally depends on the checkpoint. If human_only matches and residual
  does not, the difference is the policy, cleanly.
* If both match on 0..29, the headline gap is pure sampling over seeds 30..99.

Read-only. Run: `uv run python scripts/dev/lab42_seed_overlap.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

from ai_teleop.eval.report import group_by_config, load_trials

OLD = Path("docs/results/phase-1/band_scale0.4_trials.csv")
NEW = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/eval_lab101_band100/trials.csv")


def by_seed(path: Path, label: str) -> dict[int, bool]:
    grouped = group_by_config(load_trials(path))
    return {t.seed: t.success for t in grouped[label] if t.seed is not None}


def compare(label: str) -> None:
    old, new = by_seed(OLD, label), by_seed(NEW, label)
    shared = sorted(set(old) & set(new))
    agree = [s for s in shared if old[s] == new[s]]
    differ = [s for s in shared if old[s] != new[s]]
    print(f"\n{label}: {len(shared)} shared seeds — {len(agree)} agree, {len(differ)} differ")
    print(f"  old rate on shared seeds: {sum(old[s] for s in shared) / len(shared):.1%}")
    print(f"  new rate on shared seeds: {sum(new[s] for s in shared) / len(shared):.1%}")
    if differ:
        flips = [f"seed {s}: {'✓' if old[s] else '✗'}→{'✓' if new[s] else '✗'}" for s in differ]
        print(f"  differing: {', '.join(flips)}")


def main() -> None:
    print(f"old = {OLD} (30 seeds, 2026-07-07)\nnew = {NEW} (100 seeds, 2026-07-22)")
    for label in ("human_only", "residual"):
        compare(label)

    new_all = by_seed(NEW, "human_only")
    tail = [s for s in new_all if s >= 30]
    print(
        f"\nnew human_only on seeds 0-29: {sum(new_all[s] for s in range(30)) / 30:.1%}"
        f" | on seeds 30-99: {sum(new_all[s] for s in tail) / len(tail):.1%}"
    )


if __name__ == "__main__":
    main()
