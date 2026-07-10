"""LAB-105 diagnosis: how much of each on-policy rollout gets a *non-zero* expert
label? The reactive expert is zero beyond d_far (0.15 m), so drift states in free
space are relabeled "do nothing" — adding no recovery signal. Group the non-zero
fraction by terminal reason to see whether the timeout rollouts (policy wandering)
are mostly no-op labels.

    uv run python scripts/dev/lab105_label_coverage.py data/dagger_agg1
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from ai_teleop.data.trajectory import load_episode


def main() -> int:
    agg = Path(sys.argv[1] if len(sys.argv) > 1 else "data/dagger_agg1")
    by_reason: dict[str, list[float]] = defaultdict(list)
    rows_by_reason: dict[str, int] = defaultdict(int)
    nonzero_rows_by_reason: dict[str, int] = defaultdict(int)

    for ep_dir in sorted((agg / "runs").glob("episode_1*")):  # dagger indices are >= 1_000_000
        columns, metadata = load_episode(ep_dir / "episode.npz")
        magnitude = np.linalg.norm(columns["delta_position"], axis=1)
        nonzero = magnitude > 1e-9
        reason = str(metadata["terminal_reason"])
        by_reason[reason].append(float(nonzero.mean()))
        rows_by_reason[reason] += len(nonzero)
        nonzero_rows_by_reason[reason] += int(nonzero.sum())

    print(
        f"{'terminal_reason':<14} {'episodes':>8} {'mean %rows w/ nonzero label':>28} {'total rows':>12}"
    )
    for reason in sorted(by_reason):
        fracs = by_reason[reason]
        pooled = nonzero_rows_by_reason[reason] / rows_by_reason[reason]
        print(
            f"{reason:<14} {len(fracs):>8} {100 * float(np.mean(fracs)):>26.1f}% "
            f"({100 * pooled:>4.1f}% pooled) {rows_by_reason[reason]:>12}"
        )
    all_nonzero = sum(nonzero_rows_by_reason.values())
    all_rows = sum(rows_by_reason.values())
    print(
        f"\noverall: {100 * all_nonzero / all_rows:.1f}% of {all_rows} relabeled rows carry a non-zero expert Δ"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
