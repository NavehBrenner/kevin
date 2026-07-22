"""LAB-42 H-9: compare two corpus manifests episode-by-episode.

`dataset_10` was rebuilt from `dataset_9`'s committed config under 2026-07-22 code, so the
two share a fingerprint by construction. This prints how far apart their episodes actually
are — the measured size of the G-4 hole (a content hash over the config cannot see code
drift), on the corpus behind the project's headline result.

Read-only. Run: `uv run python scripts/dev/lab42_corpus_diff.py data/dataset_9 data/dataset_10`
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: str) -> dict:
    return json.loads((Path(path) / "metadata.json").read_text(encoding="utf-8"))


def main() -> None:
    left_path, right_path = sys.argv[1], sys.argv[2]
    left, right = load(left_path), load(right_path)

    print(f"{left_path}  generated {left['generated_at']}  fingerprint {left['fingerprint']}")
    print(f"{right_path}  generated {right['generated_at']}  fingerprint {right['fingerprint']}")
    print(f"  fingerprint identical: {left['fingerprint'] == right['fingerprint']}")
    print(f"  config identical:      {left['config'] == right['config']}")

    left_eps, right_eps = left["episodes"], right["episodes"]
    print(f"  episodes: {len(left_eps)} vs {len(right_eps)}")

    steps_differ = [
        (i, a["n_steps"], b["n_steps"])
        for i, (a, b) in enumerate(zip(left_eps, right_eps, strict=True))
        if a["n_steps"] != b["n_steps"]
    ]
    outcome_differ = [
        (i, a.get("terminal_reason"), b.get("terminal_reason"))
        for i, (a, b) in enumerate(zip(left_eps, right_eps, strict=True))
        if a.get("terminal_reason") != b.get("terminal_reason")
    ]
    for label, episodes in (("left", left_eps), ("right", right_eps)):
        successes = sum(1 for e in episodes if e.get("terminal_reason") == "success")
        print(
            f"  {label} expert success: {successes}/{len(episodes)} ({successes / len(episodes):.1%})"
        )

    print(f"\n  episodes with a different n_steps: {len(steps_differ)}/{len(left_eps)}")
    big = sorted(steps_differ, key=lambda d: -abs(d[1] - d[2]))[:10]
    for index, before, after in big:
        print(f"    ep{index:<4} {before:>6} -> {after:<6} ({after - before:+d})")
    print(f"  episodes whose outcome flipped: {len(outcome_differ)}")
    for item in outcome_differ[:10]:
        print(f"    ep{item[0]:<4} {item[1]} -> {item[2]}")


if __name__ == "__main__":
    main()
