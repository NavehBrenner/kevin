"""LAB-105 Stage-C ablation read-out: per-seed success + margins across the 3 configs.

uv run python scripts/dev/lab105_stagec_ablation_summary.py runs/eval_stageC/trials.csv
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

CONFIGS = ("human_only", "ftonly", "vision")


def _is_success(row: dict[str, str]) -> bool:
    return row.get("success", "").strip().lower() in ("true", "1")


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/eval_stageC/trials.csv")
    rows = list(csv.DictReader(path.open()))
    by_seed: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    success = dict.fromkeys(CONFIGS, 0)
    total = dict.fromkeys(CONFIGS, 0)
    for row in rows:
        config = row["config_label"]
        by_seed[row["seed"]][config] = row
        total[config] += 1
        success[config] += int(_is_success(row))

    for config in CONFIGS:
        rate = 100 * success[config] / total[config] if total[config] else float("nan")
        print(f"{config:12} {success[config]:2d}/{total[config]:2d}  ({rate:.0f}%)")

    print("\nseed | human ftonly vision   (1 = insert)")
    for seed in sorted(by_seed, key=lambda value: int(value) if value.isdigit() else 0):
        marks = " ".join(
            "1" if _is_success(by_seed[seed].get(config, {})) else "." for config in CONFIGS
        )
        print(f"{seed:>4} |   {marks}")

    # Where vision wins/loses relative to F/T-only — the headline is the sign of this.
    vision_only = sum(
        1
        for seed in by_seed
        if _is_success(by_seed[seed].get("vision", {}))
        and not _is_success(by_seed[seed].get("ftonly", {}))
    )
    ftonly_only = sum(
        1
        for seed in by_seed
        if _is_success(by_seed[seed].get("ftonly", {}))
        and not _is_success(by_seed[seed].get("vision", {}))
    )
    print(f"\nvision succeeds where F/T fails: {vision_only}")
    print(f"F/T succeeds where vision fails: {ftonly_only}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
