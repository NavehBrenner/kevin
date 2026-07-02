"""LAB-78 eyeball: command-dynamics + aim stats for a scripted set vs real-human targets.

Reuses compare_human_vs_scripted's per-episode metrics over data/dataset_0 and
prints them next to the 64-episode recorded-human targets, so we can see whether
max_approach_speed / position_bias_std need a nudge.
Run: uv run python scripts/dev/lab78_eyeball_dataset0.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from compare_human_vs_scripted import aggregate  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[2]

# Median [IQR] of the 64 recorded real-human episodes (the fit targets).
TARGETS = {
    "net_cmd_disp_mm": "442",
    "moving_frac": "0.46",
    "near_speed_med_mms": "—",
    "near_speed_p90_mms": "372",
    "cmd_tip_lat_near_med_mm": "18 [11,26]",
    "cmd_tip_lat_min_mm": "—",
}


def col(a: np.ndarray) -> str:
    return f"{np.median(a):.3g} [{np.percentile(a, 25):.2g},{np.percentile(a, 75):.2g}]"


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "dataset_0"  # ponytail: argv, not argparse
    paths = sorted((ROOT / "data" / dataset / "runs").glob("episode_*/episode.npz"))
    scr = aggregate(paths)
    print(f"\n{dataset}: {len(paths)} episodes\n")
    print(f"{'metric':<26}{'SCRIPTED (new)':<26}{'REAL-HUMAN target':<20}")
    print("-" * 72)
    for key, target in TARGETS.items():
        print(f"{key:<26}{col(scr[key]):<26}{target:<20}")


if __name__ == "__main__":
    main()
