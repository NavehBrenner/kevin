"""One-shot aggregate of runs/eval/trials.csv — the LAB-53 head-to-head headline.

ponytail: throwaway dev aggregator; the publishable version is LAB-38's job.
Reports per-config success rate + the robust Phase-1 KPIs (time-to-insert on
successes, peak contact force), so the numbers are captured before LAB-38.
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

CSV = Path(__file__).resolve().parents[2] / "runs/eval/trials.csv"


def main() -> None:
    rows = list(csv.DictReader(CSV.open()))
    by_config: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_config[row["config_label"]].append(row)

    for config_label, trials in by_config.items():
        n = len(trials)
        successes = [t for t in trials if t["success"] == "True"]
        rate = len(successes) / n
        insert_times = [float(t["time_to_insert_s"]) for t in successes if t["time_to_insert_s"]]
        peak_forces = [float(t["peak_contact_force"]) for t in trials]
        mean_tti = statistics.mean(insert_times) if insert_times else float("nan")
        print(
            f"{config_label:12s} success {len(successes):3d}/{n} ({100 * rate:4.1f}%)  "
            f"mean t_insert(succ) {mean_tti:5.2f}s  "
            f"mean peak_force {statistics.mean(peak_forces):6.2f}N  "
            f"median peak_force {statistics.median(peak_forces):6.2f}N"
        )


if __name__ == "__main__":
    main()
