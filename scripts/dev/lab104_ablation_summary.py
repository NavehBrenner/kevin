"""LAB-104 ablation summary: per-config KPIs + paired McNemar on success.

Reads a ``trials.csv`` from ``evaluate.py pair`` (one row per seed x config) and
prints, per config_label: success rate, mean ∫|jerk|, mean peak contact force,
mean time-to-insert (successes only). Then paired McNemar on success for the
key contrasts. Run from kevin/:

    uv run python scripts/dev/lab104_ablation_summary.py runs/eval/lab104_ar100/trials.csv
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from math import comb
from statistics import mean


def _mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact-binomial McNemar p over the b+c discordant pairs (p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "runs/eval/lab104_ar100/trials.csv"
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    by_config: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_config[row["config_label"]].append(row)

    success: dict[str, dict[int, bool]] = {}
    print(f"{'config':<12} {'success':>9} {'∫|jerk|':>9} {'peakF(N)':>9} {'t_insert(s)':>12}")
    for label, group in by_config.items():
        n = len(group)
        seats = [g for g in group if g["success"] == "True"]
        jerk = mean(float(g["jerk_integral"]) for g in group)
        peak = mean(float(g["peak_contact_force"]) for g in group)
        t_ins = mean(float(g["time_to_insert_s"]) for g in seats) if seats else float("nan")
        print(f"{label:<12} {len(seats):>3}/{n:<5} {jerk:>9.1f} {peak:>9.1f} {t_ins:>12.2f}")
        success[label] = {int(g["seed"]): g["success"] == "True" for g in group}

    print("\nPaired McNemar on success (b = left seats & right fails; c = reverse):")
    labels = list(success)
    pairs = [(a, b) for i, a in enumerate(labels) for b in labels[i + 1 :]]
    for a, b in pairs:
        seeds = sorted(set(success[a]) & set(success[b]))
        b_count = sum(success[a][s] and not success[b][s] for s in seeds)
        c_count = sum(not success[a][s] and success[b][s] for s in seeds)
        p = _mcnemar_exact(b_count, c_count)
        print(f"  {a:<10} vs {b:<10} │ b={b_count} c={c_count} │ p={p:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
