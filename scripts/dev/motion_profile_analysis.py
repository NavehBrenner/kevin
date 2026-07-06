"""Motion-per-step profile: recorded real-human vs scripted noisy human.

Complements `compare_human_vs_scripted.py`'s per-episode aggregates (net travel,
median speed, ...) with the *within-episode* shape: how much the arm actually
moves at each point along an episode, not just on average over the whole thing.

Uses REALIZED Cartesian motion (`||ee_pose[t+1,:3] - ee_pose[t,:3]||`, what
physically happened) rather than the `cmd_*` intent stream `compare_human_vs_scripted.py`
mostly analyzes -- the two are complementary: cmd-stream dynamics are the
*operator's* input signal, ee-pose motion is what the *controller + physics*
actually realized (post-clamp, post-impedance).

Three outputs, each recorded-vs-scripted overlaid:

1. Pooled per-step motion-magnitude histogram (all steps, all episodes).
2. Motion vs normalized episode progress (0-100%, 20 bins): median +/- IQR band,
   answering "does the arm move more at the start, middle, or end of an episode".
3. Episode length + motion-per-step, grouped by `terminal_reason` -- does a
   force_abort episode look different (motion-wise) from a timeout or a success.

Run: uv run python scripts/dev/motion_profile_analysis.py [--scripted data/dataset_9]
Writes outputs/motion_profile.png and prints the grouped-by-terminal-reason tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
N_PROGRESS_BINS = 20


def load_episode(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=True) as z:
        cols = {k: z[k] for k in z.files if k != "metadata"}
        meta = json.loads(str(z["metadata"]))
    return cols, meta


def episode_motion(cols: dict[str, np.ndarray]) -> np.ndarray:
    """Per-step realized Cartesian motion (mm/step), length T-1."""
    ee = cols["ee_pose"][:, :3].astype(float)
    return np.linalg.norm(np.diff(ee, axis=0), axis=1) * 1e3


def episode_joint_speed(cols: dict[str, np.ndarray]) -> np.ndarray:
    """Per-step joint-space speed norm (rad/s), length T."""
    jv = cols["joint_velocities"].astype(float)
    return np.linalg.norm(jv, axis=1)


def progress_binned(motion: np.ndarray, n_bins: int = N_PROGRESS_BINS) -> np.ndarray:
    """Bin one episode's per-step motion into n_bins equal-width % of elapsed steps,
    returning the mean motion in each bin (nan for an episode too short to fill a bin).
    """
    edges = np.linspace(0, len(motion), n_bins + 1).astype(int)
    out = np.full(n_bins, np.nan)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if hi > lo:
            out[i] = motion[lo:hi].mean()
    return out


def collect(paths: list[Path]) -> dict:
    """Load every episode; return pooled motion, per-episode progress curves, and
    per-episode summaries keyed by terminal_reason.
    """
    pooled_motion = []
    pooled_joint = []
    progress_curves = []
    by_reason: dict[str, list[dict]] = {}
    for p in paths:
        cols, meta = load_episode(p)
        motion = episode_motion(cols)
        joint = episode_joint_speed(cols)
        if len(motion) < N_PROGRESS_BINS:
            continue  # too short to bin meaningfully
        pooled_motion.append(motion)
        pooled_joint.append(joint)
        progress_curves.append(progress_binned(motion))
        reason = meta.get("terminal_reason", "unknown")
        by_reason.setdefault(reason, []).append({
            "n_steps": meta["n_steps"],
            "duration_s": float(cols["sim_time"][-1] - cols["sim_time"][0]),
            "motion_median_mm": float(np.median(motion)),
            "motion_p90_mm": float(np.percentile(motion, 90)),
            "motion_total_mm": float(motion.sum()),
        })
    return {
        "pooled_motion": np.concatenate(pooled_motion) if pooled_motion else np.array([]),
        "pooled_joint": np.concatenate(pooled_joint) if pooled_joint else np.array([]),
        "progress_curves": np.array(progress_curves)
        if progress_curves
        else np.zeros((0, N_PROGRESS_BINS)),
        "by_reason": by_reason,
        "n_episodes": len(paths),
    }


def summarize_reason_table(tag: str, by_reason: dict[str, list[dict]]) -> None:
    print(f"\n--- {tag}: grouped by terminal_reason ---")
    header = f"{'reason':<14}{'n':<6}{'n_steps med [IQR]':<26}{'motion_med mm/step [IQR]':<30}{'total travel med mm':<20}"
    print(header)
    print("-" * len(header))
    for reason, rows in sorted(by_reason.items()):
        n_steps = np.array([r["n_steps"] for r in rows])
        motion_med = np.array([r["motion_median_mm"] for r in rows])
        total = np.array([r["motion_total_mm"] for r in rows])

        def fmt(a: np.ndarray) -> str:
            return f"{np.median(a):.3g} [{np.percentile(a, 25):.2g},{np.percentile(a, 75):.2g}]"

        print(f"{reason:<14}{len(rows):<6}{fmt(n_steps):<26}{fmt(motion_med):<30}{fmt(total):<20}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recorded", default=str(ROOT / "data" / "recorded" / "runs"))
    ap.add_argument("--scripted", default=str(ROOT / "data" / "dataset_9"))
    args = ap.parse_args()

    rec_paths = sorted(Path(args.recorded).glob("episode_*/episode.npz"))
    scr_root = Path(args.scripted)
    scr_paths = sorted((scr_root / "runs").glob("episode_*/episode.npz")) or sorted(
        scr_root.glob("runs/episode_*.npz")
    )
    print(f"recorded: {len(rec_paths)} episodes ({args.recorded})")
    print(f"scripted: {len(scr_paths)} episodes ({args.scripted})")

    rec = collect(rec_paths)
    scr = collect(scr_paths)

    print("\n--- pooled per-step motion (mm/step, realized ee_pose delta) ---")
    for tag, d in [("recorded", rec), ("scripted", scr)]:
        m = d["pooled_motion"]
        if len(m) == 0:
            print(f"  {tag}: no data")
            continue
        print(
            f"  {tag:<10} n_steps={len(m):>7}  median={np.median(m):.4f}  "
            f"p90={np.percentile(m, 90):.4f}  mean={m.mean():.4f}  frac_moving(>0.01mm)={float((m > 0.01).mean()):.2%}"
        )

    summarize_reason_table("RECORDED", rec["by_reason"])
    summarize_reason_table("SCRIPTED", scr["by_reason"])

    # --- plots ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    both = np.concatenate([rec["pooled_motion"], scr["pooled_motion"]])
    bins = np.histogram_bin_edges(both[both > 0] if (both > 0).any() else both, bins=60)
    ax.hist(
        scr["pooled_motion"],
        bins=bins,
        alpha=0.5,
        density=True,
        label=f"scripted (n_ep={scr['n_episodes']})",
        color="tab:blue",
    )
    ax.hist(
        rec["pooled_motion"],
        bins=bins,
        alpha=0.6,
        density=True,
        label=f"recorded (n_ep={rec['n_episodes']})",
        color="tab:red",
    )
    ax.set_xlabel("per-step realized motion (mm/step)")
    ax.set_ylabel("density")
    ax.set_title("Pooled per-step motion magnitude")
    ax.legend(fontsize=8)

    ax = axes[1]
    x = np.linspace(0, 100, N_PROGRESS_BINS)
    for tag, d, color in [("scripted", scr, "tab:blue"), ("recorded", rec, "tab:red")]:
        curves = d["progress_curves"]
        if curves.shape[0] == 0:
            continue
        med = np.nanmedian(curves, axis=0)
        q25 = np.nanpercentile(curves, 25, axis=0)
        q75 = np.nanpercentile(curves, 75, axis=0)
        ax.plot(x, med, label=tag, color=color)
        ax.fill_between(x, q25, q75, alpha=0.2, color=color)
    ax.set_xlabel("episode progress (%)")
    ax.set_ylabel("mean motion in bin (mm/step)")
    ax.set_title("Motion vs. normalized episode progress (median +/- IQR)")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"Motion profile: recorded (n={rec['n_episodes']}) vs scripted (n={scr['n_episodes']})"
    )
    fig.tight_layout()
    out = ROOT / "outputs" / "motion_profile.png"
    fig.savefig(out, dpi=110)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
