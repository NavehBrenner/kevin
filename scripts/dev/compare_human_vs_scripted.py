"""Precise comparison: recorded *real human* vs scripted noisy human.

Two question families, kept separate:

1. REALIZED peg-tip geometry (KPI-grade, via the shared seating convention):
   does the real human drive the peg tip into the same lateral band the policy
   was trained/calibrated on? `lateral_error <= ~3-5 mm` is the chamfer capture
   band (bore ~4-5 mm, chamfer 1-4 mm) -- outside it, F/T has no lever.

2. COMMAND-stream dynamics (the policy's command-history input): speed,
   continuity (staircase vs smooth), and whether an approach phase exists.

Insertion axis = hole local +x; peg long axis = peg local +z, tip at
+PEG_HALF_LENGTH (matches common/seating.py). Orientation is position-only in
this rig, so we do not analyze command orientation.

Run: uv run python scripts/dev/compare_human_vs_scripted.py
Writes outputs/human_vs_scripted.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PEG_HALF_LENGTH = 0.030
D_FAR = 0.10  # expert far-field gate; "near-field" = realized tip distance below this
CHAMFER_BAND_MM = 5.0  # generous capture-band ceiling for the in-band fraction


def quat_col(quat: np.ndarray, col: int) -> np.ndarray:
    """Column `col` of the rotation matrix for a (T,4) wxyz quaternion stream."""
    q = quat / (np.linalg.norm(quat, axis=1, keepdims=True) + 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    if col == 0:
        return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], 1)
    if col == 2:
        return np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1)
    raise ValueError(col)


def episode_stats(cols: dict[str, np.ndarray]) -> dict[str, float]:
    sim_time = cols["sim_time"].astype(float)
    dt = float(np.median(np.diff(sim_time))) if len(sim_time) > 1 else 0.002
    cmd = cols["cmd_position"].astype(float)

    # --- realized tip geometry (needs privileged peg + hole poses) ---
    peg, hole = cols["peg_pose"].astype(float), cols["target_hole_pose"].astype(float)
    tip = peg[:, :3] + PEG_HALF_LENGTH * quat_col(peg[:, 3:], 2)
    n_axis = quat_col(hole[:, 3:], 0)
    err = hole[:, :3] - tip
    axial = np.sum(err * n_axis, axis=1)
    lateral = np.linalg.norm(err - axial[:, None] * n_axis, axis=1)  # m
    tip_dist = np.linalg.norm(err, axis=1)
    penetration = -axial
    contact = penetration > -0.005  # at/near the wall face
    lat_at_contact = lateral[contact] if contact.any() else lateral[-5:]

    # --- commanded-tip lateral error: pure OPERATOR AIM, pre-assist, apples-to-apples ---
    # (recorded has no assist; scripted cmd is the noisy human pre-expert-delta.)
    cmd_tip = cmd + PEG_HALF_LENGTH * quat_col(cols["cmd_quaternion"].astype(float), 2)
    cmd_err = hole[:, :3] - cmd_tip
    cmd_axial = np.sum(cmd_err * n_axis, axis=1)
    cmd_lateral = np.linalg.norm(cmd_err - cmd_axial[:, None] * n_axis, axis=1)
    near_full = tip_dist < D_FAR
    cmd_lat_near = cmd_lateral[near_full] if near_full.any() else cmd_lateral[-5:]

    # --- command-stream dynamics ---
    step_move = np.linalg.norm(np.diff(cmd, axis=0), axis=1)
    moving = step_move > 1e-5
    near = tip_dist[:-1] < D_FAR  # near-field steps (where the assist actually acts)
    speed_near = step_move[near] / dt if near.any() else np.array([0.0])

    return {
        # operator AIM (commanded tip, pre-assist) — the fit target for scripted bias
        "cmd_tip_lat_near_med_mm": float(np.median(cmd_lat_near) * 1e3),
        "cmd_tip_lat_min_mm": float(np.min(cmd_lateral) * 1e3),
        "cmd_aim_in_band": float(np.min(cmd_lateral) * 1e3 <= CHAMFER_BAND_MM),
        # realized geometry (NOTE: recorded=raw, scripted=expert-assisted — not apples-to-apples)
        "tip_lat_min_mm": float(np.min(lateral) * 1e3),
        "tip_lat_at_contact_mm": float(np.median(lat_at_contact) * 1e3),
        "in_chamfer_band": float(np.min(lateral) * 1e3 <= CHAMFER_BAND_MM),
        "max_penetration_mm": float(np.max(penetration) * 1e3),
        # command dynamics
        "cmd_start_dist_mm": float(tip_dist[0] * 1e3),
        "net_cmd_disp_mm": float(np.linalg.norm(cmd[-1] - cmd[0]) * 1e3),
        "near_speed_med_mms": float(np.median(speed_near) * 1e3),
        "near_speed_p90_mms": float(np.percentile(speed_near, 90) * 1e3),
        "moving_frac": float(moving.mean()),
        "duration_s": float(sim_time[-1] - sim_time[0]),
    }


def aggregate(paths: list[Path]) -> dict[str, np.ndarray]:
    rows = []
    for p in paths:
        z = np.load(p, allow_pickle=True)
        cols = {k: z[k] for k in z.files if k != "metadata"}
        if "cmd_position" not in cols or "peg_pose" not in cols:
            continue
        rows.append(episode_stats(cols))
    keys = sorted(rows[0]) if rows else []
    return {k: np.array([r[k] for r in rows]) for k in keys}


def col(label: str, a: np.ndarray) -> str:
    return f"{np.median(a):.3g} [{np.percentile(a, 25):.2g},{np.percentile(a, 75):.2g}]"


def main() -> None:
    rec_paths = sorted((ROOT / "data" / "recorded" / "runs").glob("episode_*/episode.npz"))
    scr_paths = sorted((ROOT / "data" / "dataset_1" / "runs").glob("episode_*.npz"))
    rec = aggregate(rec_paths)
    rec_old = aggregate(rec_paths[:8])  # original batch — rig-consistency check
    rec_new = aggregate(rec_paths[8:])
    scr = aggregate(scr_paths)

    keys = sorted(rec)
    print(
        f"\nRecorded: {len(rec_paths)} ({len(rec_paths) - 8} new + 8 original)   Scripted: {len(scr_paths)}\n"
    )
    print(f"{'metric':<24}{'RECORDED new':<24}{'RECORDED orig-8':<24}{'SCRIPTED':<24}")
    print("-" * 96)
    for k in keys:
        print(f"{k:<24}{col(k, rec_new[k]):<24}{col(k, rec_old[k]):<24}{col(k, scr[k]):<24}")

    print("\n--- decision-relevant fractions ---")
    for tag, d in [("recorded(all)", rec), ("scripted", scr)]:
        print(
            f"  {tag:<16} in-chamfer-band: {d['in_chamfer_band'].mean():.0%}   "
            f"seated(pen>1.5cm): {(d['max_penetration_mm'] > 15).mean():.0%}"
        )

    # --- plots ---
    pk = [
        "cmd_tip_lat_near_med_mm",
        "cmd_tip_lat_min_mm",
        "near_speed_med_mms",
        "moving_frac",
        "net_cmd_disp_mm",
        "duration_s",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, k in zip(axes.ravel(), pk, strict=True):
        r, s = rec[k], scr[k]
        bins = np.histogram_bin_edges(np.concatenate([r, s]), bins=14)
        ax.hist(
            s, bins=bins, alpha=0.5, label=f"scripted (n={len(s)})", color="tab:blue", density=True
        )
        ax.hist(
            r, bins=bins, alpha=0.6, label=f"recorded (n={len(r)})", color="tab:red", density=True
        )
        if k == "tip_lat_min_mm":
            ax.axvline(CHAMFER_BAND_MM, color="k", ls="--", lw=1, label="chamfer band")
        ax.set_title(k, fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle(f"Recorded human vs scripted noisy human (n={len(rec_paths)} vs {len(scr_paths)})")
    fig.tight_layout()
    out = ROOT / "outputs" / "human_vs_scripted.png"
    fig.savefig(out, dpi=110)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
