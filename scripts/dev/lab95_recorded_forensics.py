"""LAB-95 forensics: what physically drives the recorded force-aborts?

LAB-91/92 ruled out the operator's *approach-speed profile* as the lever for the
force-abort-rate / motion-signature gap (see
`project-wiki/synthesis/scripted-vs-real-operator.md`). The two candidates left
act at contact time, and they are distinguishable directly from the recorded
raw-human episodes (`data/recorded/runs`, no assist, so the realized command IS
`cmd_position`):

* **Static deep push** — the operator's command settles *past* the wall along
  the bore, so the impedance law's steady state `F ~ K * (cmd - ee)` climbs to
  the 30N watchdog. Signature: large bore-axial cmd-vs-ee tracking error at
  peak force, slow force rise (seconds), low approach speed at contact onset.
  A command-side lever (heavier-tailed bias / commanded depth) could reproduce
  this.
* **Contact transient** — the operator arrives fast and the impact spike trips
  the watchdog before any steady state. Signature: high ee/cmd speed right
  before contact onset, force rise ~ the impedance settling time (~100ms), small
  axial tracking error. No command-side lever survives the impedance filter
  here; the lever would have to be controller-side (sign-off required).

Prints per-outcome medians of both signatures over the recorded corpus.

Run: uv run python scripts/dev/lab95_recorded_forensics.py [--recorded data/recorded/runs]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402

from ai_teleop.common.utils.rotations import axis_from_quat  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

_CONTROL_HZ = 500.0
# Free-space |F| is ~8.6N (grasped-peg weight through the wrist F/T), so contact
# onset is baseline-relative: first sustained excursion this far above the
# episode's own free-space baseline (median of the first second).
_CONTACT_ONSET_ABOVE_BASELINE_N = 5.0
_ONSET_SUSTAIN_TICKS = 5
_BASELINE_TICKS = 500
_PRE_CONTACT_WINDOW = 100  # ticks (~200ms; vision commands refresh at ~30fps)
# Bore-axial TCP stiffness (control/backbone.py _DEFAULT_STIFFNESS_TCP x/y/z =
# 400/400/500 N/m); the bore is near world +x for the mildly tilted walls, so
# 400 N/m is the relevant static-push gain for predicted force = K * error.
_STIFFNESS_BORE = 400.0


def contact_forensics(
    force: np.ndarray,
    ee: np.ndarray,
    cmd: np.ndarray,
    insertion_axis: np.ndarray,
    outcome: str,
) -> dict:
    """Contact-time signature of one episode from its (T,) force / (T,3) tracks.

    Reused by the scripted-rollout probe (`lab95_scripted_contact_probe.py`) so
    recorded and scripted corpora are measured with the identical definition.
    """
    baseline = float(np.median(force[:_BASELINE_TICKS]))
    above = force > baseline + _CONTACT_ONSET_ABOVE_BASELINE_N
    # First index where the excursion is sustained for _ONSET_SUSTAIN_TICKS.
    sustained = np.convolve(above, np.ones(_ONSET_SUSTAIN_TICKS), mode="valid")
    onset_candidates = np.flatnonzero(sustained >= _ONSET_SUSTAIN_TICKS)
    if len(onset_candidates) == 0:
        return {"outcome": outcome, "peak_force_n": float(force.max()), "contact": False}
    onset = int(onset_candidates[0])
    peak = int(np.argmax(force))

    # Bore-axial command-vs-realized tracking error (m): the impedance law's
    # static force predictor. Positive = command sits deeper along the bore
    # than the arm reached.
    axial_error_at_peak = float((cmd[peak] - ee[peak]) @ insertion_axis)
    axial_error_series = np.einsum("ij,j->i", cmd[onset:] - ee[onset:], insertion_axis)
    axial_error_max = float(axial_error_series.max())

    # Approach speed just before contact (mm/s), realized and commanded.
    lo = max(0, onset - _PRE_CONTACT_WINDOW)
    window = slice(lo, onset + 1)

    def speed_mm_s(track: np.ndarray) -> float:
        d = np.diff(track[window], axis=0)
        if len(d) == 0:
            return float("nan")
        return float(np.linalg.norm(d, axis=1).mean() * _CONTROL_HZ * 1e3)

    return {
        "outcome": outcome,
        "contact": True,
        "peak_force_n": float(force[peak]),
        "axial_error_at_peak_mm": axial_error_at_peak * 1e3,
        "axial_error_max_mm": axial_error_max * 1e3,
        "predicted_static_force_n": _STIFFNESS_BORE * max(axial_error_at_peak, 0.0),
        "pre_contact_ee_speed_mm_s": speed_mm_s(ee),
        "pre_contact_cmd_speed_mm_s": speed_mm_s(cmd),
        "force_rise_ms": (peak - onset) / _CONTROL_HZ * 1e3,
        "onset_frac": onset / len(force),
    }


def episode_forensics(path: Path) -> dict | None:
    with np.load(path, allow_pickle=True) as z:
        meta = json.loads(str(z["metadata"]))
        wrist_ft = z["wrist_ft"].astype(float)
        ee = z["ee_pose"][:, :3].astype(float)
        cmd = z["cmd_position"].astype(float)
        hole = z["target_hole_pose"].astype(float)

    force = np.linalg.norm(wrist_ft[:, :3], axis=1)
    insertion_axis = axis_from_quat(hole[int(np.argmax(force)), 3:], 0)
    return contact_forensics(force, ee, cmd, insertion_axis, meta.get("terminal_reason", "unknown"))


def print_forensics_table(rows: list[dict]) -> None:
    no_contact = [r for r in rows if not r["contact"]]
    if no_contact:
        print(
            f"  {len(no_contact)} episodes never sustained baseline+"
            f"{_CONTACT_ONSET_ABOVE_BASELINE_N}N (excluded)"
        )
    rows = [r for r in rows if r["contact"]]

    metrics = [
        ("peak_force_n", "peak |F| (N)"),
        ("axial_error_at_peak_mm", "axial cmd-ee err @peak (mm)"),
        ("axial_error_max_mm", "axial cmd-ee err max (mm)"),
        ("predicted_static_force_n", "K*err predicted force (N)"),
        ("pre_contact_ee_speed_mm_s", "pre-contact ee speed (mm/s)"),
        ("pre_contact_cmd_speed_mm_s", "pre-contact cmd speed (mm/s)"),
        ("force_rise_ms", "onset->peak rise (ms)"),
        ("onset_frac", "onset position in episode"),
    ]
    by_outcome: dict[str, list[dict]] = {}
    for r in rows:
        by_outcome.setdefault(r["outcome"], []).append(r)

    header = f"{'metric':<32}" + "".join(
        f"{f'{o} (n={len(v)})':>26}" for o, v in sorted(by_outcome.items())
    )
    print("\nmedian [IQR] by terminal_reason")
    print(header)
    print("-" * len(header))
    for key, label in metrics:
        cells = []
        for _, group in sorted(by_outcome.items()):
            arr = np.array([g[key] for g in group], dtype=float)
            arr = arr[~np.isnan(arr)]
            cells.append(
                f"{np.median(arr):.3g} [{np.percentile(arr, 25):.2g},{np.percentile(arr, 75):.2g}]"
            )
        print(f"{label:<32}" + "".join(f"{c:>26}" for c in cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded", default=str(ROOT / "data" / "recorded" / "runs"))
    args = parser.parse_args()

    paths = sorted(Path(args.recorded).glob("episode_*/episode.npz"))
    rows = [r for r in (episode_forensics(p) for p in paths) if r is not None]
    print(f"{len(rows)} episodes ({args.recorded})")
    print_forensics_table(rows)


if __name__ == "__main__":
    main()
