"""Trajectory schema + per-episode writer/reader — the stable M5 contract (M4).

**One directory per episode.** Each episode is a folder
``runs/episode_NNNNN/`` holding the trajectory file ``episode.npz`` and an
``imgs/`` subfolder reserved for per-step wrist-camera frames (populated only
when generation is run with image rendering on — vision is M7, so ``imgs/`` is
normally empty). The per-episode-folder layout (vs. a flat ``episode_NNNNN.npz``)
is what lets a frame stream sit *beside* its trajectory without a second index.
Build these paths with the ``episode_dir`` / ``episode_npz_path`` /
``episode_imgs_dir`` helpers rather than formatting the names by hand.

The on-disk trajectory format is **NPZ** (numpy's ``savez``): no extra
dependencies, available everywhere CI runs, and accepted by the M5 dataset
loader (LAB-32). Each per-step column is stored as a stacked ``(T, …)`` array;
per-episode metadata is a JSON string under the ``metadata`` key.

The schema is the *only* thing M5 depends on, so it is versioned
(``SCHEMA_VERSION``) and documented in ``docs/data-schema.md``. Everything else
about data generation (noise magnitudes, gate constants, scene layout) is free to
change without breaking M5 — only the columns and their meanings are frozen.

Per-step columns (all world-frame; metres, radians-via-quaternion, newtons):

================  ========  ================================================
column            shape     meaning
================  ========  ================================================
step              ()        control-step index (0-based)
sim_time          ()        seconds since reset
wrist_ft          (6,)      wrist wrench, **bias-subtracted** (contact-only)
joint_positions   (7,)      arm joint angles
joint_velocities  (7,)      arm joint velocities
ee_pose           (7,)      TCP pose (px,py,pz,qw,qx,qy,qz)
gripper_width     ()        finger opening (m)
cmd_position      (3,)      operator command position (pre-Δ)
cmd_quaternion    (4,)      operator command orientation
cmd_grip          ()        operator command Δgrip force
delta_position    (3,)      expert Δ position  ── BC TARGET
delta_orientation (3,)      expert Δ orientation (axis-angle)  ── BC TARGET
delta_grip        ()        expert Δ grip force  ── BC TARGET
peg_pose          (7,)      PRIVILEGED true peg body pose
target_hole_pose  (7,)      PRIVILEGED true target-hole pose
distance          ()        PRIVILEGED tip→hole distance d
step_success      ()        bool — peg inserted at this step
================  ========  ================================================

The ``peg_pose`` / ``target_hole_pose`` / ``distance`` columns are privileged
ground truth — for offline analysis only, never fed to a deployed policy.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import cast

import numpy as np

from ai_teleop.data.schema import EpisodeColumns, EpisodeMetadata

# 2.0: on-disk *layout* changed from a flat ``runs/episode_NNNNN.npz`` to a
# per-episode folder ``runs/episode_NNNNN/{episode.npz, imgs/}``. The per-step
# column schema below is unchanged from 1.0.
SCHEMA_VERSION = "2.0"

# ---------------------------------------------------------------------------
# On-disk layout — one directory per episode
# ---------------------------------------------------------------------------

EPISODE_DIR_TEMPLATE = "episode_{:05d}"  # runs/episode_00000/
EPISODE_NPZ_NAME = "episode.npz"  # the trajectory file inside that folder
IMGS_DIRNAME = "imgs"  # per-step wrist-camera frames (reserved; M7)


def episode_dir(runs_dir: str | Path, episode_index: int) -> Path:
    """The per-episode folder ``<runs_dir>/episode_NNNNN/``."""
    return Path(runs_dir) / EPISODE_DIR_TEMPLATE.format(episode_index)


def episode_npz_path(runs_dir: str | Path, episode_index: int) -> Path:
    """The trajectory file ``<runs_dir>/episode_NNNNN/episode.npz``."""
    return episode_dir(runs_dir, episode_index) / EPISODE_NPZ_NAME


def episode_imgs_dir(runs_dir: str | Path, episode_index: int) -> Path:
    """The image subfolder ``<runs_dir>/episode_NNNNN/imgs/`` (frames; M7)."""
    return episode_dir(runs_dir, episode_index) / IMGS_DIRNAME


# Per-step columns and their per-step shape (() == scalar). The writer validates
# every appended row against this set so a logging bug fails loud, not silent.
COLUMN_SHAPES: dict[str, tuple[int, ...]] = {
    "step": (),
    "sim_time": (),
    "wrist_ft": (6,),
    "joint_positions": (7,),
    "joint_velocities": (7,),
    "ee_pose": (7,),
    "gripper_width": (),
    "cmd_position": (3,),
    "cmd_quaternion": (4,),
    "cmd_grip": (),
    "delta_position": (3,),
    "delta_orientation": (3,),
    "delta_grip": (),
    "peg_pose": (7,),
    "target_hole_pose": (7,),
    "distance": (),
    "step_success": (),
}


class TerminalReason(StrEnum):
    """Why an episode ended (stamped into per-episode metadata)."""

    SUCCESS = "success"  # insertion depth past threshold
    FORCE_ABORT = "force_abort"  # wrist force exceeded the cap
    TIMEOUT = "timeout"  # step budget reached without success


class EpisodeRecorder:
    """Accumulates per-step rows, then writes one NPZ episode file.

    Usage::

        recorder = EpisodeRecorder()
        for step in episode:
            recorder.add(step=..., wrist_ft=..., delta_position=..., ...)
        recorder.save(path, metadata={...})
    """

    def __init__(self) -> None:
        self._rows: list[dict[str, np.ndarray]] = []

    def add(self, **fields: object) -> None:
        missing = set(COLUMN_SHAPES) - set(fields)
        extra = set(fields) - set(COLUMN_SHAPES)
        if missing or extra:
            raise ValueError(f"row schema mismatch: missing={missing} extra={extra}")
        row = {key: np.asarray(value, dtype=np.float64) for key, value in fields.items()}
        for key, expected in COLUMN_SHAPES.items():
            if row[key].shape != expected:
                raise ValueError(f"column {key!r} expected shape {expected}, got {row[key].shape}")
        self._rows.append(row)

    def __len__(self) -> int:
        return len(self._rows)

    def save(self, path: str | Path, metadata: dict[str, object]) -> None:
        if not self._rows:
            raise ValueError("cannot save an empty episode")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = {key: np.stack([row[key] for row in self._rows]) for key in COLUMN_SHAPES}
        full_metadata = {"schema_version": SCHEMA_VERSION, "n_steps": len(self._rows), **metadata}
        arrays = {"metadata": np.array(json.dumps(full_metadata)), **columns}
        # mypy can't prove `arrays` won't carry an `allow_pickle` key (the stub's
        # only typed keyword); the keys are our own fixed column names.
        np.savez_compressed(path, allow_pickle=False, **arrays)  # type: ignore[arg-type]


def load_episode(path: str | Path) -> tuple[EpisodeColumns, EpisodeMetadata]:
    """Read one NPZ episode file back into (columns, metadata)."""
    with np.load(path, allow_pickle=False) as data:
        metadata: EpisodeMetadata = json.loads(str(data["metadata"]))
        columns = {key: data[key] for key in COLUMN_SHAPES}
    return cast(EpisodeColumns, columns), metadata
