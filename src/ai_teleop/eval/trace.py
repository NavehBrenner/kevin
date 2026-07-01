"""Eval-trace schema + writer/reader — the realized-state log evaluation records (LAB-37).

An *eval trace* is the per-tick **realized environment state** of one evaluation
episode, persisted so the KPIs can be (re)computed **offline** — without re-running
the episode. It is the producer half of the eval producer/consumer split; the
consumer is :class:`~ai_teleop.eval.observer.TrialObserver`, which reads the trace
back and computes a :class:`~ai_teleop.eval.schema.TrialKPIs` exactly as it would
live (the observer reads only an ``Observation``, so live and replay are the same
calculator — see ``concepts/passive-observer-evaluation`` in the wiki).

**Why a separate schema from the M4 corpus** (``data.trajectory.EpisodeRecorder``)?
The corpus shares the *pattern* (validated NPZ columns) but not the *meaning*: it
stores ``wrist_ft`` **bias-subtracted**, its ``delta_*`` columns are the *expert BC
target*, and it carries privileged ``distance``/``step_success``. An eval trace must
log the **raw** wrench (the observer does its own per-trial tare, mirroring the
deployed policy) and the **assist-under-test** Δ (NoAssist or the residual), and owns
no success notion — that lives only in ``eval/``. Sharing the corpus schema would
re-couple ``eval/`` to data-gen's BC semantics, the opposite of the DIP split this
milestone exists to keep. Same pattern, two honest schemas.

Per-step columns (all world-frame; metres, quaternion (w,x,y,z), newtons):

================  ========  ================================================
column            shape     meaning
================  ========  ================================================
step              ()        control-step index (0-based)
sim_time          ()        seconds since reset
joint_positions   (7,)      arm joint angles
joint_velocities  (7,)      arm joint velocities
ee_pose           (7,)      TCP pose (px,py,pz,qw,qx,qy,qz)
wrist_ft          (6,)      wrist wrench, **RAW** (not bias-subtracted)
gripper_width     ()        finger opening (m)
peg_pose          (7,)      true peg body pose (privileged — eval only)
target_hole_pose  (7,)      active target hole pose (privileged — eval only)
base_cmd_position (3,)      operator (pre-Δ) command position
base_cmd_quat     (4,)      operator command orientation
base_cmd_grip     ()        operator command Δgrip force
delta_position    (3,)      assist-under-test Δ position
delta_orientation (3,)      assist-under-test Δ orientation (axis-angle)
delta_grip        ()        assist-under-test Δ grip force
================  ========  ================================================

``base_cmd_*`` is logged so the paired-seed design can be *verified* (identical
operator stream across configs) and ``delta_*`` so the residual's contribution can be
inspected — neither is needed for the KPIs, which are pure functions of the state.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ai_teleop.common.command import Command
from ai_teleop.common.observation import Observation
from ai_teleop.domain.delta import Delta

# 1.0: the eval-trace layout is independent of the M4 corpus ``SCHEMA_VERSION`` —
# they are deliberately separate contracts.
SCHEMA_VERSION = "1.0"

TRACE_NPZ_NAME = "trace.npz"

# Per-step columns and their per-step shape (() == scalar). The writer validates
# every row against this set so a logging bug fails loud, not silent.
COLUMN_SHAPES: dict[str, tuple[int, ...]] = {
    "step": (),
    "sim_time": (),
    "joint_positions": (7,),
    "joint_velocities": (7,),
    "ee_pose": (7,),
    "wrist_ft": (6,),
    "gripper_width": (),
    "peg_pose": (7,),
    "target_hole_pose": (7,),
    "base_cmd_position": (3,),
    "base_cmd_quat": (4,),
    "base_cmd_grip": (),
    "delta_position": (3,),
    "delta_orientation": (3,),
    "delta_grip": (),
}


class EvalTraceRecorder:
    """Accumulates per-step realized state, then writes one NPZ trace file.

    Drop it into ``run_episode`` as (part of) the ``step_callback`` and call
    :meth:`record` each tick with the same pre-step arguments the observer sees::

        recorder = EvalTraceRecorder()
        # ... in the step_callback: recorder.record(observation, base_command, delta)
        recorder.save(path, metadata={"seed": 7, "config_label": "residual"})
    """

    def __init__(self, target_hole_index: int = 0) -> None:
        self._rows: list[dict[str, np.ndarray]] = []
        self._target_hole_index = target_hole_index

    def record(self, observation: Observation, base_command: Command, delta: Delta) -> None:
        """Append one realized-state row from the per-tick objects."""
        target_hole_pose = observation.hole_poses[self._target_hole_index]
        self._add(
            step=len(self._rows),
            sim_time=observation.sim_time,
            joint_positions=observation.joint_positions,
            joint_velocities=observation.joint_velocities,
            ee_pose=observation.ee_pose,
            wrist_ft=observation.wrist_ft,  # RAW — observer tares per trial
            gripper_width=observation.gripper_width,
            peg_pose=observation.peg_pose,
            target_hole_pose=target_hole_pose,
            base_cmd_position=base_command.target_position,
            base_cmd_quat=base_command.target_quaternion,
            base_cmd_grip=base_command.delta_grip_force,
            delta_position=delta.delta_position,
            delta_orientation=delta.delta_orientation,
            delta_grip=delta.delta_grip_force,
        )

    def _add(self, **fields: object) -> None:
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

    def save(self, path: str | Path, metadata: dict[str, Any]) -> None:
        if not self._rows:
            raise ValueError("cannot save an empty trace")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = {key: np.stack([row[key] for row in self._rows]) for key in COLUMN_SHAPES}
        full_metadata = {"schema_version": SCHEMA_VERSION, "n_steps": len(self._rows), **metadata}
        arrays = {"metadata": np.array(json.dumps(full_metadata)), **columns}
        np.savez_compressed(path, allow_pickle=False, **arrays)  # type: ignore[arg-type]


def load_eval_trace(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read one NPZ eval trace back into (columns, metadata)."""
    with np.load(path, allow_pickle=False) as data:
        metadata: dict[str, Any] = json.loads(str(data["metadata"]))
        columns = {key: data[key] for key in COLUMN_SHAPES}
    return columns, metadata


def replay_trace(
    columns: dict[str, np.ndarray],
) -> Iterator[tuple[int, Observation, Command, Delta]]:
    """Reconstruct the per-tick ``(step, observation, base_command, delta)`` stream.

    Yields exactly what the ``step_callback`` saw live, so a :class:`TrialObserver`
    driven over this stream computes the identical KPIs. Only the *target* hole is
    logged, so the reconstructed ``Observation`` carries a one-row ``hole_poses`` —
    drive the replaying observer with ``target_hole_index=0`` (its default) so the
    seating geometry reads that single logged hole.
    """
    n_steps = int(columns["step"].shape[0])
    for index in range(n_steps):
        observation = Observation(
            joint_positions=columns["joint_positions"][index],
            joint_velocities=columns["joint_velocities"][index],
            ee_pose=columns["ee_pose"][index],
            wrist_ft=columns["wrist_ft"][index],
            gripper_width=float(columns["gripper_width"][index]),
            peg_pose=columns["peg_pose"][index],
            hole_poses=columns["target_hole_pose"][index][None, :],
            sim_time=float(columns["sim_time"][index]),
        )
        base_command = Command(
            columns["base_cmd_position"][index],
            columns["base_cmd_quat"][index],
            float(columns["base_cmd_grip"][index]),
        )
        delta = Delta(
            columns["delta_position"][index],
            columns["delta_orientation"][index],
            float(columns["delta_grip"][index]),
        )
        yield int(columns["step"][index]), observation, base_command, delta
