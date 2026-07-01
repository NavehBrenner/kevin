"""On-disk metadata contracts — the JSON shapes the dataset reads and writes.

This is the ``data`` package's type-contract module (the role ``domain/interfaces``
plays for the assistance seam): pure structural definitions, no behavior and no
imports from the implementation, so anything may depend on it without a cycle.
It holds the type contracts for everything the dataset serializes: the per-step
**column arrays** of an ``episode.npz`` (``EpisodeColumns`` — exactly what
``load_episode`` returns) and the JSON **metadata** blobs (the per-episode
``metadata`` key and the dataset-level ``metadata.json``). The *runtime* column
contract — the ``COLUMN_SHAPES`` shape map and the NPZ reader/writer — and the
per-episode-folder layout stay in ``trajectory.py``.

These are the Python equivalent of a TypeScript ``interface`` over JSON: a
``TypedDict`` is a plain ``dict`` at runtime (no validation, no cost) but lets the
type checker know which keys exist and their types — exactly the structural
contract a TS interface gives you. Optional keys (the paired ``baseline_*``
fields, present only when the human-only baseline was run) use the inheritance +
``total=False`` pattern, since ``typing.NotRequired`` is 3.11+ and the project
targets 3.10.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np


class EpisodeColumns(TypedDict):
    """The per-step arrays inside an ``episode.npz`` — one stacked ``(T, …)``
    ``float64`` array per column, exactly the dict ``load_episode`` returns.

    Keys and their per-step shapes are the runtime contract in
    ``trajectory.COLUMN_SHAPES``; this is their static type, so loaders get
    key-checked, typed access (``columns["wrist_ft"]`` etc.). Scalar columns are
    stored as ``(T,)``. ``peg_pose`` / ``target_hole_pose`` / ``distance`` are
    privileged ground truth — offline analysis only, never fed to a deployed policy.
    """

    step: np.ndarray
    sim_time: np.ndarray
    wrist_ft: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    ee_pose: np.ndarray
    gripper_width: np.ndarray
    cmd_position: np.ndarray
    cmd_quaternion: np.ndarray
    cmd_grip: np.ndarray
    delta_position: np.ndarray
    delta_orientation: np.ndarray
    delta_grip: np.ndarray
    peg_pose: np.ndarray
    target_hole_pose: np.ndarray
    distance: np.ndarray
    step_success: np.ndarray


class DatasetConfig(TypedDict):
    """The generation knobs that define a dataset, echoed into ``metadata.json``.

    Every trajectory-determining input lives here, so the dataset is
    reproducible from the metadata file alone (see ``regenerate_from_metadata``).
    """

    max_steps: int
    max_dpos: float
    expert_d_far: float
    success_depth: float
    lateral_tolerance: float
    force_cap: float
    scene: str  # scene-file *name* (resolved against the mjcf assets dir)


class _EpisodeMetadataBase(TypedDict):
    """Required keys of a per-episode ``episode.npz`` ``metadata`` blob."""

    schema_version: str
    n_steps: int
    master_seed: int
    episode_index: int
    scene_seed: list[int]  # [master_seed, episode_index]
    human_seed: int
    fingerprint: str
    max_dpos: float
    expert_d_far: float
    target_hole_index: int
    terminal_reason: str  # a TerminalReason value
    episode_success: bool
    success_depth: float
    lateral_tolerance: float
    force_cap: float


class EpisodeMetadata(_EpisodeMetadataBase, total=False):
    """Per-episode metadata; the ``baseline_*`` keys appear iff a baseline ran."""

    baseline_terminal_reason: str | None
    baseline_success: bool | None

    # Replay spec — the scene + controller recipe so ``kvn episode --input <ep>``
    # rebuilds the exact episode it ran in (see scripts/run_episode.py
    # _rebuild_for_replay). Recorded episodes stamp these; generated episodes carry
    # generated_wall/wall_seed already and derive the rest from generation defaults.
    # force_cap None ⇒ the force-cap watchdog was off (--no-force-cap).
    policy: str
    seed: int
    generated_wall: bool
    wall_seed: int | None
    distractors: int | None
    joint_damping: float
    scene: str  # scene-file name


class _EpisodeSummaryBase(TypedDict):
    """Required keys of one entry in the dataset-level ``episodes`` list."""

    episode_index: int
    file: str  # dataset-relative path, e.g. "runs/episode_00000/episode.npz"
    n_steps: int
    target_hole_index: int
    scene_seed: list[int] | None
    human_seed: int | None
    terminal_reason: str
    success: bool


class EpisodeSummary(_EpisodeSummaryBase, total=False):
    """Compact per-episode entry in ``metadata.json``; ``baseline_*`` optional."""

    baseline_terminal_reason: str | None
    baseline_success: bool | None


class _RateBlock(TypedDict):
    """A ``{counts-by-terminal-reason, success_rate}`` aggregate block."""

    counts: dict[str, int]
    success_rate: float | None


class _ResBCDatasetMetadataBase(TypedDict):
    """Required keys of the dataset-level ``metadata.json``."""

    schema_version: str
    master_seed: int
    n_episodes: int
    generated_at: str  # ISO-8601 UTC
    fingerprint: str
    config: DatasetConfig
    expert: _RateBlock
    episodes: list[EpisodeSummary]


class ResBCDatasetMetadata(_ResBCDatasetMetadataBase, total=False):
    """Dataset-level metadata; the baseline aggregates appear iff a baseline ran.

    This is the residual-BC dataset's manifest — the single committed artifact a
    loader reads to discover, verify, and (if missing) regenerate the corpus.
    """

    baseline_no_assist: _RateBlock
    expert_lift: float
