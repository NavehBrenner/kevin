"""M4 data-generation pipeline — produce the behavioral-cloning corpus.

Core functionality (the `scripts/generate_dataset.py` CLI is just its front
door). Runs N unattended episodes (a fresh per-episode wall → realistic noisy
human → analytical expert → controller → sim) and writes **one per-episode
folder** under ``<dataset>/runs/`` — ``episode_NNNNN/episode.npz`` plus an
``imgs/`` subfolder. This is the BC training corpus M5 trains against.

The per-tick loop itself lives in `ai_teleop.sim.runner.run_episode`
(logging-free); this pipeline bolts logging on through its ``step_callback``
hook, detects the episode's terminal condition (insertion depth → success,
force-cap → abort, timeout → failure), and keeps **all** episodes (failures
included — diverse state coverage helps BC). Every episode is reproducible from
``(seed, episode_index)``.

**Paired human-only baseline.** For each episode the pipeline also re-runs the
*same scene and the same operator command stream* with the expert replaced by
``NoAssist`` (no residual correction), scored with the identical termination
logic but **not** persisted as a trajectory. The resulting "what would the noisy
human achieve on its own" success rate quantifies the expert's actual lift; it
is recorded per-episode in the trajectory metadata and aggregated in the dataset
summary. Disable with ``baseline=False`` (roughly halves wall-clock).

Output layout (one directory per master seed; one sub-directory per episode)::

    data/dataset_<seed>/
        metadata.json              # dataset-level statistics
        runs/
            episode_00000/
                episode.npz        # per-episode trajectory (the BC corpus)
                imgs/              # per-step wrist-cam frames (empty unless
                                   # render_images; vision is M7)
            episode_00001/
                ...

The on-disk schema is the stable contract M5 reads — see
`ai_teleop.data.trajectory` / `ai_teleop.data.schema` and `docs/data-schema.md`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ai_teleop.common.log import get_logger
from ai_teleop.control import Controller
from ai_teleop.data.schema import DatasetConfig, ResBCDatasetMetadata
from ai_teleop.data.step_callbacks import EpisodeLogger, TerminationProbe
from ai_teleop.data.trajectory import (
    SCHEMA_VERSION,
    TerminalReason,
    episode_imgs_dir,
    episode_npz_path,
    load_episode,
)
from ai_teleop.domain import NoAssist
from ai_teleop.expert import Expert
from ai_teleop.input import ScriptedNoisyHuman
from ai_teleop.sim.config import EnvConfig, episode_wall_seed
from ai_teleop.sim.env_setup import make_env
from ai_teleop.sim.runner import run_episode
from ai_teleop.sim.scene import SimEnv
from ai_teleop.sim.scene_source import STATIC_TASK_SCENE

log = get_logger("generate")

SCENE_PATH = STATIC_TASK_SCENE  # static 3-hole task wall — the no-generated-walls scene

# Both generated walls and our static-scene runs place the goal at hole_0. Which
# hole is the target is the task layer's choice (the env just reports every hole's
# pose); data generation always aims at hole_0.
_TARGET_HOLE_INDEX = 0

# Marker recorded as the dataset's `scene` when walls are procedurally generated
# per episode (there is no single scene file). The static escape hatch records the
# actual scene-file name instead.
_GENERATED_SCENE_LABEL = "generated"

# ~18 s @ 500 Hz. Raised 6000 → 9000 by LAB-100: under the operator speed
# draw's slow tail the peg arrives late and the expert timeouts run out of
# clock mid-seating (LAB-100 probe: every dataset_8-config timeout had already
# entered the expert's d_far band — late arrivals, not non-arrivals). The
# extra 6 s converts most of them (timeouts 20% → 8% at n=40, aborts
# unchanged) without disturbing the LAB-95/96 realism anchor (baseline
# force-aborts hold at ~50%). Comparability caveat: the recorded reference
# sessions ran 5000 steps (10 s).
DEFAULT_MAX_STEPS = 9000
DEFAULT_SUCCESS_DEPTH = 0.015  # insertion past the hole entry → success (m)
DEFAULT_LATERAL_TOLERANCE = 0.010  # max lateral error for a "seated" peg (m); LAB-77 calibration
DEFAULT_FORCE_CAP = 50.0  # wrist force magnitude that aborts the episode (N)
# Distance (m) at which the expert starts engaging. Widened 0.10 → 0.15 by
# LAB-98: under the deployment controller config the extra 5 cm is braking
# window (aborts 8% → 5%, success 72% → 75% at n=40; 0.20 adds nothing).
# Under kd=4 this knob was byte-identical 0.10–0.20 (LAB-77) — the effect is
# specific to the responsive controller. Kept as small as achieves the ceiling:
# d_far bounds the region where Phase-1 labels are non-zero.
DEFAULT_EXPERT_D_FAR = 0.15

# Controller config for the corpus (LAB-95/96): the DEPLOYMENT (teleop) config —
# what `run_episode.py --input vision` runs and what `data/recorded` was captured
# under — NOT the Controller's own careful-insertion defaults (kd=4.0, clamp
# 0.025). Training data must match the contact dynamics the policy deploys into:
# under kd=4 the free-space slew pins at ~0.05 m/s, erasing the episode-to-episode
# speed variance that IS the recorded force-abort signature (LAB-95 root cause).
DEFAULT_MAX_DPOS = 0.3  # controller command clamp (m/step)
DEFAULT_JOINT_DAMPING = 1.5  # flat joint-space kd (N·m·s/rad)

# Per-episode lognormal draw on the operator's max_approach_speed (LAB-95/96):
# median calibrated against `data/recorded` under the deployment controller
# config (0.09 m/s landed 47.5% force-aborts / 2.77x motion tail at 40 seeds);
# sigma 0.76 fits the recorded near-field command-speed spread (p90/median ~2.7).
DEFAULT_SPEED_LOGNORMAL_MEDIAN = 0.09
DEFAULT_SPEED_LOGNORMAL_SIGMA = 0.76

# Expert approach-speed brake (LAB-98): under the deployment controller config
# the arm tracks the operator's command tightly, so a hasty episode slams the
# wall at its drawn sweep speed — the kd=4-tuned expert corrected aim but not
# approach speed (dataset_7: expert force-aborts 5% → 28%). The brake retracts
# the command's axial lead beyond `gain * distance + floor`, decelerating the
# approach. Calibrated by the LAB-98 sweep (scripts/dev/
# lab98_expert_recalibration_sweep.py, n=40, master_seed 950): gain is monotone
# 0 → 1.0 (aborts 28% → 8%) and degrades by 1.5 (brake too weak); floor 8 mm
# beat 5/12 mm. Together with the widened d_far (below): expert 75% / aborts 5%.
DEFAULT_EXPERT_BRAKE_GAIN = 1.0
DEFAULT_EXPERT_BRAKE_LEAD_FLOOR = 0.008

# Shared expert/policy per-step Δ-position bound for the corpus (LAB-100). The
# expert clamps its label to this, and it is what bounds the brake's authority
# — the structural residual LAB-98 measured (operator sweeps faster than the
# clamp can absorb still crash). Raised 0.02 → 0.03 by the LAB-100 sweep
# (n=40 × two seed families, 9000-step budget): expert aborts 22% → 15% on
# the corpus family with success-episode clamp saturation at zero; 0.04
# measured the same ceiling within noise, so the smaller bound is kept.
# Matches domain.delta._MAX_DELTA_POSITION (the deployed bound).
DEFAULT_DELTA_CLAMP = 0.03

# The pre-LAB-96 corpus config (the Controller's careful-insertion defaults, no
# per-episode speed draw). Fingerprints and `regenerate_from_metadata` treat this
# as the implicit config of metadata written before these knobs existed, so
# legacy datasets keep regenerating byte-identical with matching fingerprints.
_LEGACY_JOINT_DAMPING = 4.0
_LEGACY_SPEED_LOGNORMAL_MEDIAN = 0.0
# Pre-LAB-98: no expert brake (gain 0 disables it inside `Expert`).
_LEGACY_EXPERT_BRAKE_GAIN = 0.0
# Pre-LAB-100: the ±2 cm Δ-position bound every earlier corpus was clamped at
# (the `domain.delta` module bound of the time).
_LEGACY_DELTA_CLAMP = 0.02


def _episode_fingerprint(
    *,
    seed: int,
    max_steps: int,
    max_dpos: float,
    expert_d_far: float,
    generated_walls: bool,
    joint_damping: float = _LEGACY_JOINT_DAMPING,
    speed_lognormal_median: float = _LEGACY_SPEED_LOGNORMAL_MEDIAN,
    speed_lognormal_sigma: float = DEFAULT_SPEED_LOGNORMAL_SIGMA,
    expert_brake_gain: float = _LEGACY_EXPERT_BRAKE_GAIN,
    expert_brake_lead_floor: float = DEFAULT_EXPERT_BRAKE_LEAD_FLOOR,
    delta_clamp: float = _LEGACY_DELTA_CLAMP,
) -> str:
    """Stable hash of every input that determines an episode's trajectory.

    Two runs with the same fingerprint produce byte-identical episodes, so a
    cached file carrying this fingerprint can be reused instead of re-simulated.
    The per-episode wall is derived deterministically from ``(seed, episode_index)``,
    so ``seed`` (hashed here) + the episode index (in the file path) already pin it;
    only the wall *mode* (``generated_walls``) needs to enter the hash.

    The LAB-96 knobs extend the payload only when they leave the legacy config
    (kd=4.0, no speed draw — behavior-identical to pre-LAB-96 code, RNG
    included), so a legacy dataset's committed fingerprint still matches its
    regeneration. The LAB-98 expert-brake knobs follow the same pattern (gain 0
    == the brake-free pre-LAB-98 expert, bit-exact), as does the LAB-100
    Δ-clamp (the legacy ±2 cm bound == pre-LAB-100 behavior, bit-exact).
    """
    import hashlib

    payload = f"{seed}|{max_steps}|{max_dpos:.6f}|{expert_d_far:.6f}|{generated_walls}"
    if joint_damping != _LEGACY_JOINT_DAMPING or speed_lognormal_median > 0.0:
        payload += f"|{joint_damping:.6f}|{speed_lognormal_median:.6f}|{speed_lognormal_sigma:.6f}"
    if expert_brake_gain > 0.0:
        payload += f"|{expert_brake_gain:.6f}|{expert_brake_lead_floor:.6f}"
    if delta_clamp != _LEGACY_DELTA_CLAMP:
        payload += f"|clamp{delta_clamp:.6f}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _cached_matches(path: Path, fingerprint: str) -> bool:
    """True if ``path`` is a readable episode whose stored fingerprint matches."""
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
    except (OSError, ValueError, KeyError):
        return False  # unreadable / truncated → regenerate
    return bool(metadata.get("fingerprint") == fingerprint)


def _human_seed(seed: int, episode_index: int) -> int:
    """The concrete RNG seed handed to the scripted human for this episode.

    Derived from ``(master_seed, episode_index)`` so it is reproducible, but the
    integer itself isn't obvious — so it's stamped into the per-episode metadata.
    """
    return int(np.random.SeedSequence([seed, episode_index]).generate_state(1)[0])


def _make_human(
    target_position: np.ndarray,
    home_quaternion: np.ndarray,
    *,
    seed: int,
    episode_index: int,
    speed_lognormal_median: float,
    speed_lognormal_sigma: float,
) -> ScriptedNoisyHuman:
    """Build the per-episode operator. ``(seed, episode_index)`` plus the speed-
    draw config fully determine its command stream (the per-episode approach-speed
    draw comes from the operator's own seeded RNG), so a fresh instance reproduces
    the same operator — what the paired expert/baseline runs rely on."""
    return ScriptedNoisyHuman(
        np.concatenate([target_position, home_quaternion]),
        seed=_human_seed(seed, episode_index),
        speed_lognormal_median=speed_lognormal_median,
        speed_lognormal_sigma=speed_lognormal_sigma,
    )


def _run_baseline(
    environment: SimEnv,
    controller: Controller,
    human: ScriptedNoisyHuman,
    *,
    target_hole_index: int,
    max_steps: int,
    success_depth: float,
    lateral_tolerance: float,
    force_cap: float,
) -> tuple[TerminalReason, int]:
    """Re-run the same scene + operator with ``NoAssist`` (no expert), scoring only — no
    trajectory is recorded. Returns ``(terminal_reason, n_steps)``: how the human-only run
    ended and how long it took, so the dataset can measure the expert's insertion-time win
    (baseline steps-to-terminate vs the expert episode's ``n_steps``)."""
    controller.reset()  # clear any lock the expert run left latched
    probe = TerminationProbe(
        controller,
        target_hole_index=target_hole_index,
        success_depth=success_depth,
        lateral_tolerance=lateral_tolerance,
        force_cap=force_cap,
    )
    result = run_episode(
        environment, controller, human, NoAssist(), max_steps=max_steps, step_callback=probe
    )
    return probe.terminal_reason, result.n_steps


def generate_dataset(
    out_dir: str | Path,
    n_episodes: int,
    *,
    seed: int = 0,
    max_steps: int = DEFAULT_MAX_STEPS,
    success_depth: float = DEFAULT_SUCCESS_DEPTH,
    lateral_tolerance: float = DEFAULT_LATERAL_TOLERANCE,
    force_cap: float = DEFAULT_FORCE_CAP,
    max_dpos: float = DEFAULT_MAX_DPOS,
    joint_damping: float = DEFAULT_JOINT_DAMPING,
    expert_d_far: float = DEFAULT_EXPERT_D_FAR,
    expert_brake_gain: float = DEFAULT_EXPERT_BRAKE_GAIN,
    expert_brake_lead_floor: float = DEFAULT_EXPERT_BRAKE_LEAD_FLOOR,
    speed_lognormal_median: float = DEFAULT_SPEED_LOGNORMAL_MEDIAN,
    speed_lognormal_sigma: float = DEFAULT_SPEED_LOGNORMAL_SIGMA,
    delta_clamp: float = DEFAULT_DELTA_CLAMP,
    generated_walls: bool = True,
    cache: bool = True,
    baseline: bool = True,
    render_images: bool = False,
    render_every: int = 1,
    progress: bool = False,
) -> list[Path]:
    """Generate a dataset under ``out_dir``; return the episode-file paths.

    Writes ``out_dir/runs/episode_NNNNN/episode.npz`` (the BC corpus) — one
    folder per episode, each with an ``imgs/`` subfolder — plus
    ``out_dir/metadata.json`` (dataset-level statistics). Keeps every episode
    (success or failure). Each is reproducible from ``(seed, episode_index)``
    plus the controller/expert config: the per-episode wall and the noisy human
    both derive from the seed.

    The controller runs the **deployment (teleop) config** by default
    (``joint_damping=1.5, max_dpos=0.3`` — what live vision teleop and the
    ``data/recorded`` reference corpus run), not the Controller's own
    careful-insertion defaults, and the operator draws a per-episode
    ``max_approach_speed`` from a lognormal (``speed_lognormal_*``) — the
    LAB-95/96 recipe that closes the scripted-vs-recorded force-abort and
    motion-tail gaps. All of these are fingerprinted corpus parameters. When ``cache`` is set, an existing episode file
    whose stored ``fingerprint`` matches the current config is reused instead of
    being re-simulated.

    With ``generated_walls`` (the default) each episode runs on its own freshly
    built procedural wall (a fresh, clean ``SimEnv`` per episode, seeded
    deterministically from ``(seed, episode_index)``) — genuine per-episode wall
    diversity. ``generated_walls=False`` runs every episode on the static
    hand-authored wall instead (no ``scenegen``/CadQuery), varying only the
    operator; useful for fast, dependency-light tests.

    When ``baseline`` is set, each episode is additionally re-run with the expert
    replaced by ``NoAssist`` (same scene, same operator) and scored — *not*
    saved — to measure the human-only success rate the expert improves on.

    When ``render_images`` is set, the wrist camera is rendered every
    ``render_every`` recorded steps and saved as PNGs in each episode's ``imgs/``
    folder. This is opt-in M7 (vision) plumbing — off by default, so the M5
    F/T-only corpus is unchanged and ``imgs/`` stays empty.
    """
    out_dir = Path(out_dir)
    runs_dir = out_dir / "runs"
    expert = Expert(
        d_far=expert_d_far,
        brake_gain=expert_brake_gain,
        brake_lead_floor=expert_brake_lead_floor,
        target_hole_index=_TARGET_HOLE_INDEX,
        # Explicit per-corpus bound (not the module default): regenerating a
        # legacy dataset must clamp the expert at the bound it was recorded
        # under, whatever the deployed bound is today (LAB-100).
        max_delta_position=delta_clamp,
    )
    thresholds = dict(
        success_depth=success_depth, lateral_tolerance=lateral_tolerance, force_cap=force_cap
    )
    fingerprint = _episode_fingerprint(
        seed=seed,
        max_steps=max_steps,
        max_dpos=max_dpos,
        expert_d_far=expert_d_far,
        generated_walls=generated_walls,
        joint_damping=joint_damping,
        speed_lognormal_median=speed_lognormal_median,
        speed_lognormal_sigma=speed_lognormal_sigma,
        expert_brake_gain=expert_brake_gain,
        expert_brake_lead_floor=expert_brake_lead_floor,
        delta_clamp=delta_clamp,
    )

    written: list[Path] = []
    summaries: list[dict[str, object]] = []
    frames_rendered = 0
    render_wall_time = 0.0
    for episode_index in range(n_episodes):
        path = episode_npz_path(runs_dir, episode_index)
        if cache and _cached_matches(path, fingerprint):
            written.append(path)
            summaries.append(_summary_from_cache(path, baseline=baseline))
            if progress:
                log.info("episode %5d │ ✓ loaded from cache", episode_index)
            continue

        # A fresh, clean env per episode: its own wall (generated ⇒ a distinct
        # procedural wall per index; static ⇒ the same hand-authored wall every
        # time). The env owns physics only; reset() restores its home state.
        wall_seed = episode_wall_seed(seed, episode_index) if generated_walls else None
        environment = make_env(EnvConfig(wall_seed=wall_seed), render_mode="headless")
        controller = Controller(
            environment, max_dpos_per_step=max_dpos, joint_damping=joint_damping
        )
        home_quaternion = controller.home_pose[3:]
        observation = environment.reset()
        target_position = observation.hole_poses[_TARGET_HOLE_INDEX][:3].copy()
        ft_bias = observation.wrist_ft.copy()

        # Establish the episode's imgs/ folder so the per-episode layout is
        # uniform whether or not frames are rendered (M7 fills it; M5 leaves it
        # empty). recorder.save() creates the episode folder itself.
        imgs_dir = episode_imgs_dir(runs_dir, episode_index)
        imgs_dir.mkdir(parents=True, exist_ok=True)

        human = _make_human(
            target_position,
            home_quaternion,
            seed=seed,
            episode_index=episode_index,
            speed_lognormal_median=speed_lognormal_median,
            speed_lognormal_sigma=speed_lognormal_sigma,
        )
        logger = EpisodeLogger(
            ft_bias,
            controller,
            target_hole_index=_TARGET_HOLE_INDEX,
            **thresholds,
            render_fn=environment.render_wrist_camera if render_images else None,
            imgs_dir=imgs_dir,
            render_every=render_every,
        )
        run_episode(
            environment, controller, human, expert, max_steps=max_steps, step_callback=logger
        )
        frames_rendered += logger.frames_rendered
        render_wall_time += logger.render_wall_time

        baseline_reason: TerminalReason | None = None
        baseline_n_steps: int | None = None
        if baseline:
            baseline_reason, baseline_n_steps = _run_baseline(
                environment,
                controller,
                # fresh operator with the same seed ⇒ identical command stream
                _make_human(
                    target_position,
                    home_quaternion,
                    seed=seed,
                    episode_index=episode_index,
                    speed_lognormal_median=speed_lognormal_median,
                    speed_lognormal_sigma=speed_lognormal_sigma,
                ),
                target_hole_index=_TARGET_HOLE_INDEX,
                max_steps=max_steps,
                **thresholds,
            )

        environment.close()

        episode_metadata: dict[str, object] = {
            # Base commands came from the scripted noisy human, corrected by the expert —
            # so a replay logs source=scripted (not "unknown") and can note the recorded policy.
            "source": "scripted",
            "policy": "expert",
            "master_seed": seed,
            "episode_index": episode_index,
            # scene_seed roots the per-episode derivations in (master_seed,
            # episode_index); human_seed is the concrete int seeding the scripted
            # operator; wall_seed (when generated) is the env's procedural wall.
            "scene_seed": [seed, episode_index],
            "human_seed": _human_seed(seed, episode_index),
            "fingerprint": fingerprint,
            "max_dpos": max_dpos,
            "joint_damping": joint_damping,
            "speed_lognormal_median": speed_lognormal_median,
            "speed_lognormal_sigma": speed_lognormal_sigma,
            "expert_d_far": expert_d_far,
            "expert_brake_gain": expert_brake_gain,
            "expert_brake_lead_floor": expert_brake_lead_floor,
            "delta_clamp": delta_clamp,
            "target_hole_index": _TARGET_HOLE_INDEX,
            "generated_wall": generated_walls,
            "wall_seed": wall_seed,
            "terminal_reason": logger.terminal_reason.value,
            "episode_success": logger.terminal_reason is TerminalReason.SUCCESS,
            "success_depth": success_depth,
            "lateral_tolerance": lateral_tolerance,
            "force_cap": force_cap,
        }
        if baseline_reason is not None:
            episode_metadata["baseline_terminal_reason"] = baseline_reason.value
            episode_metadata["baseline_success"] = baseline_reason is TerminalReason.SUCCESS
            episode_metadata["baseline_n_steps"] = baseline_n_steps
        logger.recorder.save(path, metadata=episode_metadata)

        written.append(path)
        summaries.append(_episode_summary(path, episode_metadata, n_steps=len(logger.recorder)))
        if progress:
            tail = f" · baseline {baseline_reason.value}" if baseline_reason is not None else ""
            log.info(
                "episode %5d │ generated · %5d steps · %s%s",
                episode_index,
                len(logger.recorder),
                logger.terminal_reason.value,
                tail,
            )

    config: DatasetConfig = {
        "max_steps": max_steps,
        "max_dpos": max_dpos,
        "joint_damping": joint_damping,
        "expert_d_far": expert_d_far,
        "expert_brake_gain": expert_brake_gain,
        "expert_brake_lead_floor": expert_brake_lead_floor,
        "speed_lognormal_median": speed_lognormal_median,
        "speed_lognormal_sigma": speed_lognormal_sigma,
        "delta_clamp": delta_clamp,
        "success_depth": success_depth,
        "lateral_tolerance": lateral_tolerance,
        "force_cap": force_cap,
        "scene": _GENERATED_SCENE_LABEL if generated_walls else SCENE_PATH.name,
    }
    _write_dataset_metadata(
        out_dir, summaries, seed=seed, fingerprint=fingerprint, baseline=baseline, config=config
    )
    if render_images and frames_rendered:
        log.info(
            "rendered %d frames in %.1fs (%.1f frames/s)",
            frames_rendered,
            render_wall_time,
            frames_rendered / render_wall_time,
        )
    return written


def _episode_summary(
    path: Path, episode_metadata: Mapping[str, object], *, n_steps: int
) -> dict[str, object]:
    """Compact per-episode entry for the dataset ``metadata.json`` (an
    ``EpisodeSummary`` shape; see ``data.schema``)."""
    summary: dict[str, object] = {
        "episode_index": episode_metadata["episode_index"],
        "file": f"runs/{path.parent.name}/{path.name}",
        "n_steps": n_steps,
        "target_hole_index": episode_metadata["target_hole_index"],
        # .get for back-compat with cached files written before seeds were stamped.
        "scene_seed": episode_metadata.get("scene_seed"),
        "human_seed": episode_metadata.get("human_seed"),
        "terminal_reason": episode_metadata["terminal_reason"],
        "success": episode_metadata["episode_success"],
    }
    if "baseline_terminal_reason" in episode_metadata:
        summary["baseline_terminal_reason"] = episode_metadata["baseline_terminal_reason"]
        summary["baseline_success"] = episode_metadata["baseline_success"]
    return summary


def _summary_from_cache(path: Path, *, baseline: bool) -> dict[str, object]:
    """Rebuild a per-episode summary from a cached episode file's metadata."""
    columns, metadata = load_episode(path)
    # metadata is JSON-loaded (values typed `object`); n_steps is an int on disk
    # but fall back to the column length if an older file omitted it.
    raw_n_steps = metadata.get("n_steps")
    n_steps = raw_n_steps if isinstance(raw_n_steps, int) else len(columns["step"])
    summary = _episode_summary(path, metadata, n_steps=n_steps)
    if baseline and "baseline_terminal_reason" not in metadata:
        # Cached file predates the baseline; mark unknown rather than fabricating
        # (this propagates to a null aggregate baseline rate in metadata.json).
        summary["baseline_terminal_reason"] = None
        summary["baseline_success"] = None
    return summary


def _rate(summaries: list[dict[str, object]], key: str) -> tuple[dict[str, int], float | None]:
    """Counts-by-terminal-reason and success rate over a ``*_terminal_reason`` /
    ``*success`` pair; rate is ``None`` if any episode is missing the field."""
    reason_key = "terminal_reason" if key == "success" else "baseline_terminal_reason"
    counts: dict[str, int] = {}
    successes = 0
    for summary in summaries:
        reason = summary.get(reason_key)
        if reason is None:
            return counts, None
        counts[str(reason)] = counts.get(str(reason), 0) + 1
        if summary.get(key):
            successes += 1
    rate = successes / len(summaries) if summaries else None
    return counts, rate


def _write_dataset_metadata(
    dataset_dir: Path,
    summaries: list[dict[str, object]],
    *,
    seed: int,
    fingerprint: str,
    baseline: bool,
    config: DatasetConfig,
) -> None:
    """Write ``dataset_dir/metadata.json`` with dataset-level statistics
    (a ``ResBCDatasetMetadata`` shape; see ``data.schema``)."""
    expert_counts, expert_rate = _rate(summaries, "success")
    metadata: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "master_seed": seed,
        "n_episodes": len(summaries),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "fingerprint": fingerprint,
        "config": config,
        "expert": {"counts": expert_counts, "success_rate": expert_rate},
        "episodes": summaries,
    }
    if baseline:
        baseline_counts, baseline_rate = _rate(summaries, "baseline_success")
        metadata["baseline_no_assist"] = {
            "counts": baseline_counts,
            "success_rate": baseline_rate,
        }
        if expert_rate is not None and baseline_rate is not None:
            metadata["expert_lift"] = round(expert_rate - baseline_rate, 6)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def regenerate_from_metadata(
    metadata_path: str | Path,
    *,
    out_dir: str | Path | None = None,
    force: bool = False,
    progress: bool = False,
) -> list[Path]:
    """Reproduce the dataset described by a committed ``metadata.json``.

    Only the metadata is version-controlled; the episode trajectories are not.
    This reads every trajectory-determining input back out of the file and
    re-runs generation, writing the episodes next to the metadata (or to
    ``out_dir``). Generation is deterministic in those inputs, so the regenerated
    episodes are byte-identical to the originals — verified afterwards via the
    shared ``fingerprint`` (a mismatch flags code or config drift).

    This is also the metadata-driven gap-filler the loader's ``download=True``
    path uses: with ``force=False`` the cache skips episodes already present, so
    only missing ones are simulated.

    The baseline is re-run iff the source metadata recorded one, so the refreshed
    ``metadata.json`` reproduces the original statistics (modulo ``generated_at``).
    """
    metadata_path = Path(metadata_path)
    metadata: ResBCDatasetMetadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = metadata["config"]

    # Walls are reproduced from their seeds, not loaded from disk: the `scene`
    # label records whether this dataset used per-episode generated walls or the
    # static one. Anything other than the static scene-file name means generated.
    generated_walls = config["scene"] != SCENE_PATH.name

    # Metadata written before LAB-96 carries no controller/speed-draw keys — it
    # was generated under the legacy config, so regenerate under it. Same for
    # the LAB-98 expert-brake keys (absent ⇒ brake off).
    joint_damping = config.get("joint_damping", _LEGACY_JOINT_DAMPING)
    speed_lognormal_median = config.get("speed_lognormal_median", _LEGACY_SPEED_LOGNORMAL_MEDIAN)
    speed_lognormal_sigma = config.get("speed_lognormal_sigma", DEFAULT_SPEED_LOGNORMAL_SIGMA)
    expert_brake_gain = config.get("expert_brake_gain", _LEGACY_EXPERT_BRAKE_GAIN)
    expert_brake_lead_floor = config.get("expert_brake_lead_floor", DEFAULT_EXPERT_BRAKE_LEAD_FLOOR)
    # Absent ⇒ pre-LAB-100 corpus, clamped at the legacy ±2 cm bound.
    delta_clamp = config.get("delta_clamp", _LEGACY_DELTA_CLAMP)

    target = Path(out_dir) if out_dir is not None else metadata_path.parent
    written = generate_dataset(
        target,
        metadata["n_episodes"],
        seed=metadata["master_seed"],
        max_steps=config["max_steps"],
        success_depth=config["success_depth"],
        lateral_tolerance=config["lateral_tolerance"],
        force_cap=config["force_cap"],
        max_dpos=config["max_dpos"],
        joint_damping=joint_damping,
        expert_d_far=config["expert_d_far"],
        expert_brake_gain=expert_brake_gain,
        expert_brake_lead_floor=expert_brake_lead_floor,
        speed_lognormal_median=speed_lognormal_median,
        speed_lognormal_sigma=speed_lognormal_sigma,
        delta_clamp=delta_clamp,
        generated_walls=generated_walls,
        cache=not force,
        baseline="baseline_no_assist" in metadata,
        progress=progress,
    )

    expected = metadata.get("fingerprint")
    actual = _episode_fingerprint(
        seed=metadata["master_seed"],
        max_steps=config["max_steps"],
        max_dpos=config["max_dpos"],
        expert_d_far=config["expert_d_far"],
        generated_walls=generated_walls,
        joint_damping=joint_damping,
        speed_lognormal_median=speed_lognormal_median,
        speed_lognormal_sigma=speed_lognormal_sigma,
        expert_brake_gain=expert_brake_gain,
        expert_brake_lead_floor=expert_brake_lead_floor,
        delta_clamp=delta_clamp,
    )
    if expected is not None and actual != expected:
        log.warning(
            "fingerprint mismatch (metadata %s != regenerated %s); "
            "the regenerated episodes may differ from the originals.",
            expected,
            actual,
        )
    return written
