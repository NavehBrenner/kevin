"""DAgger — close the BC imitation gap by on-policy expert relabel (LAB-105).

The M7 headline is blocked by a behavioral-cloning **covariate-shift** gap, not a
corpus mismatch: the clone drifts into states the privileged expert never
demonstrated and its corrections there are confidently wrong (deployed 20% vs
expert-ceiling 65% on the eval walls). DAgger is the one idea that fixes this —
**let the policy act, so it visits its own drift states, and query the expert for
the correct label at those states** — then aggregate those relabeled states into
the corpus and retrain. Batched form: rollout → relabel → aggregate → retrain,
repeated a few rounds.

This module owns the *new* mechanism only; everything else is reused:

* **Rollout + relabel** (:func:`rollout_and_relabel`) drives the shared
  ``sim.runner.run_episode`` with the learned policy as the acting ``assist`` and
  the analytical :class:`~ai_teleop.expert.Expert` as the *label provider* on
  ``data.step_callbacks.EpisodeLogger`` — so each visited state is recorded with
  the expert's correction as the BC target (the on-policy relabel). The vision
  policy's own rendered wrist frame is saved as-is (no second render).
* **Aggregation** (:func:`seed_aggregate` / :func:`append_summaries`) grows a
  dataset dir whose manifest unions the seed corpus (symlinked in, no copy) with
  each round's relabeled episodes — DAgger's data aggregation, in the exact
  on-disk schema ``data.dataset`` already loads.
* **Retrain / re-ablate** (:func:`run_dagger`) shells the aggregate through the
  existing ``train_policy`` and ``eval`` paths unchanged.

DAgger episodes roll out on a **distinct wall family** (``rollout_master_seed``,
default 105) from both the corpus (seed 82) and the held-out eval walls (seed 0),
so the eval walls stay clean and the rounds add wall diversity on top of the
on-policy states.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ai_teleop.common.log import get_logger
from ai_teleop.control import Controller
from ai_teleop.data.generate import (
    DEFAULT_EXPERT_BRAKE_GAIN,
    DEFAULT_EXPERT_BRAKE_LEAD_FLOOR,
    DEFAULT_EXPERT_D_FAR,
    DEFAULT_FORCE_CAP,
    DEFAULT_JOINT_DAMPING,
    DEFAULT_LATERAL_TOLERANCE,
    DEFAULT_MAX_DPOS,
    DEFAULT_MAX_STEPS,
    DEFAULT_SPEED_LOGNORMAL_MEDIAN,
    DEFAULT_SPEED_LOGNORMAL_SIGMA,
    DEFAULT_SUCCESS_DEPTH,
    _episode_summary,
)
from ai_teleop.data.schema import DatasetConfig
from ai_teleop.data.step_callbacks import EpisodeLogger
from ai_teleop.data.trajectory import (
    SCHEMA_VERSION,
    TerminalReason,
    episode_imgs_dir,
    episode_npz_path,
)
from ai_teleop.domain.interfaces import AssistProvider
from ai_teleop.expert import Expert
from ai_teleop.input import ScriptedNoisyHuman
from ai_teleop.sim.config import EnvConfig, episode_wall_seed
from ai_teleop.sim.env_setup import make_env
from ai_teleop.sim.runner import run_episode

log = get_logger("dagger")

_TARGET_HOLE_INDEX = 0

# Wall family for the on-policy rollouts — distinct from the corpus (82) and the
# held-out eval walls (0), so aggregation adds new walls and eval stays clean.
DEFAULT_ROLLOUT_MASTER_SEED = 105

# DAgger episode indices live far above any corpus index so they never collide
# with the symlinked seed corpus (0..N). round r, rollout i → this + r*BLOCK + i.
_DAGGER_INDEX_BASE = 1_000_000
_DAGGER_ROUND_BLOCK = 10_000


def _human_seed(master_seed: int, episode_index: int) -> int:
    """Deterministic per-episode operator seed — the shared ``(master, index)``
    derivation used by data generation and the eval harness."""
    return int(np.random.SeedSequence([master_seed, episode_index]).generate_state(1)[0])


def dagger_episode_index(round_index: int, rollout_index: int) -> int:
    """Collision-free episode index for a DAgger-relabeled episode."""
    return _DAGGER_INDEX_BASE + round_index * _DAGGER_ROUND_BLOCK + rollout_index


def expert_from_config(config: Mapping[str, Any]) -> Expert:
    """Rebuild the corpus's expert from its ``metadata.json`` config, so the
    relabels are drawn from the *same* teacher the corpus was cloned from."""
    return Expert(
        target_hole_index=_TARGET_HOLE_INDEX,
        d_far=float(config.get("expert_d_far", DEFAULT_EXPERT_D_FAR)),
        brake_gain=float(config.get("expert_brake_gain", DEFAULT_EXPERT_BRAKE_GAIN)),
        brake_lead_floor=float(
            config.get("expert_brake_lead_floor", DEFAULT_EXPERT_BRAKE_LEAD_FLOOR)
        ),
        max_delta_position=float(config.get("delta_clamp", 0.03)),
    )


def rollout_and_relabel(
    *,
    policy: AssistProvider,
    expert: AssistProvider,
    runs_dir: Path,
    dagger_index: int,
    master_seed: int,
    rollout_index: int,
    config: Mapping[str, Any],
    render_every: int | None,
    generated_walls: bool = True,
) -> dict[str, object]:
    """Run one on-policy rollout, relabel every visited state with ``expert``, and
    write it as ``runs_dir/episode_<dagger_index>/`` in the corpus schema.

    ``policy`` is the acting assist (the current learned residual); ``expert`` is
    the label provider whose correction becomes the BC target. With
    ``render_every`` set (an int) the env's wrist capture is enabled at that cadence
    so a vision policy can act, and the saved frames are the ones it saw; ``None``
    leaves capture off (F/T-only — no frames, for fast tests). Returns the episode's
    manifest summary.
    """
    max_dpos = float(config.get("max_dpos", DEFAULT_MAX_DPOS))
    joint_damping = float(config.get("joint_damping", DEFAULT_JOINT_DAMPING))
    max_steps = int(config.get("max_steps", DEFAULT_MAX_STEPS))
    thresholds = dict(
        success_depth=float(config.get("success_depth", DEFAULT_SUCCESS_DEPTH)),
        lateral_tolerance=float(config.get("lateral_tolerance", DEFAULT_LATERAL_TOLERANCE)),
        force_cap=float(config.get("force_cap", DEFAULT_FORCE_CAP)),
    )

    wall_seed = episode_wall_seed(master_seed, rollout_index) if generated_walls else None
    environment = make_env(EnvConfig(wall_seed=wall_seed), render_mode="headless")
    try:
        controller = Controller(
            environment, max_dpos_per_step=max_dpos, joint_damping=joint_damping
        )
        home_quaternion = controller.home_pose[3:]
        observation = environment.reset()
        target_position = observation.hole_poses[_TARGET_HOLE_INDEX][:3].copy()
        ft_bias = observation.wrist_ft.copy()

        # The vision policy needs a live wrist frame each control step; enable the
        # env's rate-limited capture and save that same frame (no second render).
        if render_every is not None:
            environment.enable_wrist_capture(render_every)

        human = ScriptedNoisyHuman(
            np.concatenate([target_position, home_quaternion]),
            seed=_human_seed(master_seed, rollout_index),
            speed_lognormal_median=float(
                config.get("speed_lognormal_median", DEFAULT_SPEED_LOGNORMAL_MEDIAN)
            ),
            speed_lognormal_sigma=float(
                config.get("speed_lognormal_sigma", DEFAULT_SPEED_LOGNORMAL_SIGMA)
            ),
        )
        if hasattr(policy, "reset"):
            policy.reset()  # fresh GRU hidden state + F/T bias for this episode

        imgs_dir = episode_imgs_dir(runs_dir, dagger_index)
        imgs_dir.mkdir(parents=True, exist_ok=True)
        logger = EpisodeLogger(
            ft_bias,
            controller,
            target_hole_index=_TARGET_HOLE_INDEX,
            success_depth=thresholds["success_depth"],
            lateral_tolerance=thresholds["lateral_tolerance"],
            force_cap=thresholds["force_cap"],
            label_provider=expert,
            save_observation_frame=render_every is not None,
            imgs_dir=imgs_dir if render_every is not None else None,
            render_every=render_every or 1,
        )
        run_episode(
            environment, controller, human, policy, max_steps=max_steps, step_callback=logger
        )

        path = episode_npz_path(runs_dir, dagger_index)
        episode_metadata: dict[str, object] = {
            "source": "dagger",
            "policy": "learned_residual",
            "master_seed": master_seed,
            "episode_index": dagger_index,
            "scene_seed": [master_seed, rollout_index],
            "human_seed": _human_seed(master_seed, rollout_index),
            "fingerprint": "dagger",  # rollout-derived, not seed-regenerable
            "max_dpos": max_dpos,
            "joint_damping": joint_damping,
            "target_hole_index": _TARGET_HOLE_INDEX,
            "generated_wall": generated_walls,
            "wall_seed": wall_seed,
            "terminal_reason": logger.terminal_reason.value,
            "episode_success": logger.terminal_reason is TerminalReason.SUCCESS,
            **thresholds,
        }
        logger.recorder.save(path, metadata=episode_metadata)
        return _episode_summary(path, episode_metadata, n_steps=len(logger.recorder))
    finally:
        environment.close()


def seed_aggregate(base_dir: str | Path, aggregate_dir: str | Path) -> list[dict[str, object]]:
    """Seed the aggregate corpus from ``base_dir`` and return its episode summaries.

    The seed corpus's episode folders are **symlinked** (not copied) into
    ``aggregate/runs/`` — the loader resolves the dataset-relative ``file`` paths
    through the links, so a 300-episode image corpus costs 300 symlinks, not a
    multi-GB copy. The aggregate ``metadata.json`` starts as the base manifest;
    :func:`append_summaries` extends its ``episodes`` list per round.
    """
    base_dir = Path(base_dir)
    aggregate_dir = Path(aggregate_dir)
    aggregate_runs = aggregate_dir / "runs"
    aggregate_runs.mkdir(parents=True, exist_ok=True)

    base_metadata = json.loads((base_dir / "metadata.json").read_text(encoding="utf-8"))
    for summary in base_metadata["episodes"]:
        name = Path(summary["file"]).parent.name  # episode_NNNNN
        source = (base_dir / "runs" / name).resolve()
        link = aggregate_runs / name
        if not link.exists():
            os.symlink(source, link)

    (aggregate_dir / "metadata.json").write_text(json.dumps(base_metadata, indent=2) + "\n")
    return list(base_metadata["episodes"])


def append_summaries(
    aggregate_dir: str | Path,
    all_summaries: list[dict[str, object]],
    *,
    config: DatasetConfig | Mapping[str, object] | None = None,
) -> None:
    """Rewrite the aggregate manifest's ``episodes`` list to ``all_summaries``.

    ``all_summaries`` is the full union (seed + every round so far), so this is
    idempotent per round. The dataset loader reads only ``episodes`` (+ counts),
    so nothing else needs to change for the retrain to see the aggregated corpus.
    """
    aggregate_dir = Path(aggregate_dir)
    metadata = json.loads((aggregate_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata["episodes"] = all_summaries
    metadata["n_episodes"] = len(all_summaries)
    metadata["schema_version"] = SCHEMA_VERSION
    if config is not None:
        metadata["config"] = dict(config)
    (aggregate_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def run_dagger(
    *,
    base_dir: str | Path,
    checkpoint: str | Path,
    aggregate_dir: str | Path,
    runs_root: str | Path = "outputs/policy/runs",
    rounds: int = 1,
    n_rollout: int = 40,
    rollout_master_seed: int = DEFAULT_ROLLOUT_MASTER_SEED,
    render_every: int = 20,
    device: str = "cuda",
    epochs: int = 40,
    batch_size: int = 2,
    action_rate_weight: float = 100.0,
    eval_seeds: int = 20,
    eval_master_seed: int = 0,
    error_scale: float = 1.0,
) -> list[dict[str, object]]:
    """Full batched-DAgger loop; returns a per-round result record.

    Each round: roll out ``n_rollout`` episodes of the current policy (relabeled by
    the corpus expert) onto the aggregate, retrain the frozen-encoder vision policy
    on the union at ``action_rate_weight`` (the Stage-A smoothness win), re-ablate
    on the held-out eval walls, and carry the new checkpoint into the next round.
    Reuses ``train_policy`` and the eval harness unchanged.
    """
    # Heavy / torch-only imports are lazy so the sim-only rollout path (and its
    # tests) need neither torch nor a checkpoint on disk.
    import subprocess
    import sys

    from ai_teleop.policy import LearnedResidual

    train_script = Path(__file__).resolve().parents[2] / "scripts" / "train_policy.py"

    base_dir = Path(base_dir)
    aggregate_dir = Path(aggregate_dir)
    config = json.loads((base_dir / "metadata.json").read_text(encoding="utf-8"))["config"]
    expert = expert_from_config(config)

    all_summaries = seed_aggregate(base_dir, aggregate_dir)
    current_checkpoint = Path(checkpoint)
    results: list[dict[str, object]] = []

    for round_index in range(rounds):
        log.info(
            "round %d │ rolling out %d episodes with %s", round_index, n_rollout, current_checkpoint
        )
        policy = LearnedResidual.from_checkpoint(current_checkpoint, device=device)
        successes = 0
        for rollout_index in range(n_rollout):
            summary = rollout_and_relabel(
                policy=policy,
                expert=expert,
                runs_dir=aggregate_dir / "runs",
                dagger_index=dagger_episode_index(round_index, rollout_index),
                master_seed=rollout_master_seed,
                rollout_index=rollout_index,
                config=config,
                render_every=render_every,
            )
            all_summaries.append(summary)
            successes += int(bool(summary["success"]))
            log.info(
                "  rollout %3d │ %6d steps │ %s",
                rollout_index,
                summary["n_steps"],
                summary["terminal_reason"],
            )
        append_summaries(aggregate_dir, all_summaries, config=config)
        log.info(
            "round %d │ policy rollout success %d/%d │ aggregate now %d episodes",
            round_index,
            successes,
            n_rollout,
            len(all_summaries),
        )

        run_name = f"dagger_round{round_index}"
        subprocess.run(
            [
                sys.executable,
                str(train_script),
                str(aggregate_dir),
                "--vision",
                "--freeze-image-encoder",
                "--action-rate-weight",
                str(action_rate_weight),
                "--epochs",
                str(epochs),
                "--batch-size",
                str(batch_size),
                "--num-workers",
                "4",
                "--device",
                device,
                "--runs-root",
                str(runs_root),
                "--name",
                run_name,
            ],
            check=True,
        )
        current_checkpoint = Path(runs_root) / run_name / "checkpoint.pt"

        ablation = _reablate(
            current_checkpoint,
            seeds=eval_seeds,
            master_seed=eval_master_seed,
            error_scale=error_scale,
            device=device,
        )
        log.info(
            "round %d │ eval @ error_scale %.2f │ human %.0f%% │ vision %.0f%%",
            round_index,
            error_scale,
            100 * ablation["human_only"],
            100 * ablation["vision"],
        )
        results.append({
            "round": round_index,
            "checkpoint": str(current_checkpoint),
            "rollout_success": successes / n_rollout if n_rollout else None,
            "aggregate_episodes": len(all_summaries),
            **ablation,
        })
    return results


def _reablate(
    checkpoint: str | Path,
    *,
    seeds: int,
    master_seed: int,
    error_scale: float,
    device: str,
) -> dict[str, float]:
    """Paired human-only vs vision ablation on the held-out eval walls; returns
    each config's success rate. Thin reuse of ``eval.ablation.run_paired``."""
    from ai_teleop.eval.ablation import HUMAN_ONLY, Config, run_paired
    from ai_teleop.policy import LearnedResidual

    vision = Config(
        label="vision",
        assist_factory=lambda: LearnedResidual.from_checkpoint(checkpoint, device=device),
    )
    configs = [HUMAN_ONLY, vision]
    successes = {config.label: 0 for config in configs}
    for episode_index in range(seeds):
        results = run_paired(
            episode_index, configs, master_seed=master_seed, operator_error_scale=error_scale
        )
        for label, kpis in results.items():
            successes[label] += int(kpis.success)
    return {label: successes[label] / seeds for label in successes}
