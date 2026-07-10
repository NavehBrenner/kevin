"""LAB-105 Stage-C decision probe: is the true hole position *linearly decodable*
from the FROZEN image encoder's features?

The DAgger null (LAB-105) was read as a **perception ceiling**: low BC loss + flat
task success + healthy label coverage → the frozen encoder can't localize the hole
well enough to act on the (correct) label. That diagnosis was *inferred*, never
measured. This probe measures it directly and cheaply — no training, no rollouts.

Method: the deployed policy uses a *frozen* mobilenet_v3_small backbone, so its
image features are exactly the ImageNet init (freeze ⇒ weights never updated). For a
sample of rendered wrist frames we extract that backbone's 576-d pooled feature and
ridge-regress it onto the hole position **in the end-effector frame** (what the wrist
camera actually sees: rotate ``hole_world − ee_world`` by the inverse EE quaternion).
Held out by episode (frames within an episode are correlated), we report R² and the
RMS position error in cm — overall and for the near-hole frames (d < d_far = 0.15 m)
that are the ones the reactive expert actually labels.

Read-out:
- **Good decode** (high R², low-cm error), esp. near the hole ⇒ the frozen features
  DO carry the hole location; perception is *not* the ceiling → the lever is data /
  DAgger rounds, not Stage C.
- **Poor decode** (near-zero R², large error) ⇒ the frozen ImageNet features can't
  localize the hole; no linear projection or GRU downstream can recover it →
  **Stage C** (unfreeze the backbone) is the confirmed lever.

    uv run python scripts/dev/lab105_perception_probe.py data/dataset_vision --episodes 80
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from ai_teleop.data.images import decode_frames, discover_frames
from ai_teleop.data.trajectory import load_episode
from ai_teleop.policy.config import PolicyConfig
from ai_teleop.policy.image_encoder import _build_backbone

D_FAR = 0.15  # m — the reactive expert is zero beyond this; near-hole = the labeled regime


def _ee_frame_target(ee_pose: np.ndarray, hole_pose: np.ndarray) -> np.ndarray:
    """Hole position expressed in the end-effector frame (metres), per row.

    ``*_pose`` are ``(N, 7)`` world-frame ``(px,py,pz, qw,qx,qy,qz)``. The wrist cam
    is rigid to the EE, so the visually-grounded target is the world offset rotated
    into the EE frame: ``R(q_ee)^{-1} · (hole_pos − ee_pos)``.
    """
    offset_world = hole_pose[:, :3] - ee_pose[:, :3]
    # scipy quat order is (x,y,z,w); our columns are (w,x,y,z).
    rotation = Rotation.from_quat(ee_pose[:, [4, 5, 6, 3]])
    return rotation.inv().apply(offset_world)


def _collect(dataset: Path, n_episodes: int, device: str) -> tuple[np.ndarray, ...]:
    """Extract frozen-backbone features + EE-frame hole targets for sampled frames.

    Returns ``(features, targets, distance, episode_id)`` stacked over all frames of
    the first ``n_episodes`` episodes that have rendered frames.
    """
    backbone, _ = _build_backbone(PolicyConfig(use_vision=True, image_pretrained=True))
    backbone = backbone.to(device).eval()

    episode_dirs = sorted((dataset / "runs").glob("episode_*"))
    features_all, targets_all, distance_all, episode_all = [], [], [], []
    used = 0
    for episode_dir in episode_dirs:
        if used >= n_episodes:
            break
        frames = discover_frames(episode_dir / "imgs")
        if not frames:
            continue
        columns, _ = load_episode(episode_dir / "episode.npz")

        # Map each rendered frame's step to its trajectory row (steps are the 0-based
        # control-step index; searchsorted is robust if any step is missing).
        steps = columns["step"].astype(int)
        frame_steps = np.array([step for step, _ in frames])
        rows = np.searchsorted(steps, frame_steps)
        rows = np.clip(rows, 0, len(steps) - 1)

        target = _ee_frame_target(columns["ee_pose"][rows], columns["target_hole_pose"][rows])
        images = decode_frames([path for _, path in frames]).to(device)
        with torch.no_grad():
            feats = backbone(images).cpu().numpy()  # (F, 576)

        features_all.append(feats)
        targets_all.append(target)
        distance_all.append(columns["distance"][rows])
        episode_all.append(np.full(len(frames), used))
        used += 1

    print(f"collected {sum(len(f) for f in features_all)} frames from {used} episodes")
    return (
        np.concatenate(features_all),
        np.concatenate(targets_all),
        np.concatenate(distance_all),
        np.concatenate(episode_all),
    )


def _ridge_r2(
    features: np.ndarray,
    targets: np.ndarray,
    episode_id: np.ndarray,
    ridge_lambda: float = 10.0,
    test_fraction: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit standardized ridge feats→targets, held out by episode.

    Returns ``(r2_per_axis, rmse_cm_per_axis)`` on the held-out frames. Splitting by
    *episode* (not by frame) prevents leakage from within-episode frame correlation.
    ``rmse_cm`` for a dead decoder ≈ the target's own std (in cm).
    """
    unique_episodes = np.unique(episode_id)
    rng = np.random.default_rng(0)
    rng.shuffle(unique_episodes)
    n_test = max(1, int(len(unique_episodes) * test_fraction))
    test_episodes = set(unique_episodes[:n_test].tolist())
    is_test = np.array([e in test_episodes for e in episode_id])

    x_train, x_test = features[~is_test], features[is_test]
    y_train, y_test = targets[~is_test], targets[is_test]

    # Standardize features and center targets on the train split (closed-form ridge).
    mean, std = x_train.mean(0), x_train.std(0) + 1e-8
    x_train_z, x_test_z = (x_train - mean) / std, (x_test - mean) / std
    y_mean = y_train.mean(0)
    yc = y_train - y_mean

    gram = x_train_z.T @ x_train_z + ridge_lambda * np.eye(x_train_z.shape[1])
    weights = np.linalg.solve(gram, x_train_z.T @ yc)  # (576, 3)
    prediction = x_test_z @ weights + y_mean

    residual = y_test - prediction
    ss_res = (residual**2).sum(0)
    ss_tot = ((y_test - y_test.mean(0)) ** 2).sum(0)
    r2 = 1 - ss_res / ss_tot
    rmse_cm = 100 * np.sqrt((residual**2).mean(0))
    return r2, rmse_cm


def _report(label: str, r2: np.ndarray, rmse_cm: np.ndarray) -> None:
    axes = "xyz"
    print(f"\n[{label}]")
    for i, axis in enumerate(axes):
        print(f"  {axis}: R²={r2[i]:+.3f}  RMSE={rmse_cm[i]:.2f} cm")
    print(f"  mean R²={r2.mean():+.3f}  mean RMSE={rmse_cm.mean():.2f} cm")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--ridge-lambda", type=float, default=10.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    features, targets, distance, episode_id = _collect(args.dataset, args.episodes, args.device)

    # Overall decode.
    r2, rmse_cm = _ridge_r2(features, targets, episode_id, args.ridge_lambda)
    _report("all frames", r2, rmse_cm)

    # Near-hole decode: the frames the reactive expert actually labels (d < d_far).
    near = distance < D_FAR
    print(f"\nnear-hole frames (d < {D_FAR} m): {near.sum()} / {len(near)}")
    if near.sum() > 50 and len(np.unique(episode_id[near])) >= 3:
        r2n, rmse_cmn = _ridge_r2(
            features[near], targets[near], episode_id[near], args.ridge_lambda
        )
        _report("near-hole frames", r2n, rmse_cmn)
    else:
        print("  too few near-hole frames/episodes for a clean split — skipping")

    # Baseline scale: the target's own spread (a dead decoder's RMSE ≈ this).
    spread_cm = 100 * targets.std(0)
    print(f"\ntarget spread (dead-decoder RMSE): {spread_cm.round(2)} cm per axis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
