"""LAB-106 delta-target audit — is the BC position target learnable at all?

The council's "one thing first": before any training, characterize the delta
*target* itself (no checkpoints, no rollouts). Four questions, all answered
offline from the recorded corpus columns:

A. **Frame / magnitude.** The delta is world-frame (`domain/delta.py`). Confirm
   the position label is a *lateral* (perp-to-bore) correction and report its
   per-axis spread — the zero-Δ baseline the policy must beat.

B. **Is the target ≈ −operator_bias?** At `joint_damping=1.5` the arm tracks the
   command tightly, so tip ≈ command, so the expert's lateral correction
   ≈ (hole − command) = −(bias+drift): a per-episode, zero-mean, unobservable-
   from-command quantity. Test tip≈command and the cross-episode bias spread.

C. **Linear probe: F/T-modality observables → delta target, held out by episode.**
   The non-vision analog of the LAB-105 perception probe. If lateral R²≈0 from
   [command, proprio, wrist_ft], then the F/T policy is *structurally* incapable
   of the lateral correction → "worse than zero" is expected, not a bug, and
   vision is the only modality that could help. A privileged probe (adds the true
   hole pose) is the sanity ceiling: the target IS a function of the hole.

D. **Hole world-position spread across walls** — does world-frame even vary?

Run from kevin/:  uv run python scripts/dev/lab106_delta_target_audit.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ai_teleop.common.log import configure_logging, get_logger
from ai_teleop.data.trajectory import load_episode

log = get_logger("lab106audit")

D_FAR = 0.15  # m — expert gate is zero beyond this; "active"/near-hole steps


def _insertion_axis(hole_quat: np.ndarray) -> np.ndarray:
    """Bore axis = hole local +x, per row. hole_quat cols are (w,x,y,z)."""
    return Rotation.from_quat(hole_quat[:, [1, 2, 3, 0]]).apply(np.array([1.0, 0.0, 0.0]))


def _lateral(vec: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Component of `vec` perpendicular to unit `axis`, per row."""
    axial = np.sum(vec * axis, axis=1, keepdims=True)
    return vec - axial * axis


def _ridge_r2_by_episode(
    features: np.ndarray, target: np.ndarray, episode_id: np.ndarray, lam: float = 10.0
) -> np.ndarray:
    """Per-axis held-out R² of a standardized ridge fit, split by episode."""
    episodes = np.unique(episode_id)
    rng = np.random.default_rng(0)
    rng.shuffle(episodes)
    n_test = max(1, int(0.3 * len(episodes)))
    test = set(episodes[:n_test].tolist())
    is_test = np.array([e in test for e in episode_id])

    x_tr, x_te = features[~is_test], features[is_test]
    y_tr, y_te = target[~is_test], target[is_test]
    mean, std = x_tr.mean(0), x_tr.std(0) + 1e-8
    x_tr, x_te = (x_tr - mean) / std, (x_te - mean) / std
    y_mean = y_tr.mean(0)
    gram = x_tr.T @ x_tr + lam * np.eye(x_tr.shape[1])
    weights = np.linalg.solve(gram, x_tr.T @ (y_tr - y_mean))
    pred = x_te @ weights + y_mean
    ss_res = ((y_te - pred) ** 2).sum(0)
    ss_tot = ((y_te - y_te.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / ss_tot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/dataset_vision"))
    parser.add_argument("--episodes", type=int, default=300)
    args = parser.parse_args()
    configure_logging()

    episode_dirs = sorted((args.dataset / "runs").glob("episode_*"))[: args.episodes]

    # Per-step pooled (active/near-hole only), plus per-episode aggregates.
    feat_obs, feat_priv, tgt_lat, tgt_full, epid = [], [], [], [], []
    tip_cmd_lat_all, zero_all_steps, hole_pos_per_ep, ep_lat_bias = [], [], [], []
    # ALL-steps (subsampled) set for the gating probe: obs → gated full delta.
    feat_all, tgt_all, epid_all = [], [], []
    rng_sub = np.random.default_rng(0)

    for i, ep in enumerate(episode_dirs):
        col, _ = load_episode(ep / "episode.npz")
        axis = _insertion_axis(col["target_hole_pose"][:, 3:])
        hole = col["target_hole_pose"][:, :3]
        cmd = col["cmd_position"]
        ee = col["ee_pose"][:, :3]
        delta = col["delta_position"]
        dist = col["distance"]

        obs_full = np.concatenate(
            [
                col["cmd_position"],
                col["cmd_quaternion"],
                col["cmd_grip"][:, None],
                col["ee_pose"],
                col["joint_positions"],
                col["joint_velocities"],
                col["gripper_width"][:, None],
                col["wrist_ft"],
            ],
            axis=1,
        )
        # zero-Δ baseline over ALL steps (matches the offline eval's masking).
        zero_all_steps.append(np.linalg.norm(delta, axis=1))
        hole_pos_per_ep.append(hole[0])
        # subsample all steps (~every 4th) for the gating probe.
        keep = rng_sub.random(len(delta)) < 0.25
        feat_all.append(obs_full[keep])
        tgt_all.append(delta[keep])
        epid_all.append(np.full(int(keep.sum()), i))

        active = dist < D_FAR
        if active.sum() < 5:
            continue

        # tip≈command check (lateral), near-hole.
        tip_cmd_lat_all.append(np.linalg.norm(_lateral(ee - cmd, axis)[active], axis=1))
        # per-episode lateral operator error (hole - command), near-hole mean.
        ep_lat_bias.append(_lateral(hole - cmd, axis)[active].mean(0))

        # Probe rows (active steps): observables the F/T policy actually sees.
        obs = obs_full[active]
        priv = np.concatenate([obs, hole[active], col["peg_pose"][active]], axis=1)

        feat_obs.append(obs)
        feat_priv.append(priv)
        tgt_full.append(delta[active])
        tgt_lat.append(_lateral(delta, axis)[active])
        epid.append(np.full(active.sum(), i))

    feat_obs = np.concatenate(feat_obs)
    feat_priv = np.concatenate(feat_priv)
    tgt_full = np.concatenate(tgt_full)
    tgt_lat = np.concatenate(tgt_lat)
    epid = np.concatenate(epid)
    zero_all = np.concatenate(zero_all_steps)
    tip_cmd_lat = np.concatenate(tip_cmd_lat_all)
    hole_pos = np.stack(hole_pos_per_ep)
    ep_bias = np.stack(ep_lat_bias)

    log.info("episodes used: %d | active near-hole steps pooled: %d", len(episode_dirs), len(epid))

    # A. frame / magnitude of the label.
    full_mag = np.linalg.norm(tgt_full, axis=1)
    lat_mag = np.linalg.norm(tgt_lat, axis=1)
    log.info(
        "[A] delta |lateral|/|full| on active steps: %.3f (─→ label is ~all lateral)",
        (lat_mag / (full_mag + 1e-9)).mean(),
    )
    log.info(
        "[A] zero-Δ baseline |delta| over ALL steps: %.2f mm (offline eval reported ~4.75)",
        1e3 * zero_all.mean(),
    )
    log.info(
        "[A] active-step delta per world axis std (mm): %s", np.round(1e3 * tgt_full.std(0), 2)
    )

    # B. tip≈command; target ≈ -bias.
    log.info(
        "[B] |ee - command| lateral, near-hole: mean %.2f mm, p90 %.2f mm  (small ⇒ tip≈cmd)",
        1e3 * tip_cmd_lat.mean(),
        1e3 * np.percentile(tip_cmd_lat, 90),
    )
    log.info(
        "[B] per-episode lateral operator error (hole-cmd): cross-ep std %s mm, |mean| %s mm",
        np.round(1e3 * ep_bias.std(0), 2),
        np.round(1e3 * np.abs(ep_bias.mean(0)), 2),
    )

    # C. THE probe: observables → delta target, held out by episode.
    r2_obs_lat = _ridge_r2_by_episode(feat_obs, tgt_lat, epid)
    r2_obs_full = _ridge_r2_by_episode(feat_obs, tgt_full, epid)
    r2_priv_lat = _ridge_r2_by_episode(feat_priv, tgt_lat, epid)
    log.info(
        "[C] F/T-observables → delta LATERAL, held-out R² per axis: %s (mean %.3f)",
        np.round(r2_obs_lat, 3),
        r2_obs_lat.mean(),
    )
    log.info(
        "[C] F/T-observables → delta FULL,    held-out R² per axis: %s (mean %.3f)",
        np.round(r2_obs_full, 3),
        r2_obs_full.mean(),
    )
    log.info(
        "[C] +PRIVILEGED hole/peg → delta LATERAL R² per axis: %s (mean %.3f)  [ceiling]",
        np.round(r2_priv_lat, 3),
        r2_priv_lat.mean(),
    )

    # C2. ALL-steps gating probe: can a linear model reproduce {0 far, correction
    # near} AND beat the zero-Δ baseline in mm? If yes, "worse than zero" is a
    # fitting failure of the trained net, not an unlearnable target.
    feat_all = np.concatenate(feat_all)
    tgt_all = np.concatenate(tgt_all)
    epid_all = np.concatenate(epid_all)
    r2_all = _ridge_r2_by_episode(feat_all, tgt_all, epid_all)
    # held-out mm error of the linear probe vs zero, on the all-steps holdout.
    eps = np.unique(epid_all)
    rng = np.random.default_rng(0)
    rng.shuffle(eps)
    test = set(eps[: max(1, int(0.3 * len(eps)))].tolist())
    is_te = np.array([e in test for e in epid_all])
    xtr, xte = feat_all[~is_te], feat_all[is_te]
    ytr, yte = tgt_all[~is_te], tgt_all[is_te]
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-8
    w = np.linalg.solve(
        ((xtr - mu) / sd).T @ ((xtr - mu) / sd) + 10 * np.eye(xtr.shape[1]),
        ((xtr - mu) / sd).T @ (ytr - ytr.mean(0)),
    )
    pred = ((xte - mu) / sd) @ w + ytr.mean(0)
    probe_mm = 1e3 * np.linalg.norm(pred - yte, axis=1).mean()
    zero_mm = 1e3 * np.linalg.norm(yte, axis=1).mean()
    log.info(
        "[C2] ALL-steps obs→gated-delta held-out R² per axis: %s (mean %.3f)",
        np.round(r2_all, 3),
        r2_all.mean(),
    )
    log.info(
        "[C2] ALL-steps held-out |err| mm: linear-probe %.2f  vs  zero-Δ %.2f  "
        "(offline eval: F/T 7.74)",
        probe_mm,
        zero_mm,
    )

    # D. hole world-position spread across walls.
    log.info(
        "[D] hole world-position across %d walls: std %s mm, range %s mm",
        len(hole_pos),
        np.round(1e3 * hole_pos.std(0), 1),
        np.round(1e3 * (hole_pos.max(0) - hole_pos.min(0)), 1),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
