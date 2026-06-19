"""Residual correction policy (vision-conditioned, BC-trained).

Outputs Δpose + Δgrip-force given current sensors and the noisy-human's command
(the hard safety clamp lives downstream in the seam, not here). Architecture
(Decision A, locked): a **single stateful GRU core over an early-fused
observation** — each control step the command, F/T, and proprioception streams
contribute their current (normalized) per-step value, concatenated into one input
vector and fed to one GRU that carries its hidden state across the whole episode
(reset per episode). The vector streams get no learned per-stream encoder; an MLP
head maps the per-step hidden state to the 7-D correction. Phase 2 only widens the
fused input with a fine-tuned image-CNN embedding — core and head are unchanged.

Trained offline against the expert by behavioral cloning (see ``expert``). See
``docs/design/policy-model.md`` for the full architecture and the locked encoder
decisions. ``ResidualPolicy`` (``model.py``) exposes a batched sequence
``forward`` for training and an O(1) per-tick ``step`` for deployment;
``LearnedResidual`` (``residual_policy.py``) wraps a trained checkpoint as the
runtime ``AssistProvider`` behind the M3 seam.

This package imports ``torch`` (the ``ml`` optional-dependency group), so import
it only where the ML stack is available — the rest of ``ai_teleop`` stays
torch-free.
"""

from ai_teleop.policy.config import DEFAULT_TBPTT_STEPS, PolicyConfig, TrainConfig
from ai_teleop.policy.losses import LossConfig, residual_bc_loss
from ai_teleop.policy.model import ResidualPolicy
from ai_teleop.policy.residual_policy import (
    POLICY_CHECKPOINT_VERSION,
    LearnedResidual,
    LoadedCheckpoint,
    load_checkpoint,
    save_checkpoint,
)
from ai_teleop.policy.run_artifacts import build_metadata, plot_history, write_run_artifacts

__all__ = [
    "PolicyConfig",
    "TrainConfig",
    "DEFAULT_TBPTT_STEPS",
    "ResidualPolicy",
    "LearnedResidual",
    "LoadedCheckpoint",
    "load_checkpoint",
    "save_checkpoint",
    "POLICY_CHECKPOINT_VERSION",
    "LossConfig",
    "residual_bc_loss",
    "build_metadata",
    "plot_history",
    "write_run_artifacts",
]
