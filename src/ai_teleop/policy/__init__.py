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
``forward`` for training and an O(1) per-tick ``step`` for deployment.
"""
