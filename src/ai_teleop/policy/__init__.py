"""Residual correction policy (vision-conditioned, BC-trained).

Outputs bounded Δpose + Δgrip-force given current sensors and the noisy-human's
command. Multi-stream encoder design: GRU encoders for the command and F/T
histories, an MLP for proprioception, and (Phase 2) an image CNN fine-tuned
end-to-end from a pretrained init, fused by an MLP head. Trained offline against
the expert (see `expert`). See ``docs/design/policy-model.md`` for the full
architecture and the locked encoder decisions. To be populated post-Milestone 4.
"""
