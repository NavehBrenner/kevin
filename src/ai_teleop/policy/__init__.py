"""Residual correction policy (vision-conditioned, BC-trained).

Outputs bounded Δpose + Δgrip-force given current sensors and the noisy-human's
command. Trained offline against the expert (see `expert`). To be populated
post-Milestone 4.
"""
