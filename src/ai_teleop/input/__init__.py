"""Input strategies (Strategy pattern).

Three concrete implementations share a common interface (defined in `domain`):

- ScriptedNoisyHuman : programmatic actor that drives data generation and
                       statistical KPI evaluation.
- VisionInput        : webcam-driven via MediaPipe Hands (Phase 2, demos).
- KeyboardInput      : developer fallback / debugging.

To be populated as needed across Milestones 3-5.
"""

from ai_teleop.input.scripted_noisy_human import ScriptedNoisyHuman, bore_aligned_grasp

__all__ = ["ScriptedNoisyHuman", "bore_aligned_grasp"]
