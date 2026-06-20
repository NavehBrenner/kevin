"""Input strategies (Strategy pattern).

Three concrete implementations share a common interface (defined in `domain`):

- ScriptedNoisyHuman : programmatic actor that drives data generation and
                       statistical KPI evaluation.
- VisionInput        : webcam-driven via MediaPipe Hands (Phase 2, demos).
- KeyboardInput      : developer fallback / debugging.

To be populated as needed across Milestones 3-5.
"""

from ai_teleop.input.scripted_noisy_human import ScriptedNoisyHuman, bore_aligned_grasp
from ai_teleop.input.vision_input import VisionInput, WorkspaceCalibration

# MediaPipeHandTracker / HandReading live in `.hand_tracker`; import them directly
# (the tracker pulls in the optional `vision-input` extra only when constructed).

__all__ = [
    "ScriptedNoisyHuman",
    "VisionInput",
    "WorkspaceCalibration",
    "bore_aligned_grasp",
]
