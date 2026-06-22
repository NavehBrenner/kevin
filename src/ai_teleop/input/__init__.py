"""Input strategies (Strategy pattern).

Three concrete implementations share a common interface (defined in `domain`):

- ScriptedNoisyHuman : programmatic actor that drives data generation and
                       statistical KPI evaluation.
- VisionInput        : stereo-webcam hand tracking via stereohand (Phase 2, demos).
- KeyboardInput      : developer fallback / debugging.

To be populated as needed across Milestones 3-5.
"""

from ai_teleop.input.scripted_noisy_human import ScriptedNoisyHuman, bore_aligned_grasp
from ai_teleop.input.vision_input import VisionInput, WorkspaceCalibration

# StereoHandSource / HandReading live in `.hand_tracker`; import them directly
# (StereoHandSource pulls in the optional `stereo-input` extra only when constructed).

__all__ = [
    "ScriptedNoisyHuman",
    "VisionInput",
    "WorkspaceCalibration",
    "bore_aligned_grasp",
]
