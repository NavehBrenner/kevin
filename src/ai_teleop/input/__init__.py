"""Input strategies (Strategy pattern).

Three concrete implementations share a common interface (defined in `domain`):

- ScriptedNoisyHuman : programmatic actor that drives data generation and
                       statistical KPI evaluation.
- VisionInput        : webcam-driven via MediaPipe Hands (Phase 2, demos).
- KeyboardInput      : developer fallback / debugging.

To be populated as needed across Milestones 3-5.
"""
