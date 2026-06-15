"""Data-generation pipeline and dataset loaders.

Runs episodes (scripted noisy-human + expert + controller + sim), logs structured
trajectories to disk, and provides loaders for BC training.

The on-disk trajectory schema (``trajectory.py``) is the stable contract M5
trains against — see ``docs/data-schema.md``.
"""

from ai_teleop.data.trajectory import (
    COLUMN_SHAPES,
    SCHEMA_VERSION,
    EpisodeRecorder,
    TerminalReason,
    load_episode,
)

__all__ = [
    "COLUMN_SHAPES",
    "SCHEMA_VERSION",
    "EpisodeRecorder",
    "TerminalReason",
    "load_episode",
]
