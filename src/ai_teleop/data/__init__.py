"""Data-generation pipeline and dataset loaders.

Runs episodes (scripted noisy-human + expert + controller + sim), logs structured
trajectories to disk, and provides loaders for BC training.

The on-disk trajectory schema (``trajectory.py``) is the stable contract M5
trains against — see ``docs/data-schema.md``.
"""

from ai_teleop.data.dataset import Episode, OfflineResidualBCDataset, split_episodes
from ai_teleop.data.generate import generate_dataset, regenerate_from_metadata
from ai_teleop.data.schema import (
    DatasetConfig,
    EpisodeMetadata,
    EpisodeSummary,
    ResBCDatasetMetadata,
)
from ai_teleop.data.trajectory import (
    COLUMN_SHAPES,
    EPISODE_NPZ_NAME,
    IMGS_DIRNAME,
    SCHEMA_VERSION,
    EpisodeRecorder,
    TerminalReason,
    episode_dir,
    episode_imgs_dir,
    episode_npz_path,
    load_episode,
)

__all__ = [
    # schema + reader/writer
    "COLUMN_SHAPES",
    "SCHEMA_VERSION",
    "EpisodeRecorder",
    "TerminalReason",
    "load_episode",
    # on-disk layout
    "EPISODE_NPZ_NAME",
    "IMGS_DIRNAME",
    "episode_dir",
    "episode_npz_path",
    "episode_imgs_dir",
    # metadata contracts (TypedDicts)
    "DatasetConfig",
    "EpisodeMetadata",
    "EpisodeSummary",
    "ResBCDatasetMetadata",
    # loader
    "Episode",
    "split_episodes",
    "OfflineResidualBCDataset",
    # generation pipeline
    "generate_dataset",
    "regenerate_from_metadata",
]
