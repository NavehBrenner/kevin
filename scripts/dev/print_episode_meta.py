"""Dump an episode.npz's metadata blob — quick inspection aid for replay debugging.

Run: uv run python scripts/dev/print_episode_meta.py data/dataset_7/runs/episode_00001/episode.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ai_teleop.data.trajectory import load_episode  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    args = parser.parse_args()
    _, metadata = load_episode(args.episode)
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
