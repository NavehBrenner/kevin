"""Recompute every committed dataset's fingerprint from its metadata.json.

The guard for the LAB-42 / C-1 `GenerationConfig` refactor: the fingerprint a
committed manifest regenerates under must not change. Run before and after any
change to `_episode_fingerprint` / `regenerate_from_metadata` and diff the output.

    uv run python scripts/dev/lab42_fingerprint_audit.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ai_teleop.data.generate import GenerationConfig  # noqa: E402


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    listed = subprocess.run(
        ["git", "ls-files", "data/**/metadata.json"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()

    mismatches = 0
    for name in listed:
        metadata = json.loads((repo / name).read_text(encoding="utf-8"))
        committed = metadata.get("fingerprint")
        recomputed = GenerationConfig.from_metadata(metadata).fingerprint()
        flag = "ok   " if recomputed == committed else "DRIFT"
        mismatches += recomputed != committed
        print(f"{flag} {name:40s} committed={committed} recomputed={recomputed}")
    print(f"\n{len(listed)} manifests, {mismatches} drifted")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
