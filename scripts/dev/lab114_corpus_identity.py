"""LAB-114 H-B: are `dataset_9` and `dataset_10` actually two corpora?

H-B assumes they differ — audit finding H-9 reported 35 of 200 trajectories changed when
`dataset_10` was regenerated. But training on the two at the same seed produced
**bit-identical checkpoints** (matching `checkpoint_sha256`), which is only possible if the
loader sees the same bytes. This checks what is actually on disk: shared inodes (symlink or
hard link), then a content comparison of every episode array the loader reads.

Read-only. Run: `uv run python scripts/dev/lab114_corpus_identity.py`
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

A = Path("data/dataset_9")
B = Path("data/dataset_10")


def episode_files(root: Path) -> dict[int, Path]:
    manifest = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    return {int(e["episode_index"]): root / e["file"] for e in manifest["episodes"]}


def array_digest(path: Path) -> str:
    """Hash the arrays as the loader reads them — not the container bytes, which carry
    compression and ordering noise that a byte-level diff would report as a difference."""
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=True) as data:
        for key in sorted(data.files):
            value = data[key]
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def main() -> None:
    files_a, files_b = episode_files(A), episode_files(B)
    print(f"{A}: {len(files_a)} episodes │ {B}: {len(files_b)} episodes")

    shared_inode = 0
    same_content = 0
    differing: list[int] = []
    for index in sorted(set(files_a) & set(files_b)):
        path_a, path_b = files_a[index], files_b[index]
        if path_a.resolve() == path_b.resolve() or path_a.stat().st_ino == path_b.stat().st_ino:
            shared_inode += 1
            same_content += 1
            continue
        if array_digest(path_a) == array_digest(path_b):
            same_content += 1
        else:
            differing.append(index)

    print(f"same file on disk (symlink/hard link): {shared_inode}")
    print(f"identical array content:               {same_content}")
    print(f"differing episodes:                    {len(differing)} {differing[:10]}")

    manifest_a = (A / "metadata.json").read_bytes()
    manifest_b = (B / "metadata.json").read_bytes()
    print(f"manifests byte-identical: {manifest_a == manifest_b}")

    # The manifests disagree about `n_steps` on 35 episodes (audit H-9) while the episode
    # files are identical — so at most one manifest describes the trajectories on disk.
    # Whichever matches is the corpus that actually owns these files.
    for root, files in ((A, files_a), (B, files_b)):
        manifest = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        declared = {int(e["episode_index"]): e.get("n_steps") for e in manifest["episodes"]}
        mismatched = []
        for index, path in sorted(files.items()):
            with np.load(path, allow_pickle=True) as data:
                actual = int(data["cmd_position"].shape[0])  # any per-step stream; T is T
            if declared[index] != actual:
                mismatched.append((index, declared[index], actual))
        print(
            f"{root}: manifest n_steps disagrees with the arrays on "
            f"{len(mismatched)} episodes {mismatched[:5]}"
        )


if __name__ == "__main__":
    main()
