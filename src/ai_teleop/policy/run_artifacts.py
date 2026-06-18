"""Per-training-run artifact folder (LAB-34) — make every run self-documenting.

Each training run writes its own directory so a run is fully reconstructable and
inspectable after the fact (and so runs can be compared / steered without re-reading
the training code):

```
<runs-root>/<run-name>/
├── checkpoint.pt   # the deployable model (LearnedResidual.from_checkpoint reads this)
├── metadata.json   # all hyperparameters + dataset stats + results (human-readable)
├── history.json    # per-epoch train/val loss + lr curves
└── history.png     # the loss curves plotted (log-y), for a quick eyeball
```

``checkpoint.pt`` is the same payload ``policy.save_checkpoint`` writes (weights +
normalization + config + schema versions); ``metadata.json`` is the human-facing
companion. Matplotlib is a core dependency, so the plot is always produced; the
``Agg`` backend keeps it headless (no display needed on this WSL box).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: render to file, never to a display
import matplotlib.pyplot as plt  # noqa: E402

from ai_teleop.data.dataset import NormStats  # noqa: E402
from ai_teleop.policy.config import PolicyConfig, TrainConfig  # noqa: E402
from ai_teleop.policy.losses import LossConfig  # noqa: E402
from ai_teleop.policy.model import ResidualPolicy  # noqa: E402
from ai_teleop.policy.residual_policy import save_checkpoint  # noqa: E402

CHECKPOINT_NAME = "checkpoint.pt"
METADATA_NAME = "metadata.json"
HISTORY_NAME = "history.json"
HISTORY_PLOT_NAME = "history.png"

History = dict[str, list[float]]


def git_commit() -> str | None:
    """Short HEAD sha for provenance, or ``None`` outside a git checkout."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None if completed.returncode == 0 else None


def summarize_history(history: History) -> dict[str, Any]:
    """Derive the headline results block from the per-epoch curves."""
    val = history.get("val_loss", [])
    train = history.get("train_loss", [])
    return {
        "epochs_run": len(train),
        "best_val_loss": min(val) if val else None,
        "best_epoch": int(min(range(len(val)), key=val.__getitem__)) if val else None,
        "final_train_loss": train[-1] if train else None,
        "final_val_loss": val[-1] if val else None,
    }


def plot_history(history: History, path: str | Path) -> None:
    """Plot train/val loss vs epoch (log-y) to ``path``."""
    figure, axes = plt.subplots(figsize=(7.0, 4.5))
    epochs = range(len(history.get("train_loss", [])))
    if history.get("train_loss"):
        axes.plot(epochs, history["train_loss"], marker=".", label="train")
    if history.get("val_loss"):
        axes.plot(range(len(history["val_loss"])), history["val_loss"], marker=".", label="val")
    axes.set_xlabel("epoch")
    axes.set_ylabel("BC loss")
    axes.set_yscale("log")
    axes.set_title("Residual BC training")
    axes.grid(True, which="both", alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def build_metadata(
    *,
    config: PolicyConfig,
    loss_config: LossConfig,
    train_config: TrainConfig,
    history: History,
    dataset: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the human-readable ``metadata.json`` contents for a run."""
    metadata: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "dataset": dataset,
        "model_config": asdict(config),
        "loss_config": asdict(loss_config),
        "train_config": asdict(train_config),
        "results": summarize_history(history),
    }
    if extra:
        metadata.update(extra)
    return metadata


def write_run_artifacts(
    run_dir: str | Path,
    *,
    model: ResidualPolicy,
    config: PolicyConfig,
    norm_stats: NormStats,
    loss_config: LossConfig,
    history: History,
    metadata: dict[str, Any],
) -> Path:
    """Write checkpoint + metadata + history (json + plot) into ``run_dir``.

    Returns the run directory. The checkpoint is the deployable artifact
    (``LearnedResidual.from_checkpoint(run_dir / "checkpoint.pt")``); the rest is
    for inspection and comparison across runs.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    save_checkpoint(
        run_dir / CHECKPOINT_NAME,
        model=model,
        config=config,
        norm_stats=norm_stats,
        loss_config=loss_config,
        train_history=history,
    )
    (run_dir / HISTORY_NAME).write_text(json.dumps(history, indent=2) + "\n")
    plot_history(history, run_dir / HISTORY_PLOT_NAME)
    (run_dir / METADATA_NAME).write_text(json.dumps(metadata, indent=2) + "\n")
    return run_dir
