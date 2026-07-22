"""BC training pipeline for the residual policy (LAB-34) — corpus in, run folder out.

Core functionality (the `scripts/train_policy.py` CLI is just its front door, the
same split `ai_teleop.data.generate` / `scripts/generate_dataset.py` uses). Two
entry points, at two levels:

- :func:`train` — the loop itself, over **prebuilt** ``DataLoader``s: the
  per-channel rotation-aware loss (`policy.losses.residual_bc_loss`), Adam + a
  plateau LR schedule, TBPTT over the stateful GRU, early stopping on the val
  curve, best-weight restore. Takes loaders so it can be exercised on synthetic
  data with no corpus on disk (`tests/test_residual_policy.py`).
- :func:`train_policy` — the whole pipeline: load an M4 corpus, train, and write
  the self-documenting run folder (``checkpoint.pt`` + ``metadata.json`` +
  ``history.{json,png}`` — see `policy.run_artifacts`). Returns a `TrainedRun`
  carrying the checkpoint path, so a programmatic caller never has to rebuild it
  from a naming convention.

`ai_teleop.dagger` calls :func:`train_policy` directly, once per DAgger round.
It used to shell out to the CLI script with a 14-element argv — the checkpoint
path reconstructed by string convention, failures arriving as an exit code, and
nothing type-checked across the boundary. That was forced by *where the pipeline
lived*, not by anything about the training itself (audit finding G-1).

Training reads only the existing corpus — regenerating data is M4's job, not a
training knob (see ``docs/milestone-5-spec.md`` anti-scope).
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from ai_teleop.common.log import get_logger
from ai_teleop.data import EpisodeBatch, build_dataloaders
from ai_teleop.policy.config import PolicyConfig, TrainConfig
from ai_teleop.policy.losses import LossConfig, residual_bc_loss
from ai_teleop.policy.model import ResidualPolicy
from ai_teleop.policy.run_artifacts import (
    CHECKPOINT_NAME,
    History,
    build_metadata,
    write_run_artifacts,
)

log = get_logger("train")

# Shared by `train_policy` and the CLI's argparse defaults, so the two cannot drift
# (the failure mode audit findings C-1 and C-3 both turned on).
DEFAULT_RUNS_ROOT = "outputs/policy/runs"
DEFAULT_BATCH_SIZE = 16
DEFAULT_VAL_FRACTION = 0.2


@dataclass(frozen=True)
class TrainedRun:
    """What one training run produced.

    ``checkpoint_path`` is returned rather than left to the caller to assemble from
    ``runs_root / name / "checkpoint.pt"`` — that convention was exactly what the
    DAgger loop had to hard-code while training was reachable only as a subprocess.
    """

    run_dir: Path
    checkpoint_path: Path
    history: History
    metadata: dict[str, Any]


def _step_mask(lengths: Tensor, max_steps: int) -> Tensor:
    """``(B, T)`` 1/0 mask of real (non-padding) steps from per-episode lengths."""
    indices = torch.arange(max_steps, device=lengths.device)
    return (indices[None, :] < lengths[:, None]).to(torch.float32)


def _to_device(batch: EpisodeBatch, device: torch.device) -> EpisodeBatch:
    return EpisodeBatch(
        command=batch.command.to(device),
        force_torque=batch.force_torque.to(device),
        proprioception=batch.proprioception.to(device),
        delta=batch.delta.to(device),
        lengths=batch.lengths.to(device),
        # Carried through for the Phase-2 vision path; None for the F/T-only corpus.
        images=batch.images.to(device) if batch.images is not None else None,
        image_frame_index=(
            batch.image_frame_index.to(device) if batch.image_frame_index is not None else None
        ),
    )


def _epoch(
    model: ResidualPolicy,
    loader: DataLoader,
    loss_config: LossConfig,
    device: torch.device,
    *,
    tbptt_steps: int,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """Run one epoch; return the step-weighted mean loss. Trains iff ``optimizer``.

    ``scaler`` (Stage C) enables mixed-precision: the forward runs under ``autocast``
    and the loss is scaled before backward so tiny fp16 gradients don't underflow.
    ``None`` ⇒ full fp32 (the Phase-1 / frozen path, unchanged).
    """
    training = optimizer is not None
    model.train(training)
    use_amp = scaler is not None

    total_loss = 0.0
    total_steps = 0.0
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        max_steps = batch.command.shape[1]
        mask = _step_mask(batch.lengths, max_steps)  # (B, T)

        if training:
            assert optimizer is not None
            optimizer.zero_grad()

        hidden: Tensor | None = None
        # autocast wraps the forward only; backward/step stay outside it (fp32 master).
        with (
            torch.set_grad_enabled(training),
            torch.autocast(device.type, enabled=use_amp),
        ):
            # Encode the CNN **once per batch** (LAB-102). The per-step image embedding is a
            # pure function of the frames — independent of the GRU's TBPTT truncation — so
            # re-running the whole backbone on every chunk was pure waste (and blew VRAM:
            # N_chunks × the full B·F-frame grad graph). Encode once here, slice the resulting
            # embedding per chunk. None on the F/T-only path.
            image_embedding: Tensor | None = None
            if model.config.use_vision:
                # A vision config always loads frames (train_policy ties load_images to
                # config.use_vision), so both are populated here.
                assert batch.images is not None and batch.image_frame_index is not None
                image_embedding = model.per_step_image_embedding(
                    batch.images, batch.image_frame_index
                )

            # Accumulate chunk losses and backward once per batch. With `hidden` detached at
            # each boundary the chunk graphs are independent, so Σ backward(chunk_loss) equals
            # backward(Σ chunk_loss) — identical gradients — but the shared CNN graph is then
            # traversed once, not retained/rebuilt per chunk.
            batch_loss: Tensor | None = None
            for start in range(0, max_steps, tbptt_steps):
                end = min(start + tbptt_steps, max_steps)
                chunk_mask = mask[:, start:end]
                chunk_valid = float(chunk_mask.sum())
                if chunk_valid == 0.0:
                    break  # all-padding tail (chunks are time-ordered)

                predicted, next_hidden = model.forward(
                    batch.command[:, start:end],
                    batch.force_torque[:, start:end],
                    batch.proprioception[:, start:end],
                    image_embedding=(
                        image_embedding[:, start:end] if image_embedding is not None else None
                    ),
                    hidden=hidden,
                )
                chunk_loss = residual_bc_loss(
                    predicted, batch.delta[:, start:end], chunk_mask, config=loss_config
                )
                if training:
                    batch_loss = chunk_loss if batch_loss is None else batch_loss + chunk_loss
                # Carry the recurrent state across chunks but cut the graph (TBPTT).
                hidden = next_hidden.detach()

                total_loss += float(chunk_loss.detach()) * chunk_valid
                total_steps += chunk_valid

        if training and batch_loss is not None:
            assert optimizer is not None
            if scaler is not None:
                scaler.scale(batch_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                optimizer.step()

    return total_loss / total_steps if total_steps > 0 else math.nan


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    config: PolicyConfig | None = None,
    loss_config: LossConfig | None = None,
    train_config: TrainConfig | None = None,
    device: str = "cpu",
) -> tuple[ResidualPolicy, History]:
    """BC-train a residual policy on prebuilt loaders; return ``(best_model, history)``.

    Restores the best-validation weights before returning (early stopping), so the
    returned model is the one worth checkpointing. ``history`` holds the per-epoch
    ``train_loss`` / ``val_loss`` curves for plotting and the sanity check that both
    decrease.
    """
    config = config or PolicyConfig()
    loss_config = loss_config or LossConfig()
    train_config = train_config or TrainConfig()
    torch_device = torch.device(device)

    model = ResidualPolicy(config).to(torch_device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_config.learning_rate, weight_decay=train_config.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer, factor=train_config.lr_factor, patience=train_config.lr_patience
    )

    # Stage-C memory levers (only bite with an *unfrozen* backbone on CUDA).
    if config.use_vision:
        assert model.image_encoder is not None
        model.image_encoder.use_gradient_checkpoint = train_config.checkpoint_image_encoder
        model.image_encoder.encode_chunk_size = train_config.image_encode_chunk
    amp_enabled = train_config.use_amp and torch_device.type == "cuda"
    scaler = torch.amp.GradScaler() if amp_enabled else None

    history: History = {"train_loss": [], "val_loss": []}
    best_val_loss = math.inf
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(train_config.epochs):
        train_loss = _epoch(
            model,
            train_loader,
            loss_config,
            torch_device,
            tbptt_steps=train_config.tbptt_steps,
            optimizer=optimizer,
            scaler=scaler,
        )
        val_loss = _epoch(
            model,
            val_loader,
            loss_config,
            torch_device,
            tbptt_steps=train_config.tbptt_steps,
            optimizer=None,
        )
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        log.info(
            "epoch %3d │ train %.5f │ val %.5f │ lr %.2e",
            epoch,
            train_loss,
            val_loss,
            optimizer.param_groups[0]["lr"],
        )

        # Checkpoint the true minimum (any strict improvement), but reset the
        # early-stop patience only on a *meaningful* (> min_delta) improvement —
        # so the saved weights are the genuine best, not a min_delta-quantized one.
        meaningful_improvement = val_loss < best_val_loss - train_config.min_delta
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
        if meaningful_improvement:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_config.patience:
                log.info("early stop at epoch %d (best val %.5f)", epoch, best_val_loss)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def _n_episodes(loader: DataLoader) -> int:
    """Episode count behind a loader. ``Dataset`` isn't ``Sized`` in the torch stubs,
    but every dataset this project builds is a plain list of episodes."""
    return len(loader.dataset)  # type: ignore[arg-type]


def default_run_name(config: PolicyConfig) -> str:
    """``<UTC-timestamp>_h<hidden>l<layers>`` — sortable + carries the model size."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_h{config.hidden_size}l{config.num_layers}"


def _dataset_stats(
    dataset_dir: Path, train_loader: DataLoader, val_loader: DataLoader
) -> dict[str, Any]:
    """Provenance about the corpus this run trained on (for the run metadata)."""
    manifest = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    return {
        "dir": str(dataset_dir),
        "master_seed": manifest.get("master_seed"),
        "fingerprint": manifest.get("fingerprint"),
        "schema_version": manifest.get("schema_version"),
        "n_episodes_total": manifest.get("n_episodes"),
        "n_train_episodes": _n_episodes(train_loader),
        "n_val_episodes": _n_episodes(val_loader),
        "expert_success_rate": manifest.get("expert", {}).get("success_rate"),
    }


def train_policy(
    dataset_dir: str | Path,
    *,
    config: PolicyConfig | None = None,
    loss_config: LossConfig | None = None,
    train_config: TrainConfig | None = None,
    runs_root: str | Path = DEFAULT_RUNS_ROOT,
    name: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = 0,
    num_workers: int = 0,
    device: str = "cpu",
) -> TrainedRun:
    """Train on an M4 corpus and write the run folder; return what it produced.

    Loads ``dataset_dir`` into train/val loaders (episode-level split, ``seed``),
    trains via :func:`train`, then writes ``checkpoint.pt`` + ``metadata.json`` +
    ``history.{json,png}`` under ``runs_root/name`` (``name`` defaults to a
    timestamp + model-size tag).

    ``load_images`` follows ``config.use_vision``: the modality is a property of the
    policy being trained, so a vision config always loads wrist frames and an
    F/T-only config never does — a caller cannot accidentally pair the two. Same for
    ``config.use_command_ee_delta``, which must gate the train-side stream assembly
    exactly as it gates deployment.

    Raises ``FileNotFoundError`` if ``dataset_dir`` holds no ``metadata.json``.
    """
    config = config or PolicyConfig()
    loss_config = loss_config or LossConfig()
    train_config = train_config or TrainConfig()

    dataset_dir = Path(dataset_dir)
    if not (dataset_dir / "metadata.json").exists():
        raise FileNotFoundError(
            f"no metadata.json under {dataset_dir} — is this an M4 dataset directory?"
        )

    log.info("loading corpus from %s ...", dataset_dir)
    train_loader, val_loader, norm_stats = build_dataloaders(
        dataset_dir,
        batch_size=batch_size,
        val_fraction=val_fraction,
        seed=seed,
        load_images=config.use_vision,
        num_workers=num_workers,
        command_ee_delta=config.use_command_ee_delta,
    )
    log.info("episodes: %d train │ %d val", _n_episodes(train_loader), _n_episodes(val_loader))

    started = time.perf_counter()
    model, history = train(
        train_loader,
        val_loader,
        config=config,
        loss_config=loss_config,
        train_config=train_config,
        device=device,
    )
    wall_time_s = round(time.perf_counter() - started, 1)
    log.info("training done in %.1fs", wall_time_s)

    run_dir = Path(runs_root) / (name or default_run_name(config))
    metadata = build_metadata(
        config=config,
        loss_config=loss_config,
        train_config=train_config,
        history=history,
        dataset=_dataset_stats(dataset_dir, train_loader, val_loader),
        extra={
            "run_name": run_dir.name,
            "batch_size": batch_size,
            "split_seed": seed,
            "val_fraction": val_fraction,
            "device": device,
            "wall_time_s": wall_time_s,
        },
    )
    write_run_artifacts(
        run_dir,
        model=model,
        config=config,
        norm_stats=norm_stats,
        loss_config=loss_config,
        history=history,
        metadata=metadata,
    )
    log.info(
        "run → %s │ best val %.5f │ checkpoint.pt + metadata.json + history.{json,png}",
        run_dir,
        min(history["val_loss"], default=math.nan),
    )
    return TrainedRun(
        run_dir=run_dir,
        checkpoint_path=run_dir / CHECKPOINT_NAME,
        history=history,
        metadata=metadata,
    )
