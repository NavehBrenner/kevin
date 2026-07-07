"""BC training for the Phase-1 residual policy (LAB-34) — train, checkpoint, deploy.

Trains the single-stateful-GRU residual (``policy.model.ResidualPolicy``) by
behavioral cloning against the M4 expert corpus: the per-channel rotation-aware
loss (``policy.losses.residual_bc_loss``), Adam + plateau LR schedule, an
episode-level train/val split (from the loader), early stopping on the val curve,
and a checkpoint that ``policy.LearnedResidual`` loads for real-time deployment
behind the M3 seam.

The reusable core is :func:`train`, which takes prebuilt ``DataLoader``s so it can
be exercised on synthetic data with no corpus on disk (see
``tests/test_residual_policy.py``). :func:`main` is its CLI front door over an M4
dataset directory.

Each run writes a self-documenting folder under ``--runs-root`` (default
``outputs/policy/runs/<timestamp>_h<hidden>l<layers>/``) holding ``checkpoint.pt``
(the deployable model), ``metadata.json`` (every hyperparameter + dataset stats +
results), ``history.json``, and ``history.png`` — so runs can be monitored and
compared without re-reading this code. See ``policy.run_artifacts``.

Run from the ``kevin/`` directory::

    uv run python scripts/train_policy.py data/dataset_1 --epochs 40
    uv run python scripts/train_policy.py data/dataset_1 --name ft_baseline --hidden-size 256
    uv run kvn train data/dataset_1 --epochs 40            # via the project CLI

Training reads only the existing corpus — regenerating data is M4's job, not a
training knob (see ``docs/milestone-5-spec.md`` anti-scope).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Allow running before the package is installed in the venv.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)
from ai_teleop.data import EpisodeBatch, build_dataloaders  # noqa: E402
from ai_teleop.policy import (  # noqa: E402
    DEFAULT_TBPTT_STEPS,
    LossConfig,
    PolicyConfig,
    ResidualPolicy,
    TrainConfig,
    build_metadata,
    residual_bc_loss,
    write_run_artifacts,
)

log = get_logger("train")


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
) -> float:
    """Run one epoch; return the step-weighted mean loss. Trains iff ``optimizer``."""
    training = optimizer is not None
    model.train(training)

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
        with torch.set_grad_enabled(training):
            # Encode the CNN **once per batch** (LAB-102). The per-step image embedding is a
            # pure function of the frames — independent of the GRU's TBPTT truncation — so
            # re-running the whole backbone on every chunk was pure waste (and blew VRAM:
            # N_chunks × the full B·F-frame grad graph). Encode once here, slice the resulting
            # embedding per chunk. None on the F/T-only path.
            image_embedding: Tensor | None = None
            if model.config.use_vision:
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
                batch_loss.backward()

        if training:
            assert optimizer is not None
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
) -> tuple[ResidualPolicy, dict[str, list[float]]]:
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

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", help="M4 dataset directory (holds metadata.json).")
    parser.add_argument(
        "--runs-root",
        default="outputs/policy/runs",
        help="Parent dir for per-run folders (default: outputs/policy/runs).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Run-folder name (default: a UTC timestamp + model-size tag).",
    )
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes. With --vision, use >0 (e.g. 4) so wrist frames "
        "decode in parallel worker processes instead of the main process.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Train/val split seed.")
    parser.add_argument("--hidden-size", type=int, default=PolicyConfig.hidden_size)
    parser.add_argument("--num-layers", type=int, default=PolicyConfig.num_layers)
    parser.add_argument("--tbptt-steps", type=int, default=DEFAULT_TBPTT_STEPS)
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Train the Phase-2 vision-conditioned policy: load wrist frames and add the "
        "image-CNN stream. Requires a dataset generated with --record all/images.",
    )
    parser.add_argument(
        "--image-backbone",
        default=PolicyConfig.image_backbone,
        help="torchvision backbone for the image encoder (with --vision).",
    )
    parser.add_argument(
        "--freeze-image-encoder",
        action="store_true",
        help="Freeze-fallback: use the pretrained backbone as a fixed extractor (train only the projection).",
    )
    parser.add_argument(
        "--no-image-pretrained",
        action="store_true",
        help="Initialize the image backbone from scratch instead of ImageNet weights (with --vision).",
    )
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available, else cpu).",
    )
    add_logging_arguments(parser)
    return parser


def _default_run_name(config: PolicyConfig) -> str:
    """``<UTC-timestamp>_h<hidden>l<layers>`` — sortable + carries the model size."""
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_h{config.hidden_size}l{config.num_layers}"


def _dataset_stats(dataset_dir: Path, train_loader: DataLoader, val_loader: DataLoader) -> dict:
    """Provenance about the corpus this run trained on (for the run metadata)."""
    manifest = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    return {
        "dir": str(dataset_dir),
        "master_seed": manifest.get("master_seed"),
        "fingerprint": manifest.get("fingerprint"),
        "schema_version": manifest.get("schema_version"),
        "n_episodes_total": manifest.get("n_episodes"),
        "n_train_episodes": len(train_loader.dataset),
        "n_val_episodes": len(val_loader.dataset),
        "expert_success_rate": manifest.get("expert", {}).get("success_rate"),
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    configure_from_args(args)

    dataset_dir = Path(args.dataset_dir)
    if not (dataset_dir / "metadata.json").exists():
        log.error("no metadata.json under %s — is this an M4 dataset directory?", dataset_dir)
        return 2

    log.info("loading corpus from %s ...", dataset_dir)
    train_loader, val_loader, norm_stats = build_dataloaders(
        dataset_dir,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        load_images=args.vision,
        num_workers=args.num_workers,
    )
    log.info("episodes: %d train │ %d val", len(train_loader.dataset), len(val_loader.dataset))

    config = PolicyConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        use_vision=args.vision,
        image_backbone=args.image_backbone,
        image_pretrained=not args.no_image_pretrained,
        freeze_image_encoder=args.freeze_image_encoder,
    )
    if args.vision:
        log.info(
            "vision ON │ backbone %s │ pretrained %s │ frozen %s │ embed %d",
            config.image_backbone,
            config.image_pretrained,
            config.freeze_image_encoder,
            config.image_embed_dim,
        )
    loss_config = LossConfig()
    train_config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        tbptt_steps=args.tbptt_steps,
        patience=args.patience,
    )

    started = time.perf_counter()
    model, history = train(
        train_loader,
        val_loader,
        config=config,
        loss_config=loss_config,
        train_config=train_config,
        device=args.device,
    )
    wall_time_s = round(time.perf_counter() - started, 1)
    log.info("training done in %.1fs", wall_time_s)

    run_dir = Path(args.runs_root) / (args.name or _default_run_name(config))
    metadata = build_metadata(
        config=config,
        loss_config=loss_config,
        train_config=train_config,
        history=history,
        dataset=_dataset_stats(dataset_dir, train_loader, val_loader),
        extra={
            "run_name": run_dir.name,
            "batch_size": args.batch_size,
            "split_seed": args.seed,
            "val_fraction": args.val_fraction,
            "device": args.device,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
