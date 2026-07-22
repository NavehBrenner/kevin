"""M5 BC-training CLI — thin entry point over `ai_teleop.policy.train`.

The training pipeline is core functionality and lives in the package
(`ai_teleop.policy.train`); this script is just its command-line front door (also
reachable as `kvn train`). See that module for the loop, the run-folder layout, and
the programmatic entry point `ai_teleop.dagger` uses.

Each run writes a self-documenting folder under ``--runs-root`` (default
``outputs/policy/runs/<timestamp>_h<hidden>l<layers>/``) holding ``checkpoint.pt``
(the deployable model), ``metadata.json`` (every hyperparameter + dataset stats +
results), ``history.json``, and ``history.png``.

Run from the ``kevin/`` directory::

    uv run python scripts/train_policy.py data/dataset_1 --epochs 40
    uv run python scripts/train_policy.py data/dataset_1 --name ft_baseline --hidden-size 256
    uv run kvn train data/dataset_1 --epochs 40            # via the project CLI
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Allow running before the package is installed in the venv.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_teleop.common.log import (  # noqa: E402
    add_logging_arguments,
    configure_from_args,
    get_logger,
)
from ai_teleop.policy import (  # noqa: E402
    DEFAULT_TBPTT_STEPS,
    LossConfig,
    PolicyConfig,
    TrainConfig,
)
from ai_teleop.policy.train import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_RUNS_ROOT,
    DEFAULT_VAL_FRACTION,
    train_policy,
)

log = get_logger("train")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", help="M4 dataset directory (holds metadata.json).")
    parser.add_argument(
        "--runs-root",
        default=DEFAULT_RUNS_ROOT,
        help=f"Parent dir for per-run folders (default: {DEFAULT_RUNS_ROOT}).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Run-folder name (default: a UTC timestamp + model-size tag).",
    )
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
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
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Stage C: mixed-precision training (autocast + GradScaler). Halves activation "
        "VRAM; use with an unfrozen backbone. CUDA only (ignored on CPU).",
    )
    parser.add_argument(
        "--checkpoint-image-encoder",
        action="store_true",
        help="Stage C: gradient-checkpoint the image backbone (recompute activations in "
        "backward). The biggest VRAM cut — lets the unfrozen backbone fine-tune on 8 GB.",
    )
    parser.add_argument(
        "--image-encode-chunk",
        type=int,
        default=0,
        help="Stage C: max frames per backbone forward inside encode_frames (0 = whole "
        "batch). Bounds peak VRAM to one chunk regardless of B·F; pair with "
        "--checkpoint-image-encoder (e.g. 32) when fine-tuning on a small GPU.",
    )
    parser.add_argument(
        "--action-rate-weight",
        type=float,
        default=LossConfig.weight_action_rate,
        help="Smoothness penalty weight (LAB-104): penalizes the per-step change in the "
        "predicted Δ to kill the sub-clamp jerk regression. 0 disables (default).",
    )
    parser.add_argument(
        "--weight-position",
        type=float,
        default=LossConfig.weight_position,
        help="BC loss weight on the position channel (LAB-106). The Δposition target is "
        "~mm-scale next to orientation in radians, so the default 1.0 under-serves it; raise "
        "to force the optimizer to fit the (success-critical) lateral correction.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=TrainConfig.weight_decay,
        help="Adam weight decay (L2). >0 regularizes the far-field spurious-correction floor "
        "the BC net emits where the expert is structurally zero (LAB-106).",
    )
    parser.add_argument(
        "--command-ee-delta",
        action="store_true",
        help="LAB-106: append the raw (command_position − ee_position) tracking-error vector "
        "to the proprioception stream so the GRU can learn the free-space zero (the residual "
        "∝ this vector). Adds 3 input dims; both train and deploy assembly gate on it.",
    )
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available, else cpu).",
    )
    add_logging_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    configure_from_args(args)

    config = PolicyConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        use_vision=args.vision,
        image_backbone=args.image_backbone,
        image_pretrained=not args.no_image_pretrained,
        freeze_image_encoder=args.freeze_image_encoder,
        use_command_ee_delta=args.command_ee_delta,
    )
    if args.vision:
        log.info(
            "vision ON │ backbone %s │ pretrained %s │ frozen %s │ embed %d │ amp %s │ ckpt %s",
            config.image_backbone,
            config.image_pretrained,
            config.freeze_image_encoder,
            config.image_embed_dim,
            args.amp,
            args.checkpoint_image_encoder,
        )

    try:
        train_policy(
            args.dataset_dir,
            config=config,
            loss_config=LossConfig(
                weight_position=args.weight_position, weight_action_rate=args.action_rate_weight
            ),
            train_config=TrainConfig(
                epochs=args.epochs,
                learning_rate=args.lr,
                weight_decay=args.weight_decay,
                tbptt_steps=args.tbptt_steps,
                patience=args.patience,
                use_amp=args.amp,
                checkpoint_image_encoder=args.checkpoint_image_encoder,
                image_encode_chunk=args.image_encode_chunk,
            ),
            runs_root=args.runs_root,
            name=args.name,
            batch_size=args.batch_size,
            val_fraction=args.val_fraction,
            seed=args.seed,
            num_workers=args.num_workers,
            device=args.device,
        )
    except FileNotFoundError as error:
        log.error("%s", error)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
