from dataclasses import dataclass

# Full-sequence BPTT by default; M4 episodes can be thousands of steps, so the
# train loop chunks time into windows this long and carries a detached hidden
# state across them (truncated BPTT) to bound memory. A window larger than any
# episode == plain full BPTT.
DEFAULT_TBPTT_STEPS = 256


@dataclass(frozen=True)
class PolicyConfig:
    # inputs
    command_dim: int = 9
    force_torque_dim: int = 6
    proprioception_dim: int = 24

    @property
    def input_dim(self) -> int:
        """Width of the concatenated **vector** streams (command + F/T + proprio).

        Vision-independent by design: this is the Phase-1 fused width and stays 39
        regardless of ``use_vision``. The image embedding widens the GRU input
        separately (see ``gru_input_dim``), preserving the clean ``Phase2 − Phase1``
        input-width change (``docs/design/policy-model.md``).
        """
        return self.command_dim + self.force_torque_dim + self.proprioception_dim

    # vision (Phase 2 — docs/design/policy-model.md Decision B). Off by default so
    # the F/T-only Phase-1 model and its checkpoints are unchanged; every field is
    # defaulted so an old checkpoint's config dict still deserializes.
    use_vision: bool = False
    image_embed_dim: int = 128  # width of the CNN embedding fused into the GRU input
    image_backbone: str = "mobilenet_v3_small"  # torchvision backbone name
    image_pretrained: bool = True  # ImageNet-pretrained init (fine-tuned end-to-end)
    freeze_image_encoder: bool = False  # freeze-fallback: use the backbone as a fixed extractor

    @property
    def gru_input_dim(self) -> int:
        """Actual GRU input width: the vector streams, widened by the image
        embedding when vision is on. Phase 1 == ``input_dim``; Phase 2 adds
        ``image_embed_dim`` — the core/head are otherwise identical across phases.
        """
        return self.input_dim + (self.image_embed_dim if self.use_vision else 0)

    # model
    hidden_size: int = 128
    num_layers: int = 2
    head_hidden: tuple[int, ...] = (128,)
    dropout: float = 0
    use_tanh_head: bool = False

    # outputs
    output_dim: int = 7


@dataclass(frozen=True)
class TrainConfig:
    """Optimization + early-stopping knobs (model/loss shape live in their own configs)."""

    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    tbptt_steps: int = DEFAULT_TBPTT_STEPS
    patience: int = 8  # early-stop after this many epochs without val improvement
    min_delta: float = 1e-4  # smallest val-loss drop that counts as improvement
    lr_patience: int = 3  # ReduceLROnPlateau patience
    lr_factor: float = 0.5

    # Memory levers for Stage-C image-encoder fine-tuning (unfrozen backbone). Both are
    # training-time only — they change neither weights nor architecture, so they live
    # here, not in the serialized PolicyConfig. Off by default (Phase-1 / frozen paths
    # don't need them). See docs + concepts/vision-conditioned-policy.md (perception ceiling).
    use_amp: bool = False  # mixed-precision autocast + GradScaler (halves activation VRAM)
    checkpoint_image_encoder: bool = (
        False  # recompute backbone activations in backward (biggest cut)
    )
    image_encode_chunk: int = 0  # cap frames/backbone-forward in encode_frames (0 = whole batch)
