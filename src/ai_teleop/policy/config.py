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
        return self.command_dim + self.force_torque_dim + self.proprioception_dim

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
