from dataclasses import dataclass


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
