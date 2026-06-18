import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from ai_teleop.policy.config import PolicyConfig


def _fuse(*args: Tensor) -> Tensor:
    return torch.cat(args, dim=-1)


class ResidualPolicy(nn.Module):
    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()

        self.config = config

        self.core_gru = nn.GRU(
            config.input_dim,
            config.hidden_size,
            config.num_layers,
            dropout=config.dropout,
            batch_first=True,
        )

        # construc the regression hed network
        head_dims = [config.hidden_size, *config.head_hidden, config.output_dim]
        layers: list[nn.Module] = []
        for i in range(len(head_dims) - 1):
            layers.append(nn.Linear(head_dims[i], head_dims[i + 1]))
            if i < len(head_dims) - 2:
                layers.append(nn.Tanh())

        self.regression_head = nn.Sequential(*layers)

    def forward(
        self,
        command: Tensor,
        force_torque: Tensor,
        proprioception: Tensor,
        lengths: Tensor | None = None,
        hidden=None,
    ):
        x = _fuse(command, force_torque, proprioception)  # (B, max_T, 39)
        gru_input: Tensor | PackedSequence = x
        if lengths is not None:
            gru_input = pack_padded_sequence(
                x, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        # batch first = True, so shape if (B, sequence_length, hidden_dim)
        output, h_n = self.core_gru(gru_input, hidden)

        if lengths is not None:
            output, _ = pad_packed_sequence(
                output, batch_first=True, total_length=command.shape[1]
            )  # (B, T_max, hidden_dim)

        delta = self.regression_head(output)  # (B, T_max, output_dim)

        return delta, h_n

    def step(
        self,
        command: Tensor,
        force_torque: Tensor,
        proprioception: Tensor,
        hidden: Tensor | None = None,
    ):
        x = _fuse(command, force_torque, proprioception).unsqueeze(1)  # (B, 1, 39)
        output, h_n = self.core_gru(x, hidden)  # (B, 1, hidden_dim)
        delta = self.regression_head(output.squeeze(1))  # (B, output_dim)

        return delta, h_n
