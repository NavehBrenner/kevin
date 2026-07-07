import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from ai_teleop.policy.config import PolicyConfig
from ai_teleop.policy.image_encoder import ImageEncoder


def _fuse(*args: Tensor) -> Tensor:
    return torch.cat(args, dim=-1)


class ResidualPolicy(nn.Module):
    """Single stateful GRU core over an early-fused observation (Decision A).

    Phase 1 fuses the command / F-T / proprioception vector streams; Phase 2
    (``config.use_vision``) widens that input with a per-step wrist-image embedding
    from a fine-tuned CNN (``ImageEncoder``, Decision B). The core and head are
    identical across phases — only the fused input width changes — so the
    ``Phase2 − Phase1`` ablation stays clean. See ``docs/design/policy-model.md``.
    """

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()

        self.config = config

        # Phase-2 vision branch: built only when enabled, so the F/T-only model
        # neither constructs nor loads a CNN. It widens the GRU input by
        # ``image_embed_dim`` (config.gru_input_dim accounts for this).
        self.image_encoder = ImageEncoder(config) if config.use_vision else None

        self.core_gru = nn.GRU(
            config.gru_input_dim,
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

    def per_step_image_embedding(self, images: Tensor, image_frame_index: Tensor) -> Tensor:
        """Compact frames ``(B, F, 3, 224, 224)`` + per-step index ``(B, T)`` → ``(B, T, embed)``.

        Encodes the ``F`` unique frames once, then gathers each step's most-recent
        frame embedding via ``image_frame_index`` (the forward-fill index the loader
        builds). This is the compact-frame contract: cost scales with rendered
        frames, not steps.

        Public so the TBPTT training loop can run the CNN **once per batch** and feed
        chunk-sliced embeddings back through ``forward(image_embedding=...)``, instead
        of re-encoding the whole CNN on every truncation chunk (LAB-102).
        """
        assert self.image_encoder is not None
        frame_embeddings = self.image_encoder.encode_frames(images)  # (B, F, embed)
        gather_index = image_frame_index.unsqueeze(-1).expand(
            -1, -1, frame_embeddings.shape[-1]
        )  # (B, T, embed)
        return frame_embeddings.gather(1, gather_index)  # (B, T, embed)

    def forward(
        self,
        command: Tensor,
        force_torque: Tensor,
        proprioception: Tensor,
        images: Tensor | None = None,
        image_frame_index: Tensor | None = None,
        image_embedding: Tensor | None = None,
        lengths: Tensor | None = None,
        hidden=None,
    ):
        streams = [command, force_torque, proprioception]
        if self.config.use_vision:
            # Prefer a precomputed per-step embedding stream (B, T, embed): the training
            # loop encodes the CNN once per batch and passes chunk slices, so the backbone
            # isn't re-run on every TBPTT chunk (LAB-102). Fall back to encoding raw frames
            # for callers that pass a full sequence (eval, tests).
            if image_embedding is not None:
                streams.append(image_embedding)
            elif images is not None and image_frame_index is not None:
                streams.append(self.per_step_image_embedding(images, image_frame_index))
            else:
                raise ValueError(
                    "use_vision=True requires image_embedding, or images and image_frame_index"
                )

        x = _fuse(*streams)  # (B, max_T, gru_input_dim)
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
        image: Tensor | None = None,
        hidden: Tensor | None = None,
    ):
        streams = [command, force_torque, proprioception]
        if self.config.use_vision:
            if image is None:
                raise ValueError("use_vision=True requires an image")
            assert self.image_encoder is not None
            streams.append(self.image_encoder(image))  # (B, embed)

        x = _fuse(*streams).unsqueeze(1)  # (B, 1, gru_input_dim)
        output, h_n = self.core_gru(x, hidden)  # (B, 1, hidden_dim)
        delta = self.regression_head(output.squeeze(1))  # (B, output_dim)

        return delta, h_n
