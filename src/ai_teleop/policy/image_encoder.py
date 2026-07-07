"""Wrist-image CNN encoder (LAB-81) — the Phase-2 vision stream.

Implements `docs/design/policy-model.md` **Decision B**: a small torchvision
backbone (pretrained-init by default, fine-tuned end-to-end) that maps a
``(3, 224, 224)`` wrist frame to an ``image_embed_dim`` embedding ``e_img``,
which is early-fused into the GRU input alongside the command / F-T / proprio
vector streams. The all-holes auxiliary loss (Decision B option 2) is not built
here — it stays a later ``λ`` knob; this module is the encoder itself.

Two operating points from Decision B are exposed via ``PolicyConfig``:

- **pretrained-init + fine-tune end-to-end (default).** ``image_pretrained=True``
  starts from ImageNet weights and lets BC gradients adapt the whole backbone —
  directly attacking the vision-BC data-hunger risk. Inputs must be ImageNet-
  normalized, which the data loader already does (``data/images.py``).
- **freeze fallback.** ``freeze_image_encoder=True`` freezes the backbone as a
  fixed feature extractor and trains only the projection — the safety fallback if
  joint fine-tuning proves unstable/slow within the time budget.

The compact-frame contract matters for cost: an episode is stored as ``F`` unique
(decimated) frames plus a per-step index, so we encode ``F`` frames, **not** ``T``
steps (``encode_frames``). At ~100 Hz control this is what keeps the image branch
off the per-step hot path — it runs once per new frame, holding ``e_img`` between
frames (``docs/design/policy-model.md`` latency budget).
"""

from __future__ import annotations

from torch import Tensor, nn

from ai_teleop.policy.config import PolicyConfig

# Backbones we support, mapped to the attribute holding their final classifier
# Linear (whose in_features is the pooled feature width we project from). Kept to a
# small, known set so the projection surgery is explicit rather than reflective.
_CLASSIFIER_ATTR: dict[str, str] = {
    "mobilenet_v3_small": "classifier",  # nn.Sequential(...); classifier[0].in_features == 576
    "resnet18": "fc",  # nn.Linear; fc.in_features == 512
}


def _build_backbone(config: PolicyConfig) -> tuple[nn.Module, int]:
    """Instantiate the torchvision backbone and strip its classifier.

    Returns ``(backbone, feature_dim)`` where ``backbone`` still ends in an
    (identity) classifier slot we overwrite with our projection, and
    ``feature_dim`` is the pooled feature width feeding that projection. Import is
    local so the rest of ``ai_teleop.policy`` need not pull torchvision until a
    vision model is actually built.
    """
    from torchvision import models

    if config.image_backbone not in _CLASSIFIER_ATTR:
        raise ValueError(
            f"unsupported image_backbone {config.image_backbone!r}; "
            f"known: {sorted(_CLASSIFIER_ATTR)}"
        )

    weights = "DEFAULT" if config.image_pretrained else None
    backbone = getattr(models, config.image_backbone)(weights=weights)

    attr = _CLASSIFIER_ATTR[config.image_backbone]
    classifier = getattr(backbone, attr)
    # mobilenet's classifier is a Sequential whose first Linear carries the pooled
    # width; resnet's fc is that Linear directly.
    final_linear = classifier[0] if isinstance(classifier, nn.Sequential) else classifier
    assert isinstance(final_linear, nn.Linear)
    feature_dim = int(final_linear.in_features)
    # Replace the classifier with an identity so the backbone outputs the pooled
    # feature vector; the learned projection lives on ImageEncoder.
    setattr(backbone, attr, nn.Identity())
    return backbone, feature_dim


class ImageEncoder(nn.Module):
    """Wrist-frame → ``e_img`` embedding (torchvision backbone + linear projection)."""

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone, feature_dim = _build_backbone(config)
        self.projection = nn.Linear(feature_dim, config.image_embed_dim)

        if config.freeze_image_encoder:
            # Freeze-fallback: backbone is a fixed extractor; only the projection trains.
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

    def forward(self, images: Tensor) -> Tensor:
        """``(N, 3, 224, 224)`` frames → ``(N, image_embed_dim)`` embeddings."""
        return self.projection(self.backbone(images))

    def encode_frames(self, frames: Tensor) -> Tensor:
        """Encode a per-episode compact frame stack ``(B, F, 3, 224, 224)`` → ``(B, F, embed)``.

        Flatten the batch and frame axes so the CNN sees one ``(B·F, 3, 224, 224)``
        batch — we encode the ``F`` **unique** decoded frames per episode, never the
        ``T`` steps (see the module docstring's compact-frame note).
        """
        batch_size, num_frames = frames.shape[0], frames.shape[1]
        flat = frames.reshape(batch_size * num_frames, *frames.shape[2:])
        embeddings = self.forward(flat)
        return embeddings.reshape(batch_size, num_frames, self.config.image_embed_dim)
