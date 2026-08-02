"""Vision Transformer backbones for teaching and linear probing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from ..tokenizer import PatchTokenizer
from .layers import TransformerBlock


@dataclass(frozen=True)
class VisionTransformerConfig:
    """Dimensions for the compact ViT implemented in this repository."""

    image_size: int = 224
    patch_size: int = 16
    channels: int = 3
    hidden_size: int = 768
    heads: int = 12
    blocks: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    rope_theta: float = 10_000.0

    def __post_init__(self) -> None:
        dimensions = (
            self.image_size,
            self.patch_size,
            self.channels,
            self.hidden_size,
            self.heads,
            self.blocks,
        )
        

class VisionTransformer(nn.Module):
    """A small encoder-only ViT whose attention equations remain inspectable."""

    def __init__(self, config: VisionTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = PatchTokenizer(
            image_size=config.image_size,
            patch_size=config.patch_size,
            channels=config.channels,
            hidden_size=config.hidden_size,
        )
        self.grid_size = (self.tokenizer.grid_size, self.tokenizer.grid_size)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                hidden_size=config.hidden_size,
                heads=config.heads,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                rope_theta=config.rope_theta,
            )
            for _ in range(config.blocks)
        )
        self.output_norm = nn.LayerNorm(config.hidden_size)

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    def forward(self, images: Tensor) -> Tensor:
        """Return normalized patch tokens with shape ``[batch, patches, hidden]``."""
        hidden_states = self.tokenizer(images)
        for block in self.blocks:
            hidden_states = block(hidden_states, self.grid_size)
        return self.output_norm(hidden_states)


def _vision_hidden_size(config: Any) -> int:
    """Read a vision width from common Hugging Face configuration layouts."""
    candidates = (getattr(config, "vision_config", None), config)
    for candidate in candidates:
        hidden_size = getattr(candidate, "hidden_size", None)
        if isinstance(hidden_size, int) and hidden_size > 0:
            return hidden_size
    raise ValueError("The pretrained checkpoint does not expose a vision hidden_size.")


class PretrainedVisionBackbone(nn.Module):
    """Load a Hugging Face vision encoder and keep it frozen for a linear probe."""

    def __init__(self, model_name: str, revision: str | None = None) -> None:
        super().__init__()
        if not model_name:
            raise ValueError("model_name cannot be empty.")

        from transformers import AutoConfig, AutoModel

        model_kwargs = {} if revision is None else {"revision": revision}
        checkpoint_config = AutoConfig.from_pretrained(model_name, **model_kwargs)
        if checkpoint_config.model_type == "siglip":
            from transformers import SiglipVisionModel

            self.vision_model = SiglipVisionModel.from_pretrained(model_name, **model_kwargs)
        else:
            loaded_model = AutoModel.from_pretrained(model_name, **model_kwargs)
            self.vision_model = getattr(loaded_model, "vision_model", loaded_model)
        self.model_name = model_name
        self.revision = revision
        self._frozen = False
        self.hidden_size = _vision_hidden_size(self.vision_model.config)
        self.freeze()

    def freeze(self) -> None:
        """Disable gradients and stochastic layers in the pretrained encoder."""
        self._frozen = True
        for parameter in self.vision_model.parameters():
            parameter.requires_grad_(False)
        self.vision_model.eval()

    def unfreeze(self) -> None:
        """Explicitly enable gradients when moving beyond linear probing."""
        self._frozen = False
        for parameter in self.vision_model.parameters():
            parameter.requires_grad_(True)

    def train(self, mode: bool = True) -> "PretrainedVisionBackbone":
        super().train(mode)
        if self._frozen:
            self.vision_model.eval()
        return self

    def forward(self, pixel_values: Tensor) -> Tensor:
        """Return the checkpoint's vision tokens, or a pooled token as a fallback."""
        outputs = self.vision_model(pixel_values=pixel_values)
        tokens = getattr(outputs, "last_hidden_state", None)
        if tokens is None and isinstance(outputs, (tuple, list)) and outputs:
            tokens = outputs[0]
        if tokens is not None:
            if tokens.ndim == 2:
                tokens = tokens.unsqueeze(1)
            if tokens.ndim != 3:
                raise ValueError("The vision encoder must return [B, tokens, hidden] features.")
            return tokens

        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = getattr(outputs, "image_embeds", None)
        if pooled is None:
            raise ValueError("The pretrained checkpoint returned no usable vision features.")
        if pooled.ndim != 2:
            raise ValueError("Pooled vision features must have shape [B, hidden].")
        return pooled.unsqueeze(1)
