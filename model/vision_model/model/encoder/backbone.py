"""From-scratch ViT and Hugging Face visual backbones."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..tokenizer import PatchTokenizer
from .layers import TransformerBlock


@dataclass(frozen=True)
class VisionTransformerConfig:
    image_size: int = 224
    patch_size: int = 16
    channels: int = 3
    hidden_size: int = 384
    heads: int = 6
    blocks: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1


class VisionTransformer(nn.Module):
    """A standard class-token ViT trained from random initialization."""

    def __init__(self, config: VisionTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = PatchTokenizer(
            config.image_size, config.patch_size, config.channels, config.hidden_size
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.tokenizer.num_patches + 1, config.hidden_size)
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(config.hidden_size, config.heads, config.mlp_ratio, config.dropout)
            for _ in range(config.blocks)
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    @property
    def hidden_size(self) -> int:
        return self.config.hidden_size

    def forward(self, images: Tensor) -> Tensor:
        tokens = self.tokenizer(images)
        class_token = self.class_token.expand(tokens.shape[0], -1, -1)
        hidden_states = self.dropout(torch.cat((class_token, tokens), dim=1) + self.position_embedding)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.norm(hidden_states)[:, 0]


class PretrainedVisionBackbone(nn.Module):
    """Load a Hugging Face visual encoder for full fine-tuning or probing."""

    def __init__(self, model_name: str, revision: str | None = None) -> None:
        super().__init__()
        from transformers import AutoModel

        options = {} if revision is None else {"revision": revision}
        loaded = AutoModel.from_pretrained(model_name, **options)
        self.vision_model = getattr(loaded, "vision_model", loaded)
        self.model_name = model_name
        self.revision = revision
        config = getattr(self.vision_model.config, "vision_config", self.vision_model.config)
        self.hidden_size = config.hidden_size

    def forward(self, pixel_values: Tensor) -> Tensor:
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is not None:
            return pooled
        return outputs.last_hidden_state[:, 0]
