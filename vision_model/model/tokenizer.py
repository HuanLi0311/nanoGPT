"""Turn an image tensor into a sequence of continuous patch tokens."""

from __future__ import annotations

import torch
from torch import nn


class PatchTokenizer(nn.Module):
    """Patchify images and learn a projection for each flattened patch."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        channels: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        if min(image_size, patch_size, channels, hidden_size) <= 0:
            raise ValueError("image_size, patch_size, channels, and hidden_size must be positive.")
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size.")

        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size**2
        self.projection = nn.Linear(channels * patch_size**2, hidden_size)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        """Return flattened patches with shape [batch, num_patches, channels * patch_area]."""
        if images.ndim != 4:
            raise ValueError("images must have shape [B, C, H, W].")
        batch_size, channels, height, width = images.shape
        if (channels, height, width) != (self.channels, self.image_size, self.image_size):
            raise ValueError("images do not match the configured channel count and image size.")

        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        return patches.permute(0, 2, 3, 1, 4, 5).reshape(batch_size, self.num_patches, -1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return continuous vision tokens with shape [batch, num_patches, hidden_size]."""
        return self.projection(self.patchify(images))
