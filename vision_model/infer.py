"""Run classification inference with a saved linear-probe checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoImageProcessor

from .model.encoder.backbone import PretrainedVisionBackbone
from .model.heads.classification import LinearProbe
from .settings import resolve_device


def _load_probe(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[PretrainedVisionBackbone, LinearProbe, list[str]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("backbone_name")
    classes = checkpoint.get("classes")
    hidden_size = checkpoint.get("hidden_size")
    if not isinstance(model_name, str) or not isinstance(classes, list) or not classes:
        raise ValueError("Checkpoint metadata is missing the backbone name or class list.")
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        raise ValueError("Checkpoint metadata is missing a valid hidden_size.")

    revision = checkpoint.get("backbone_revision")
    if revision is not None and not isinstance(revision, str):
        raise ValueError("Checkpoint backbone_revision must be a string or null.")
    backbone = PretrainedVisionBackbone(model_name, revision=revision)
    if backbone.hidden_size != hidden_size:
        raise ValueError("Checkpoint hidden_size does not match the pretrained backbone.")
    probe = LinearProbe(hidden_size, len(classes))
    probe.load_state_dict(checkpoint["head_state_dict"])
    backbone.to(device).eval()
    probe.to(device).eval()
    return backbone, probe, [str(label) for label in classes]


def predict(
    image_path: Path,
    checkpoint_path: Path,
    top_k: int,
    device: torch.device,
) -> list[tuple[str, float]]:
    """Return the highest-probability classes for one image."""
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    backbone, probe, classes = _load_probe(checkpoint_path, device)
    model_name = backbone.model_name
    revision = backbone.revision
    processor_kwargs = {} if revision is None else {"revision": revision}
    processor = AutoImageProcessor.from_pretrained(model_name, **processor_kwargs)

    image = Image.open(image_path).convert("RGB")
    encoded = processor(images=image, return_tensors="pt")
    pixel_values = encoded["pixel_values"].to(device)
    with torch.no_grad():
        probabilities = torch.softmax(probe(backbone(pixel_values)), dim=-1)[0]
    count = min(top_k, len(classes))
    scores, indices = torch.topk(probabilities, count)
    return [(classes[int(index)], float(score)) for score, index in zip(scores, indices)]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify one image with a linear probe.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("vision_model/checkpoints/siglip2_linear_probe.pt"),
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    device = resolve_device(args.device)
    for label, probability in predict(args.image, args.checkpoint, args.top_k, device):
        print(f"{label}\t{probability:.6f}")


if __name__ == "__main__":
    main()
