"""Classify one image from either vision-training checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from transformers import AutoImageProcessor

from .model import ClassificationHead, PretrainedVisionBackbone, VisionTransformer, VisionTransformerConfig
from .model.training.data import ProcessorTransform, resolve_device


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, list[str], object]:
    checkpoint = torch.load(path, map_location=device)
    config, classes = checkpoint["config"], checkpoint["classes"]
    if checkpoint["kind"] == "vit":
        backbone = VisionTransformer(VisionTransformerConfig(**config["model"]))
        size = config["model"]["image_size"]
        transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    else:
        options = {} if config["model"].get("revision") is None else {"revision": config["model"]["revision"]}
        processor = AutoImageProcessor.from_pretrained(config["model"]["name"], **options)
        backbone = PretrainedVisionBackbone(config["model"]["name"], config["model"].get("revision"))
        transform = ProcessorTransform(processor)
    model = nn.Sequential(backbone, ClassificationHead(backbone.hidden_size, len(classes))).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), classes, transform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = resolve_device(args.device)
    model, classes, transform = load_checkpoint(args.checkpoint, device)
    image = transform(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = model(image).softmax(dim=-1)[0]
    scores, indices = probabilities.topk(min(args.top_k, len(classes)))
    for score, index in zip(scores, indices):
        print(f"{classes[index]}\t{score.item():.6f}")


if __name__ == "__main__":
    main()
