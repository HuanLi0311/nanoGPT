"""Train the from-scratch Vision Transformer on an ImageFolder dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch import nn
from torchvision import transforms

from ..model import ClassificationHead, VisionTransformer, VisionTransformerConfig
from ..training.data import make_loaders, resolve_device, set_seed
from ..training.loop import evaluate, train_epoch
from ..training.optimizer import build_optimizer


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("vision_model/config/train.yaml"))
    return parser.parse_args()


def main() -> None:
    config = yaml.safe_load(arguments().config.read_text(encoding="utf-8"))
    data, model_config, training, checkpoint = (
        config["data"], config["model"], config["training"], config["checkpoint"]
    )
    set_seed(training["seed"])
    device = resolve_device(training["device"])
    size = model_config["image_size"]
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    train_loader, validation_loader, classes = make_loaders(
        data["train_dir"], data["validation_dir"], transform, training["batch_size"], training["num_workers"], training["seed"]
    )
    backbone = VisionTransformer(VisionTransformerConfig(**model_config))
    model = nn.Sequential(backbone, ClassificationHead(backbone.hidden_size, len(classes))).to(device)
    optimizer = build_optimizer(model, training["learning_rate"], training["weight_decay"])
    output, best_accuracy = Path(checkpoint["path"]), -1.0
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, training["epochs"] + 1):
        train_loss, train_accuracy = train_epoch(model, train_loader, device, optimizer)
        validation_loss, validation_accuracy = evaluate(model, validation_loader, device)
        print(f"epoch {epoch}: train={train_loss:.4f}/{train_accuracy:.4f} val={validation_loss:.4f}/{validation_accuracy:.4f}")
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save({"kind": "vit", "model": model.state_dict(), "classes": classes, "config": config}, output)


if __name__ == "__main__":
    main()
