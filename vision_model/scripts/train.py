"""Train a frozen SigLIP2 vision encoder with a linear classification probe."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor

from ..model.encoder.backbone import PretrainedVisionBackbone
from ..model.heads.classification import LinearProbe
from ..settings import load_settings, resolve_device, set_seed
from ..model.training.loss import cross_entropy
from ..model.training.optimizer import build_optimizer


class ProcessedImageFolder(Dataset):
    """Apply a Hugging Face image processor to an ImageFolder sample."""

    def __init__(self, root: str | Path, processor: Any) -> None:
        from torchvision.datasets import ImageFolder

        self.dataset = ImageFolder(str(root))
        self.processor = processor

    @property
    def classes(self) -> list[str]:
        return list(self.dataset.classes)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image, label = self.dataset[index]
        encoded = self.processor(images=image, return_tensors="pt")
        pixel_values = encoded.get("pixel_values")
        if not isinstance(pixel_values, Tensor) or pixel_values.ndim != 4:
            raise ValueError("The image processor must return [1, channels, height, width].")
        return pixel_values.squeeze(0), torch.tensor(label, dtype=torch.long)


def _make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        pin_memory=device.type == "cuda",
    )


def _run_epoch(
    backbone: nn.Module,
    probe: LinearProbe,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one train or evaluation epoch and return mean loss and accuracy."""
    training = optimizer is not None
    backbone.eval()
    probe.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for pixel_values, targets in loader:
        pixel_values = pixel_values.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            features = backbone(pixel_values)
        logits = probe(features)
        loss = cross_entropy(logits, targets)
        if training:
            loss.backward()
            optimizer.step()

        batch_size = targets.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += int((logits.argmax(dim=-1) == targets).sum().item())
        total_examples += batch_size

    if total_examples == 0:
        raise ValueError("The data loader produced no examples.")
    return total_loss / total_examples, total_correct / total_examples


def _save_checkpoint(
    path: Path,
    backbone: PretrainedVisionBackbone,
    probe: LinearProbe,
    classes: list[str],
    settings: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    head_state = {name: value.detach().cpu() for name, value in probe.state_dict().items()}
    torch.save(
        {
            "backbone_name": backbone.model_name,
            "backbone_revision": backbone.revision,
            "hidden_size": backbone.hidden_size,
            "classes": classes,
            "head_state_dict": head_state,
            "settings": settings,
        },
        path,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SigLIP2 linear probe.")
    parser.add_argument("--config", type=Path, default=Path("vision_model/config/train.yaml"))
    return parser.parse_args()


def main() -> None:
    config = load_settings(parse_arguments().config)
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    checkpoint_config = config["checkpoint"]

    seed = int(training_config.get("seed", 7))
    set_seed(seed)
    device = resolve_device(str(training_config.get("device", "auto")))
    model_name = str(model_config["name"])
    revision = model_config.get("revision")
    pretrained_kwargs = {} if revision is None else {"revision": str(revision)}
    processor = AutoImageProcessor.from_pretrained(model_name, **pretrained_kwargs)

    train_dataset = ProcessedImageFolder(data_config["train_dir"], processor)
    validation_dataset = ProcessedImageFolder(data_config["validation_dir"], processor)
    if not train_dataset.classes:
        raise ValueError("The training directory contains no class folders.")
    if train_dataset.classes != validation_dataset.classes:
        raise ValueError("Training and validation class folders must have the same ordering.")

    batch_size = int(training_config.get("batch_size", 4))
    epochs = int(training_config.get("epochs", 5))
    num_workers = int(training_config.get("num_workers", 0))
    if min(batch_size, epochs) <= 0 or num_workers < 0:
        raise ValueError("batch_size and epochs must be positive; num_workers cannot be negative.")

    train_loader = _make_loader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=seed,
        device=device,
    )
    validation_loader = _make_loader(
        validation_dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        seed=seed + 1,
        device=device,
    )

    backbone = PretrainedVisionBackbone(model_name, revision=str(revision) if revision else None)
    probe = LinearProbe(backbone.hidden_size, len(train_dataset.classes))
    backbone.to(device)
    probe.to(device)
    optimizer = build_optimizer(
        probe,
        learning_rate=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 1e-2)),
    )

    checkpoint_path = Path(checkpoint_config["path"])
    best_accuracy = -1.0
    print(f"Device: {device}")
    print(f"Backbone: {model_name}")
    print(f"Classes: {', '.join(train_dataset.classes)}")
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = _run_epoch(
            backbone, probe, train_loader, device, optimizer=optimizer
        )
        validation_loss, validation_accuracy = _run_epoch(
            backbone, probe, validation_loader, device
        )
        print(
            f"epoch {epoch:3d}/{epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={validation_loss:.4f} val_acc={validation_accuracy:.4f}"
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            _save_checkpoint(
                checkpoint_path,
                backbone,
                probe,
                train_dataset.classes,
                settings=config,
            )
            print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
