"""Train the one RoPE Transformer defined in model/model.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..model.model import Transformer
from ..model.embeddings.tokenizer import file_digest, load_tokenizer, tokenizer_fingerprint
from ..settings import load_settings
from .optimizer import Optimizer


def random_batch(
    token_ids: np.ndarray, sequence_length: int, batch_size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    maximum_start = len(token_ids) - sequence_length
    if maximum_start <= 0:
        raise ValueError("Corpus has too few tokens for the chosen sequence length.")
    starts = rng.integers(0, maximum_start, size=batch_size)
    return (
        np.stack([token_ids[start : start + sequence_length] for start in starts]),
        np.stack([token_ids[start + 1 : start + sequence_length + 1] for start in starts]),
    )


def save_model(model: Transformer, path: Path, metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parameters = {
        name: parameter.detach().cpu().numpy()
        for name, parameter in model.named_parameters()
    }
    np.savez_compressed(path, **parameters)
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Transformer described by train.yaml.")
    parser.add_argument("--config", type=Path, default=Path("language_model/config/train.yaml"))
    return parser.parse_args()


def main() -> None:
    config = load_settings(parse_arguments().config)
    data = config["data"]
    model_config = config["model"]
    training = config["training"]
    checkpoint = config["checkpoint"]
    corpus_path = Path(data["corpus"])
    if not corpus_path.is_file():
        raise ValueError(f"Corpus file does not exist: {corpus_path}")
    tokenizer = load_tokenizer(Path(data["tokenizer_dir"]), data["tokenizer_prefix"])
    if tokenizer.get_vocab_size() != data["vocabulary_size"]:
        raise ValueError("Configured vocabulary_size does not match the trained tokenizer.")
    token_ids = np.asarray(
        tokenizer.encode(corpus_path.read_text(encoding="utf-8")).ids, dtype=np.int64
    )
    model = Transformer(
        tokenizer.get_vocab_size(),
        model_config["max_sequence_length"],
        model_config["hidden_size"],
        model_config["heads"],
        model_config["blocks"],
        training["seed"],
    )
    optimizer = Optimizer(model.parameters(), training["learning_rate"])
    rng = np.random.default_rng(training["seed"] + 100)
    print(f"Corpus tokens: {len(token_ids)}")

    for step in range(1, training["steps"] + 1):
        inputs, targets = random_batch(
            token_ids, model.max_sequence_length, training["batch_size"], rng
        )
        inputs = torch.from_numpy(inputs)
        targets = torch.from_numpy(targets)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), training["maximum_gradient_norm"]
            )
        )
        optimizer.step(training["optimizer"])
        if step == 1 or step % training["log_every"] == 0 or step == training["steps"]:
            print(
                f"step {step:5d}/{training['steps']}: "
                f"loss={loss.item():.6f} grad_norm={gradient_norm:.4f}"
            )

    output_path = Path(checkpoint["path"])
    save_model(
        model,
        output_path,
        {
            "model": model.configuration,
            "training": training,
            "tokenizer": tokenizer_fingerprint(
                Path(data["tokenizer_dir"]), data["tokenizer_prefix"]
            ),
            "corpus_sha256": file_digest(corpus_path),
        },
    )
    print(f"Saved model: {output_path}")

if __name__ == "__main__":
    main()
