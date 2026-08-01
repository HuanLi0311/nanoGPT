"""Train and use the byte-level BPE tokenizer selected by train.yaml."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from ...settings import load_settings
from tokenizers import ByteLevelBPETokenizer


SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|bos|>", "<|eos|>"]


def model_paths(model_dir: Path, prefix: str) -> tuple[Path, Path]:
    return model_dir / f"{prefix}-vocab.json", model_dir / f"{prefix}-merges.txt"


def file_digest(path: Path) -> str:
   return hashlib.sha256(path.read_bytes()).hexdigest()


def tokenizer_fingerprint(model_dir: Path, prefix: str) -> dict[str, str]:
    vocab_path, merges_path = model_paths(model_dir, prefix)
    return {"vocab": file_digest(vocab_path), "merges": file_digest(merges_path)}


def train(args: argparse.Namespace) -> None:
    data = load_settings(args.config)["data"]
    corpus = Path(data["corpus"])
    vocabulary_size = data["vocabulary_size"]
    min_frequency = data["min_frequency"]
    model_dir = Path(data["tokenizer_dir"])
    prefix = data["tokenizer_prefix"]
    if not corpus.is_file():
        raise ValueError(f"Corpus file does not exist: {corpus}")
    if vocabulary_size < 256 or min_frequency < 1:
        raise ValueError(
            "vocabulary_size must be at least 256 and min_frequency must be positive."
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(corpus)],
        vocab_size=vocabulary_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.save_model(str(model_dir), prefix=prefix)


def load_tokenizer(model_dir: Path, prefix: str) -> ByteLevelBPETokenizer:
    vocab_path, merges_path = model_paths(model_dir, prefix)
    return ByteLevelBPETokenizer(str(vocab_path), str(merges_path))


def encode(args: argparse.Namespace):
    tokenizer = load_tokenizer(args.model_dir, args.prefix)
    token_ids = tokenizer.encode(args.text).ids
    return token_ids
