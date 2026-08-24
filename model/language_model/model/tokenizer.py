"""Load and encode with the byte-level BPE tokenizer."""

from __future__ import annotations

import hashlib
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from torch import Tensor


def model_paths(model_dir: Path, prefix: str) -> tuple[Path, Path]:
    return model_dir / f"{prefix}-vocab.json", model_dir / f"{prefix}-merges.txt"


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokenizer_fingerprint(model_dir: Path, prefix: str) -> dict[str, str]:
    vocab_path, merges_path = model_paths(model_dir, prefix)
    return {"vocab": file_digest(vocab_path), "merges": file_digest(merges_path)}


def load_tokenizer(model_dir: Path, prefix: str) -> ByteLevelBPETokenizer:
    vocab_path, merges_path = model_paths(model_dir, prefix)
    tokenizer = ByteLevelBPETokenizer(str(vocab_path), str(merges_path))
    tokenizer.add_special_tokens([token for token in ("<|eos|>", "<|im_start|>", "<|im_end|>") if tokenizer.token_to_id(token) is not None])
    return tokenizer


def embedding(token_embedding: Tensor, token_ids: Tensor) -> Tensor:
    hidden_states = token_embedding[token_ids]
    return hidden_states
