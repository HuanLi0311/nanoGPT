"""Train and use the byte-level BPE tokenizer selected by train.yaml."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ..settings import load_settings

try:
    from tokenizers import ByteLevelBPETokenizer
except ModuleNotFoundError as error:
    raise SystemExit("Missing dependency. Run: pip install tokenizers") from error


SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|bos|>", "<|eos|>"]


def model_paths(model_dir: Path, prefix: str) -> tuple[Path, Path]:
    return model_dir / f"{prefix}-vocab.json", model_dir / f"{prefix}-merges.txt"


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tokenizer(model_dir: Path, prefix: str) -> ByteLevelBPETokenizer:
    vocab_path, merges_path = model_paths(model_dir, prefix)
    if not vocab_path.is_file() or not merges_path.is_file():
        raise ValueError("Tokenizer model files are missing. Run tokenizer training first.")
    return ByteLevelBPETokenizer(str(vocab_path), str(merges_path))


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
    vocab_path, merges_path = model_paths(model_dir, prefix)
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Vocabulary: {vocab_path}")
    print(f"Merges: {merges_path}")


def encode(args: argparse.Namespace) -> None:
    tokenizer = load_tokenizer(args.model_dir, args.prefix)
    encoding = tokenizer.encode(args.text)
    print("input text:", json.dumps(args.text, ensure_ascii=True))
    print("tokens:", json.dumps(encoding.tokens, ensure_ascii=True))
    print("token IDs:", encoding.ids)
    print("decoded text:", json.dumps(tokenizer.decode(encoding.ids), ensure_ascii=True))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and use a byte-level BPE tokenizer.")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="Train the tokenizer described by train.yaml.")
    train_parser.add_argument("--config", type=Path, default=Path("language_model/config/train.yaml"))
    train_parser.set_defaults(handler=train)
    encode_parser = commands.add_parser("encode", help="Encode text with a saved tokenizer.")
    encode_parser.add_argument("--model-dir", type=Path, default=Path("language_model/tokenizer_model"))
    encode_parser.add_argument("--prefix", default="byte_bpe")
    encode_parser.add_argument("--text", required=True)
    encode_parser.set_defaults(handler=encode)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    args.handler(args)


if __name__ == "__main__":
    main()
