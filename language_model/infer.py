"""Load a trained Transformer checkpoint and generate text from a prompt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .model.model import Transformer
from .model.tokenizer import ByteLevelBPETokenizer, encode, load_tokenizer, tokenizer_fingerprint


def load_model(path: Path) -> tuple[Transformer, dict[str, object]]:
    metadata_path = path.with_suffix(".json")
    model_config["vocabulary_size"] = int(data["vocabulary_size"])
    model = Transformer(seed=0, **model_config)
    with np.load(path) as checkpoint:
        model.load_state_dict(
            {name: torch.from_numpy(checkpoint[name]) for name in checkpoint.files}
        )

    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return model, metadata


def sample_token(
    logits: np.ndarray, temperature: float, top_k: int | None, rng: np.random.Generator
) -> int:
    scores = logits.astype(np.float64) / temperature
    if top_k is not None:
        count = min(top_k, len(scores))
        kept_ids = np.argpartition(scores, -count)[-count:]
        filtered = np.full_like(scores, -np.inf)
        filtered[kept_ids] = scores[kept_ids]
        scores = filtered
    probabilities = np.exp(scores - scores.max())
    return int(rng.choice(len(scores), p=probabilities / probabilities.sum()))


def generate(
    model: Transformer,
    tokenizer: ByteLevelBPETokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    seed: int,
    stop_token: str,
) -> str:

    stop_id = tokenizer.get_vocab().get(stop_token)
    rng = np.random.default_rng(seed)
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            inputs = torch.tensor(
                token_ids[-model.max_sequence_length :], dtype=torch.long, device=next(model.parameters()).device
            )[None]
            logits = model(inputs)
            next_token_id = sample_token(logits[0, -1].cpu().numpy(), temperature, top_k, rng)
            if next_token_id == stop_id:
                break
            token_ids.append(next_token_id)
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained NumPy Transformer checkpoint.")
    parser.add_argument("--model", type=Path, default=Path("language_model/checkpoints/demo_transformer.npz"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("language_model/data"))
    parser.add_argument("--tokenizer-prefix", default="byte_bpe")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stop-token", default="<|eos|>")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    model, metadata = load_model(args.model)
    tokenizer = load_tokenizer(args.tokenizer_dir, args.tokenizer_prefix)
    expected_tokenizer = metadata.get("tokenizer")
    if expected_tokenizer is None and isinstance(metadata.get("data"), dict):
        expected_tokenizer = metadata["data"].get("tokenizer")
    if expected_tokenizer != tokenizer_fingerprint(args.tokenizer_dir, args.tokenizer_prefix):
        raise ValueError("Tokenizer files do not match the checkpoint.")
    text = generate(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.seed,
        args.stop_token,
    )
    try:
        text.encode(sys.stdout.encoding or "utf-8")
    except UnicodeEncodeError:
        print(json.dumps(text, ensure_ascii=True))
    else:
        print(text)


if __name__ == "__main__":
    main()
