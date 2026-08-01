"""Load a trained Transformer checkpoint and generate text from a prompt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .model.model import Transformer
from .model.embeddings.tokenizer import ByteLevelBPETokenizer, load_tokenizer, tokenizer_fingerprint


def load_model(path: Path) -> tuple[Transformer, dict[str, object]]:
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        raise ValueError("Model .npz and matching metadata .json files must both exist.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_config = metadata.get("model")
    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint metadata is missing the model configuration.")
    model = Transformer(seed=0, **model_config)
    with np.load(path) as checkpoint:
        try:
            model.load_state_dict(
                {name: torch.from_numpy(checkpoint[name]) for name in checkpoint.files}
            )
        except RuntimeError as error:
            raise ValueError("Checkpoint parameters do not match the saved model configuration.") from error
    model.eval()
    return model, metadata


def sample_token(
    logits: np.ndarray, temperature: float, top_k: int | None, rng: np.random.Generator
) -> int:
    if temperature <= 0 or top_k is not None and top_k <= 0:
        raise ValueError("temperature and top_k must be positive.")
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
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative.")
    token_ids = tokenizer.encode(prompt).ids
    if not token_ids:
        raise ValueError("Prompt must produce at least one token.")
    stop_id = tokenizer.get_vocab().get(stop_token)
    rng = np.random.default_rng(seed)
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            inputs = torch.tensor(token_ids[-model.max_sequence_length :], dtype=torch.long)[None]
            logits = model(inputs)
            next_token_id = sample_token(logits[0, -1].cpu().numpy(), temperature, top_k, rng)
            if next_token_id == stop_id:
                break
            token_ids.append(next_token_id)
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained NumPy Transformer checkpoint.")
    parser.add_argument("--model", type=Path, default=Path("language_model/checkpoints/demo_transformer.npz"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("language_model/tokenizer_model"))
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
    if metadata.get("tokenizer") != tokenizer_fingerprint(args.tokenizer_dir, args.tokenizer_prefix):
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
