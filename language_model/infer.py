"""Generate text with the trained Transformer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from .model.model import Transformer
from .model.tokenizer import ByteLevelBPETokenizer, load_tokenizer


def load_model(path: Path) -> Transformer:
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    model_config = dict(metadata["model"])
    model_config["vocabulary_size"] = metadata["data"]["vocabulary_size"]
    model = Transformer(seed=0, **model_config)
    model.load_state_dict(load_file(str(path)))
    return model.to("cuda" if torch.cuda.is_available() else "cpu").eval()


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
    max_new_tokens: int = 40,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int = 7,
    stop_token: str = "<|eos|>",
) -> str:
    token_ids = tokenizer.encode(prompt).ids
    stop_id = tokenizer.token_to_id(stop_token)
    rng = np.random.default_rng(seed)
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            inputs = torch.tensor(token_ids[-model.max_sequence_length :], dtype=torch.long, device=next(model.parameters()).device)[None]
            next_token_id = sample_token(model(inputs)[0, -1].cpu().numpy(), temperature, top_k, rng)
            if next_token_id == stop_id:
                break
            token_ids.append(next_token_id)
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def infer(prompt: str) -> str:
    model = load_model(Path("language_model/checkpoints/transformer.safetensors"))
    tokenizer = load_tokenizer(Path("language_model/data/encode/pretrain"), "byte_bpe")
    outputs = generate(model, tokenizer, prompt)
    return outputs
