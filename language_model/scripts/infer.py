"""Generate text with the trained Transformer."""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors.torch import load_file

from ..model.model import Transformer
from ..model.tokenizer import ByteLevelBPETokenizer, load_tokenizer



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
    stop_ids = {tokenizer.token_to_id(token) for token in (stop_token, "<|im_end|>")}
    rng = np.random.default_rng(seed)
    cache = None
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            inputs = token_ids[-model.max_sequence_length :] if cache is None else token_ids[-1:]
            inputs = torch.tensor(inputs, dtype=torch.long, device=next(model.parameters()).device)[None]
            logits, cache = model(inputs, cache, use_cache=True)
            next_token_id = sample_token(logits[0, -1].cpu().numpy(), temperature, top_k, rng)
            if next_token_id in stop_ids:
                break
            token_ids.append(next_token_id)
            if cache[0][0].shape[-2] == model.max_sequence_length:
                cache = None
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def infer(prompt: str, config_path: Path = Path("language_model/config/observatory_rl.yaml")) -> str:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    system = config["data"].get("system_prompt")
    if system:
        prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return generate(load_model(Path(config["checkpoint"]["path"])), load_tokenizer(Path(config["data"]["tokenizer_dir"]), config["data"]["tokenizer_prefix"]), prompt, **config.get("inference", {}))


def main() -> None:
    os.chdir(Path(__file__).parents[1])
    parser = ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--config", type=Path, default=Path("language_model/config/observatory_rl.yaml"))
    args = parser.parse_args()
    print(infer(args.prompt, args.config).split("assistant\n")[-1])


if __name__ == "__main__":
    main()
