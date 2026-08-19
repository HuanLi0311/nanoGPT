"""GRPO on compact final plans; rewards inspect plans, never hidden reasoning."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file

from ..model.model import Transformer
from ..model.tokenizer import load_tokenizer
from ..model.training.optimizer import Optimizer


def chat(prompt: str, system: str) -> str:
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def reward(text: str, answer: str) -> float:
    return sum(part in text for part in answer.split(";")) / 4


def sample(model, prompt, stops, limit, temperature):
    ids, generated = prompt[:], []
    cache = None
    with torch.inference_mode():
        for _ in range(limit):
            inputs = ids[-model.max_sequence_length :] if cache is None else ids[-1:]
            logits, cache = model(torch.tensor(inputs, device=next(model.parameters()).device)[None], cache, use_cache=True)
            logits = logits[0, -1]
            token = int(torch.multinomial(torch.softmax(logits / temperature, -1), 1))
            if token in stops:
                break
            ids.append(token)
            generated.append(token)
            cache = None if cache[0][0].shape[-2] == model.max_sequence_length else cache
    return generated


def log_probability(model, prompt, completion):
    ids = prompt + completion
    logits = model(torch.tensor(ids[:-1], device=next(model.parameters()).device)[None])[0, len(prompt) - 1 :]
    targets = torch.tensor(completion, device=logits.device)
    return torch.log_softmax(logits, -1).gather(1, targets[:, None]).mean()


def main() -> None:
    config = yaml.safe_load(Path(sys.argv[1] if len(sys.argv) > 1 else "language_model/config/rl.yaml").read_text(encoding="utf-8"))
    data, training, checkpoint = config["data"], config["training"], config["checkpoint"]
    torch.manual_seed(int(training["seed"]))
    device, path = torch.device("cuda" if torch.cuda.is_available() else "cpu"), Path(checkpoint["sft_path"])
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    model = Transformer(int(metadata["data"]["vocabulary_size"]), **metadata["model"], seed=0).to(device)
    model.load_state_dict(load_file(str(path)))
    reference = copy.deepcopy(model).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    tokenizer = load_tokenizer(Path(data["tokenizer_dir"]), data["tokenizer_prefix"])
    rows = [json.loads(line) for line in Path(data["path"]).read_text(encoding="utf-8").splitlines()]
    stops = {tokenizer.token_to_id(token) for token in ("<|eos|>", "<|im_end|>")}
    optimizer, group = Optimizer(model.parameters(), float(training["learning_rate"])), int(training["group_size"])
    for step in range(1, int(training["steps"]) + 1):
        row = rows[(step - 1) % len(rows)]
        prompt = tokenizer.encode(chat(row["prompt"], data["system_prompt"])).ids
        model.eval()
        completions = [sample(model, prompt, stops, int(training["max_new_tokens"]), float(training["temperature"])) for _ in range(group)]
        completions = [item if item else [tokenizer.token_to_id("<|eos|>")] for item in completions]
        texts = [tokenizer.decode(item, skip_special_tokens=True) for item in completions]
        rewards = torch.tensor([reward(text, row["answer"]) for text in texts], device=device)
        advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-4)
        with torch.inference_mode():
            old = [log_probability(model, prompt, item) for item in completions]
            ref = [log_probability(reference, prompt, item) for item in completions]
        for _ in range(int(training["epochs"])):
            model.train()
            losses = []
            for advantage, completion, old_logp, ref_logp in zip(advantages, completions, old, ref):
                current = log_probability(model, prompt, completion)
                ratio = (current - old_logp).exp()
                clipped = ratio.clamp(1 - float(training["clip_range"]), 1 + float(training["clip_range"]))
                kl = (ref_logp - current).exp() - (ref_logp - current) - 1
                losses.append(-torch.minimum(ratio * advantage, clipped * advantage) + float(training["kl_coefficient"]) * kl)
            optimizer.zero_grad()
            torch.stack(losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["maximum_gradient_norm"]))
            optimizer.step()
        if step == 1 or step % int(training["log_every"]) == 0 or step == int(training["steps"]):
            print(f"step {step:4d}/{training['steps']}: reward={rewards.mean().item():.3f} answer={texts[0]!r}")
    output = Path(checkpoint["path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}, str(output))
    output.with_suffix(".json").write_text(json.dumps({"model": metadata["model"], "data": metadata["data"], "training": training, "sft_path": str(path)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
