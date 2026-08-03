"""Train the Transformer from pre-tokenized uint32 shards with torchrun DDP."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from safetensors.torch import save_file
from torch.nn.parallel import DistributedDataParallel

from ..model.model import Transformer
from ..training.loss import pretrain_loss
from ..training.optimizer import Optimizer


def main() -> None:
    config = yaml.safe_load(Path(sys.argv[1] if len(sys.argv) > 1 else "language_model/config/pretrain.yaml").read_text(encoding="utf-8"))
    data, model_config, training, checkpoint = (
        config["data"], config["model"], config["training"], config["checkpoint"]
    )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    encoded_dir = Path(data["encoded_dir"])
    metadata = json.loads((encoded_dir / "metadata.json").read_text(encoding="utf-8"))
    shards = sorted(encoded_dir.glob("train_*.bin"))
    sequence_length = int(model_config["max_sequence_length"])
    batch_size = int(training["batch_size"])
    model = Transformer(
        int(metadata["vocabulary_size"]), sequence_length, int(model_config["hidden_size"]),
        int(model_config["heads"]), int(model_config["blocks"]),
        dropout=float(model_config["dropout"]), seed=int(training["seed"]),
    ).to(device)
    wrapped = DistributedDataParallel(model, device_ids=[local_rank]) if world_size > 1 else model
    optimizer = Optimizer(wrapped.parameters(), float(training["learning_rate"]))
    rng = np.random.default_rng(int(training["seed"]) + rank)
    if rank == 0:
        print(f"Device: {device}; world_size: {world_size}; shards: {len(shards)}")

    for step in range(1, int(training["steps"]) + 1):
        tokens = np.memmap(shards[(step * world_size + rank) % len(shards)], dtype=np.uint32, mode="r")
        maximum_start = len(tokens) - sequence_length - 1

        starts = rng.integers(0, maximum_start, size=batch_size)
        batch = np.stack([tokens[start : start + sequence_length + 1] for start in starts])
        optimizer.zero_grad()
        inputs = torch.from_numpy(batch[:, :-1].astype(np.int64, copy=False)).to(device)
        targets = torch.from_numpy(batch[:, 1:].astype(np.int64, copy=False)).to(device)
        logits = wrapped(inputs)
        loss = pretrain_loss(logits, targets)
        loss.backward()

        gradient_norm = float(torch.nn.utils.clip_grad_norm_(wrapped.parameters(), training["maximum_gradient_norm"]))
        optimizer.step()
        if rank == 0 and (step == 1 or step % training["log_every"] == 0 or step == training["steps"]):
            print(f"step {step:5d}/{training['steps']}: loss={loss.item():.6f} grad_norm={gradient_norm:.4f}")

    if rank == 0:
        output = Path(checkpoint["path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}, str(output))
        output.with_suffix(".json").write_text(json.dumps({"model": model_config, "training": training, "data": metadata}, indent=2), encoding="utf-8")
        print(f"Saved model: {output}")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
