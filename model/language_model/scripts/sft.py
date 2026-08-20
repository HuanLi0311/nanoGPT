"""Fine-tune a pre-trained Transformer on prepared SFT shards."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel

from ..model.model import Transformer
from ..model.training.loss import sft_loss
from ..model.training.optimizer import Optimizer


def main() -> None:
    config = yaml.safe_load(Path(sys.argv[1] if len(sys.argv) > 1 else "model/language_model/config/sft.yaml").read_text(encoding="utf-8"))
    data, training, checkpoint = config["data"], config["training"], config["checkpoint"]
    world_size, rank, local_rank = (int(os.environ.get(name, default)) for name, default in (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")))
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    encoded_dir, pretrained_path = Path(data["encoded_dir"]), Path(checkpoint["pretrained_path"])
    sft_metadata = json.loads((encoded_dir / "metadata.json").read_text(encoding="utf-8"))
    pretrained_metadata = json.loads(pretrained_path.with_suffix(".json").read_text(encoding="utf-8"))
    model_config, pretrained_data = pretrained_metadata["model"], pretrained_metadata["data"]

    shards = sorted(encoded_dir.glob("input_train_*.bin"))
    labels = sorted(encoded_dir.glob("labels_train_*.bin"))

    model = Transformer(int(pretrained_data["vocabulary_size"]), **model_config, seed=int(training["seed"]))
    if rank == 0:
        model.load_state_dict(load_file(str(pretrained_path)))
    model.to(device)
    wrapped = DistributedDataParallel(model, device_ids=[local_rank]) if world_size > 1 else model
    optimizer = Optimizer(wrapped.parameters(), float(training["learning_rate"]))
    length, batch_size = int(model_config["max_sequence_length"]), int(training["batch_size"])
    rng = np.random.default_rng(int(training["seed"]) + rank)
    if rank == 0:
        print(f"Device: {device}; world_size: {world_size}; shards: {len(shards)}")

    for step in range(1, int(training["steps"]) + 1):
        index = (step * world_size + rank) % len(shards)
        token_ids = np.memmap(shards[index], dtype=np.uint32, mode="r")
        target_ids = np.memmap(labels[index], dtype=np.int32, mode="r")
        starts = rng.integers(0, len(token_ids) - length - 1, size=batch_size)
        input_batch = np.stack([token_ids[start : start + length] for start in starts])
        target_batch = np.stack([target_ids[start + 1 : start + length + 1] for start in starts])
        inputs = torch.from_numpy(input_batch.astype(np.int64, copy=False)).to(device)
        targets = torch.from_numpy(target_batch.astype(np.int64, copy=False)).to(device)
        optimizer.zero_grad()

        logits = wrapped(inputs)
        loss = sft_loss(logits, targets)
        loss.backward()

        gradient_norm = float(torch.nn.utils.clip_grad_norm_(wrapped.parameters(), training["maximum_gradient_norm"]))
        optimizer.step()
        if rank == 0 and (step == 1 or step % training["log_every"] == 0 or step == training["steps"]):
            print(f"step {step:5d}/{training['steps']}: loss={loss.item():.6f} grad_norm={gradient_norm:.4f}")

    if rank == 0:
        output = Path(checkpoint["path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}, str(output))
        output.with_suffix(".json").write_text(json.dumps({"model": model_config, "training": training, "data": sft_metadata, "pretrained_path": str(pretrained_path)}, indent=2), encoding="utf-8")
        print(f"Saved model: {output}")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
