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
from ..model.training.loss import pretrain_loss
from ..model.training.optimizer import Optimizer


def validate(model, shards, length, batch_size, batches, device):
    rng, losses = np.random.default_rng(0), []
    model.eval()
    with torch.inference_mode():
        for index in range(batches):
            tokens = np.memmap(shards[index % len(shards)], dtype=np.uint32, mode="r")
            starts = rng.integers(0, len(tokens) - length - 1, size=batch_size)
            batch = np.stack([tokens[start : start + length + 1] for start in starts])
            batch = torch.from_numpy(batch.astype(np.int64, copy=False)).to(device)
            losses.append(pretrain_loss(model(batch[:, :-1]), batch[:, 1:]).item())
    model.train()
    return float(np.mean(losses))


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
    validation_shards = sorted(encoded_dir.glob("validation_*.bin"))
    if not validation_shards:
        raise ValueError("validation shards are required to save the best model")
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
    resume_path = Path(checkpoint["path"]).with_suffix(".resume.pt")
    best_path = Path(checkpoint["best_path"])
    save_every = int(training.get("save_every", 0))
    start_step = 0
    if rank == 0:
        resume_path.parent.mkdir(parents=True, exist_ok=True)
        best_path.parent.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    if resume_path.exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        best_loss = state.get("best_loss", float("inf"))
    else:
        best_loss = float("inf")
    if rank == 0:
        print(f"Device: {device}; world_size: {world_size}; shards: {len(shards)}")

    for step in range(start_step + 1, int(training["steps"]) + 1):
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
        if save_every and step % save_every == 0:
            if rank == 0:
                validation_loss = validate(model, validation_shards, sequence_length, batch_size, int(training["validation_batches"]), device)
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    temporary = best_path.with_suffix(best_path.suffix + ".tmp")
                    save_file({name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}, str(temporary))
                    temporary.replace(best_path)
                    best_path.with_suffix(".json").write_text(json.dumps({"model": model_config, "training": training, "data": metadata, "step": step, "validation_loss": best_loss}, indent=2), encoding="utf-8")
                    print(f"step {step:5d}: validation_loss={validation_loss:.6f}; saved best model")
                temporary = resume_path.with_suffix(".tmp")
                torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_loss": best_loss}, temporary)
                temporary.replace(resume_path)
            if world_size > 1:
                dist.barrier()
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
