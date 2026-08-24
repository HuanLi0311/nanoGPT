"""Fail fast on the dependencies and artifacts required by the Qwen SFT launcher."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch


MODULES = (
    "accelerate",
    "datasets",
    "matplotlib",
    "numpy",
    "peft",
    "PIL",
    "pyarrow",
    "ray",
    "safetensors",
    "tokenizers",
    "transformers",
    "torchvision",
    "yaml",
    "verl.trainer.sft_trainer",
)


def check(data: Path, model: Path, allow_no_cuda: bool) -> None:
    for module in MODULES:
        importlib.import_module(module)
    from torch.distributed.fsdp import FullyShardedDataParallel

    assert FullyShardedDataParallel is not None
    if not allow_no_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; launch SFT from air-node-03 after activating nanoagent")
    for name in ("codex_train.parquet", "codex_test.parquet"):
        path = data / name
        if not path.is_file():
            raise SystemExit(f"missing SFT Parquet: {path}")
        table = pq.ParquetFile(path)
        if table.metadata.num_rows == 0 or "messages" not in table.schema_arrow.names:
            raise SystemExit(f"invalid SFT Parquet: {path}")
    for name in ("config.json", "tokenizer_config.json"):
        if not (model / name).is_file():
            raise SystemExit(f"missing Qwen checkpoint artifact: {model / name}")
    print(json.dumps({
        "cuda_available": torch.cuda.is_available(),
        "cuda_count": torch.cuda.device_count(),
        "data": str(data),
        "model": str(model),
        "torch": torch.__version__,
    }))


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=root / "model/language_model/data/post_train/data/rendered/sft")
    parser.add_argument("--model", type=Path, default=root / "model/language_model/checkpoints/qwen/Qwen3-8B-Base")
    parser.add_argument("--allow-no-cuda", action="store_true")
    args = parser.parse_args()
    check(args.data, args.model, args.allow_no_cuda)


if __name__ == "__main__":
    main()
