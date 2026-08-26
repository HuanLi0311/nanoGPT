#!/usr/bin/env python3
"""Inspect a safetensors checkpoint without loading tensor payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import tempfile
from pathlib import Path


def inspect_checkpoint(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("file is too small for safetensors header")
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length > path.stat().st_size - 8:
            raise ValueError("invalid safetensors header length")
        header = json.loads(handle.read(header_length).decode("utf-8"))
    tensors = []
    parameter_count = 0
    for name, info in sorted(header.items()):
        if name == "__metadata__":
            continue
        shape = info.get("shape")
        offsets = info.get("data_offsets")
        if not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError("unexpected tensor header for %s" % name)
        numel = 1
        for dimension in shape:
            numel *= int(dimension)
        parameter_count += numel
        tensors.append({"name": name, "dtype": info.get("dtype"), "shape": shape, "numel": numel, "data_offsets": offsets})
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "schema_version": 1,
        "format": "safetensors",
        "path": str(path.resolve()),
        "file_size": path.stat().st_size,
        "checkpoint_sha256": digest.hexdigest(),
        "header_bytes": header_length,
        "tensor_count": len(tensors),
        "parameter_count": parameter_count,
        "metadata": header.get("__metadata__", {}),
        "tensors": tensors,
    }


def self_check():
    with tempfile.TemporaryDirectory(prefix="skill-train-") as directory:
        path = Path(directory) / "tiny.safetensors"
        header = {"x": {"dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24]}}
        body = b"\0" * 24
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + body)
        result = inspect_checkpoint(path)
        assert result["parameter_count"] == 6 and result["tensor_count"] == 1
        assert len(result["checkpoint_sha256"]) == 64


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        self_check()
        print(json.dumps({"self_check": "ok"}))
        return 0
    if not args.checkpoint:
        parser.error("--checkpoint is required unless --self-check is used")
    result = inspect_checkpoint(args.checkpoint)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
