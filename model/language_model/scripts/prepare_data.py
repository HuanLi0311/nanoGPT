"""Download a tokenizer and encode pretraining or SFT shards."""

import json
import os
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pyarrow.parquet as pq
from tokenizers import ByteLevelBPETokenizer

from ..model.tokenizer import load_tokenizer
from .tool_message import message_content


####################################################################################
#                                    Tokenizer                                     #
####################################################################################
def tokenizer(data):
    directory, prefix = Path(data["tokenizer_dir"]), data["tokenizer_prefix"]
    if "tokenizer_repo" not in data:
        encoder = ByteLevelBPETokenizer()
        limit = int(data.get("tokenizer_train_documents", 100_000))

        def texts():
            seen = 0
            for source in sorted(Path(data["raw_dir"]).rglob("*.parquet")):
                column = "text" if "text" in pq.ParquetFile(source).schema_arrow.names else "raw_content"
                for batch in pq.ParquetFile(source).iter_batches(columns=[column], batch_size=4096):
                    values = batch.column(column).to_pylist()
                    yield values[:limit - seen]
                    seen += len(values)
                    if seen >= limit:
                        return

        directory.mkdir(parents=True, exist_ok=True)
        encoder.train_from_iterator(texts(), vocab_size=int(data["vocabulary_size"]), min_frequency=2, special_tokens=["<|eos|>", "<|im_start|>", "<|im_end|>"])
        encoder.save_model(str(directory), prefix)
        return
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("vocab.json", "merges.txt"):
        target = directory / f"{prefix}-{name}"
        url = f"{endpoint}/{data['tokenizer_repo']}/resolve/{data.get('tokenizer_revision', 'main')}/{name}"
        with urlopen(url, timeout=60) as response:
            target.with_suffix(target.suffix + ".tmp").write_bytes(response.read())
        target.with_suffix(target.suffix + ".tmp").replace(target)


####################################################################################
#                                     Pretrain                                     #
####################################################################################
def pretrain(data):
    directory, prefix = Path(data["tokenizer_dir"]), data["tokenizer_prefix"]
    encoder = load_tokenizer(directory, prefix)
    eos, limit, out = encoder.token_to_id(data["eos_token"]), data["shard_tokens"], Path(data["encoded_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.bin"):
        old.unlink()
    shards, counts = {"train": 0, "validation": 0}, {"train": 0, "validation": 0}
    buffers = {"train": [], "validation": []}
    for source in sorted(Path(data["raw_dir"]).rglob("*.parquet")):
        print(f"Encoding: {source}", flush=True)
        split = "validation" if source.name in data.get("validation_files", []) or "valid" in source.name.lower() else "train"
        parquet = pq.ParquetFile(source)
        column = "text" if "text" in parquet.schema_arrow.names else "raw_content"
        for batch in parquet.iter_batches(columns=[column], batch_size=4096):
            for encoded in encoder.encode_batch(batch.column(column).to_pylist()):
                buffers[split].extend((*encoded.ids, eos))
            while len(buffers[split]) >= limit:
                np.asarray(buffers[split][:limit], dtype=np.uint32).tofile(out / f"{split}_{shards[split]:05d}.bin")
                del buffers[split][:limit]
                shards[split] += 1
                counts[split] += limit
    for split, buffer in buffers.items():
        if buffer:
            np.asarray(buffer, dtype=np.uint32).tofile(out / f"{split}_{shards[split]:05d}.bin")
            shards[split] += 1
            counts[split] += len(buffer)
    (out / "metadata.json").write_text(json.dumps({"dtype": "uint32", "vocabulary_size": encoder.get_vocab_size(), "tokens": counts, "shards": shards}, indent=2))


####################################################################################
#                                       SFT                                        #
####################################################################################
def sft(data):
    directory, prefix = Path(data["tokenizer_dir"]), data["tokenizer_prefix"]
    encoder = load_tokenizer(directory, prefix)
    limit, out = data["shard_tokens"], Path(data["encoded_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.bin"):
        old.unlink()
    shards, counts = {"train": 0, "validation": 0}, {"train": 0, "validation": 0}
    for split, name in (("train", "train_sft"), ("validation", "test_sft")):
        ids, labels = [], []
        for source in sorted(Path(data["raw_dir"]).glob(f"{name}-*.parquet")):
            for batch in pq.ParquetFile(source).iter_batches(columns=["messages"], batch_size=512):
                texts, answers = [], []
                for messages in batch.column("messages").to_pylist():
                    text, answer = "", []
                    for message in messages:
                        content = message_content(message)
                        text += f"<|im_start|>{message['role']}\n"
                        start = len(text)
                        text += f"{content}<|im_end|>\n"
                        if message["role"] == "assistant":
                            answer.append((start, len(text)))
                    texts.append(text)
                    answers.append(answer)
                for encoded, answer in zip(encoder.encode_batch(texts), answers):
                    ids.extend(encoded.ids)
                    labels.extend(token if any(start <= offset[0] < end for start, end in answer) else -100 for token, offset in zip(encoded.ids, encoded.offsets))
                while len(ids) >= limit:
                    np.asarray(ids[:limit], dtype=np.uint32).tofile(out / f"input_{split}_{shards[split]:05d}.bin")
                    np.asarray(labels[:limit], dtype=np.int32).tofile(out / f"labels_{split}_{shards[split]:05d}.bin")
                    del ids[:limit], labels[:limit]
                    shards[split] += 1
                    counts[split] += limit
        if ids:
            np.asarray(ids, dtype=np.uint32).tofile(out / f"input_{split}_{shards[split]:05d}.bin")
            np.asarray(labels, dtype=np.int32).tofile(out / f"labels_{split}_{shards[split]:05d}.bin")
            shards[split] += 1
            counts[split] += len(ids)
    (out / "metadata.json").write_text(json.dumps({"input_dtype": "uint32", "label_dtype": "int32", "ignore_index": -100, "vocabulary_size": encoder.get_vocab_size(), "tokens": counts, "shards": shards, "chat_template": "<|im_start|>{role}\\n{content}<|im_end|>\\n"}, indent=2))
