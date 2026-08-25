"""Remove selected non-agent topics from the legacy prompt-only SFT shards."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


LEGACY = re.compile(r"train_sft-\d{5}-of-\d{5}-[0-9a-f]+\.parquet")
TOPIC_PATTERNS = {
    "business_product_marketing": re.compile(
        r"\b(shopify|theme|business|marketing|customer|product|sales|finance|company|market|startup|"
        r"advertis(?:e|ing|ement)?|retail|e-commerce|entrepreneur|brand|pricing|accounting|investment)\b",
        re.IGNORECASE,
    ),
    "lifestyle_home_shopping": re.compile(
        r"\b(organize|closet|recipe|cook|home|fashion|shopping|gift|relationship|garden|clothes|shoe|"
        r"parenting|household|furniture|decor|cleaning)\b",
        re.IGNORECASE,
    ),
    "travel_geography_attractions": re.compile(
        r"\b(travel|tourist|landmark|hotel|london|paris|trip|destination|city|country|visit|museum|"
        r"vacation|tourism|geography|attraction)\b",
        re.IGNORECASE,
    ),
}


def topic_hits(prompt: str) -> list[str]:
    return [name for name, pattern in TOPIC_PATTERNS.items() if pattern.search(prompt)]


def iter_filtered(path: Path, *, write_to: Path | None = None) -> tuple[Counter, int, int]:
    parquet = pq.ParquetFile(path)
    counts: Counter = Counter()
    kept = dropped = 0
    writer = pq.ParquetWriter(write_to, parquet.schema_arrow, compression="zstd") if write_to else None
    try:
        for batch in parquet.iter_batches(batch_size=2048):
            prompts = batch.column("prompt").to_pylist()
            keep = []
            for prompt in prompts:
                hits = topic_hits(prompt or "")
                if hits:
                    dropped += 1
                    counts["dropped_union"] += 1
                    counts.update(hits)
                    keep.append(False)
                else:
                    kept += 1
                    keep.append(True)
            if writer and any(keep):
                writer.write_batch(batch.filter(pa.array(keep)))
    finally:
        if writer:
            writer.close()
    counts["input"] += kept + dropped
    counts["kept"] += kept
    return counts, kept, dropped


def main() -> None:
    root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=root / "model/language_model/data/post_train/data/rendered/sft")
    parser.add_argument("--archive-dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    sources = sorted(path for path in args.input_dir.glob("train_sft-*.parquet") if LEGACY.fullmatch(path.name))
    if not sources:
        raise SystemExit("no legacy train_sft shards found")

    report = {"filter_version": "legacy-topic-v1", "files": {}, "totals": Counter()}
    for source in sources:
        counts, _, _ = iter_filtered(source)
        report["files"][source.name] = dict(counts)
        report["totals"].update(counts)
    report["totals"] = dict(report["totals"])
    if args.check_only:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    archive = args.archive_dir or args.input_dir / "archive/legacy_train_sft_excluded"
    temporary = args.input_dir / ".legacy_sft_filter_tmp"
    if temporary.exists() or archive.exists():
        raise SystemExit(f"refusing to reuse existing path: {temporary if temporary.exists() else archive}")
    temporary.mkdir(parents=True)
    try:
        for source in sources:
            target = temporary / source.name
            counts, _, _ = iter_filtered(source, write_to=target)
            if dict(counts) != report["files"][source.name]:
                raise RuntimeError(f"non-deterministic filter result for {source.name}")
            check, _, dropped_after = iter_filtered(target)
            if dropped_after:
                raise RuntimeError(f"filtered shard still matches excluded topics: {source.name}")
            if check["input"] != counts["kept"]:
                raise RuntimeError(f"row-count mismatch for {source.name}")

        archive.mkdir(parents=True)
        for source in sources:
            shutil.move(str(source), archive / source.name)
            os.replace(temporary / source.name, source)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

    report["archive_dir"] = str(archive)
    report_path = args.input_dir / "legacy_sft_filter_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
