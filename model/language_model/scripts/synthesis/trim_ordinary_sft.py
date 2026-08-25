"""Trim non-coding rows from the prompt-only ordinary SFT shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


CODING_TERMS = (
    "python", "javascript", "typescript", "java", "c++", "c#", "ruby", "php", "swift",
    "kotlin", "rust", "golang", "matlab", "bash", "shell script", "powershell", "sql",
    "html", "css", "json", "yaml", "xml", "regex", "source code", "codebase", "code snippet",
    "programming", "software development", "software engineer", "developer", "repository", "repo",
    "github", "gitlab", "version control", "pull request", "debugging", "debugger", "compile",
    "compiler", "runtime", "package manager", "framework", "sdk", "cli", "command line", "terminal",
    "unit test", "test suite", "backend", "front-end", "web app", "web development", "app development",
    "machine learning", "deep learning", "neural network", "computer science", "operating system",
    "linux", "docker", "kubernetes", "cloud computing", "cybersecurity", "encryption", "npm", "pip",
    "conda", "maven", "gradle", "cargo",
)
CODING_RE = re.compile(
    r"(?:```|^\s*(?:def|class|import|from|select|function)\b|"
    + "|".join(re.escape(term) for term in CODING_TERMS)
    + r")",
    re.IGNORECASE | re.MULTILINE,
)
CODING_CONTEXT_RE = re.compile(
    r"\b(?:write|create|build|implement|develop|design|fix|debug|refactor|run|execute)\b"
    r".{0,100}\b(?:program|function|script|class|algorithm|code|software|application|app|"
    r"website|api|database|query|module)\b|"
    r"\b(?:program|function|script|class|algorithm|code|software|application|app|website|"
    r"api|database|query|module)\b.{0,100}\b(?:write|create|build|implement|develop|design|"
    r"fix|debug|refactor|run|execute)\b",
    re.IGNORECASE | re.DOTALL,
)

TOPICS = (
    ("travel", r"\b(travel|tourist|landmark|hotel|london|paris|trip|destination|city|country|visit|museum|vacation|tourism|geography|attraction)\b"),
    ("lifestyle", r"\b(organize|closet|recipe|cook|home|fashion|shopping|gift|relationship|garden|clothes|shoe|parenting|household|furniture|decor|cleaning)\b"),
    ("business", r"\b(shopify|theme|business|marketing|customer|product|sales|finance|company|market|startup|advertis(?:e|ing|ement)?|retail|e-commerce|entrepreneur|brand|pricing|accounting|investment)\b"),
    ("creative", r"\b(essay|poem|poetry|story|fiction|novel|character|creative writing|dialogue|speech|presentation|screenplay)\b"),
    ("education", r"\b(education|school|student|teacher|classroom|university|college|curriculum|learning|research|thesis|academic|professor|science education)\b"),
    ("health", r"\b(health|medical|medicine|disease|doctor|patient|hospital|symptom|diagnos|therapy|mental health|nutrition|exercise|diet|cancer|surgery|vaccine|drug|medication)\b"),
    ("law_politics", r"\b(law|legal|court|judge|government|president|senate|congress|policy|politic|election|campaign|rights|regulation|news|journalist)\b"),
)
TOPIC_RES = tuple((name, re.compile(pattern, re.IGNORECASE)) for name, pattern in TOPICS)
PRIORITY = {name: index for index, (name, _) in enumerate(TOPICS)}
PRIORITY["other_noncoding"] = len(PRIORITY)


def is_coding(prompt: str) -> bool:
    return bool(CODING_RE.search(prompt) or CODING_CONTEXT_RE.search(prompt))


def topic(prompt: str) -> str:
    for name, pattern in TOPIC_RES:
        if pattern.search(prompt):
            return name
    return "other_noncoding"


def ordinary_paths(root: Path, split: str) -> list[Path]:
    prefix = f"{split}_sft-"
    return sorted(path for path in root.glob(f"{prefix}*.parquet") if "synthesis" not in path.name)


def collect(root: Path, split: str) -> tuple[list[dict], Counter]:
    rows: list[dict] = []
    counts: Counter = Counter()
    for path in ordinary_paths(root, split):
        table = pq.read_table(path, columns=["prompt", "prompt_id"])
        for row_index, (prompt, prompt_id) in enumerate(zip(table["prompt"].to_pylist(), table["prompt_id"].to_pylist())):
            text = prompt or ""
            if is_coding(text):
                counts["coding_guard"] += 1
                continue
            label = topic(text)
            counts[label] += 1
            rows.append({
                "file": path.name,
                "row": row_index,
                "prompt_id": str(prompt_id),
                "topic": label,
            })
    return rows, counts


def select(rows: list[dict], target: int, split: str) -> list[dict]:
    ordered = sorted(
        rows,
        key=lambda row: (
            PRIORITY[row["topic"]],
            hashlib.sha256(f"{split}:{row['prompt_id']}".encode()).hexdigest(),
        ),
    )
    if len(ordered) < target:
        raise RuntimeError(f"{split}: only {len(ordered)} safe non-coding rows, need {target}")
    selected = ordered[:target]
    if any(row["topic"] == "coding_guard" for row in selected):
        raise AssertionError("coding row selected")
    return selected


def rewrite(root: Path, split: str, selected: list[dict], archive: Path) -> dict:
    by_file: dict[str, set[int]] = {}
    for row in selected:
        by_file.setdefault(row["file"], set()).add(row["row"])
    temp = root / f".{split}_ordinary_trim_tmp"
    if temp.exists():
        raise RuntimeError(f"refusing to reuse temporary path: {temp}")
    temp.mkdir()
    stats = {}
    try:
        for source in ordinary_paths(root, split):
            parquet = pq.ParquetFile(source)
            target = temp / source.name
            writer = pq.ParquetWriter(target, parquet.schema_arrow, compression="zstd")
            input_rows = kept_rows = 0
            offset = 0
            try:
                for batch in parquet.iter_batches(batch_size=2048):
                    size = batch.num_rows
                    removed = by_file.get(source.name, set())
                    keep = [row_index not in removed for row_index in range(offset, offset + size)]
                    writer.write_batch(batch.filter(pa.array(keep)))
                    input_rows += size
                    kept_rows += sum(keep)
                    offset += size
            finally:
                writer.close()
            expected = parquet.metadata.num_rows - len(by_file.get(source.name, set()))
            if input_rows != parquet.metadata.num_rows or kept_rows != expected:
                raise RuntimeError(f"row-count mismatch for {source.name}")
            stats[source.name] = {"input": input_rows, "removed": input_rows - kept_rows, "kept": kept_rows}

        archive.mkdir(parents=True)
        for source in ordinary_paths(root, split):
            shutil.move(str(source), archive / source.name)
            (temp / source.name).replace(source)
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4] / "model/language_model/data/post_train/data/rendered/sft")
    parser.add_argument("--train-target", type=int, default=10_000)
    parser.add_argument("--test-target", type=int, default=8_000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    archive = args.root / "archive/ordinary_sft_trim_20260825"
    manifest_path = args.root / "ordinary_sft_trim_20260825.json"
    if not args.dry_run and (archive.exists() or manifest_path.exists()):
        raise SystemExit(f"refusing to reuse existing output: {archive if archive.exists() else manifest_path}")

    targets = {"train": args.train_target, "test": args.test_target}
    selected_by_split = {}
    summary = {"policy": "conservative_non_coding_topic_trim_v1", "targets": targets, "splits": {}}
    for split, target in targets.items():
        candidates, counts = collect(args.root, split)
        selected = select(candidates, target, split)
        selected_by_split[split] = selected
        summary["splits"][split] = {
            "files": [path.name for path in ordinary_paths(args.root, split)],
            "input_rows": sum(counts.values()),
            "coding_guard_rows": counts.pop("coding_guard", 0),
            "safe_noncoding_rows": len(candidates),
            "safe_noncoding_topics": dict(counts),
            "selected_rows": len(selected),
            "selected_topics": dict(Counter(row["topic"] for row in selected)),
        }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    for split in ("train", "test"):
        summary["splits"][split]["files_after"] = rewrite(root=args.root, split=split, selected=selected_by_split[split], archive=archive)
    summary["archive"] = str(archive)
    summary["removed_rows"] = {split: selected for split, selected in selected_by_split.items()}
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
