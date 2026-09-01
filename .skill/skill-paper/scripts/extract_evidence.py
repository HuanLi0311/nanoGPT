#!/usr/bin/env python3
"""Extract lightweight, location-aware evidence from local papers.

The script deliberately uses heuristics: it is a reproducible index for a
human audit, not a claim that every sentence or table was understood.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


EVIDENCE_KINDS = (
    "title",
    "sections",
    "equations",
    "assumptions",
    "datasets",
    "baselines",
    "hyperparameters",
    "metrics",
    "tables",
    "numerical_results",
    "conclusions",
    "citations",
)
LATEX_SUFFIXES = {".tex", ".ltx", ".latex", ".sty", ".cls"}
TEXT_SUFFIXES = LATEX_SUFFIXES | {
    ".bib",
    ".txt",
    ".md",
    ".json",
    ".yaml",
    ".yml",
}
NUMBER = re.compile(r"(?<!\w)\d+(?:\.\d+)?%?(?!\w)")
KNOWN_HEADINGS = {
    "abstract",
    "introduction",
    "related work",
    "background",
    "method",
    "methods",
    "approach",
    "experiments",
    "experimental setup",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "appendix",
}
PATTERNS = {
    "assumptions": re.compile(
        r"\b(?:assumption|assume|assuming|hypothesis|hypotheses|we suppose|under the)\b",
        re.I,
    ),
    "datasets": re.compile(
        r"\b(?:dataset|datasets|benchmark|benchmarks|corpus|corpora|train(?:ing)? set|test set|evaluation set)\b",
        re.I,
    ),
    "baselines": re.compile(
        r"\b(?:baseline|baselines|ablations?|state[- ]of[- ]the[- ]art|compared with|compared to|prior work)\b",
        re.I,
    ),
    "hyperparameters": re.compile(
        r"\b(?:learning rate|batch size|micro[- ]batch|epochs?|steps?|temperature|dropout|weight decay|warmup|sequence length|context length|seed|optimizer|hidden size|beam size|top[- ]p)\b",
        re.I,
    ),
    "metrics": re.compile(
        r"\b(?:accuracy|exact match|f1|precision|recall|bleu|rouge|perplexity|loss|pass@\d+|success rate|reward|win rate|auroc|auc)\b",
        re.I,
    ),
    "conclusions": re.compile(
        r"\b(?:we (?:show|find|conclude|demonstrate|observe)|our results|these results|we achieve|we improve|suggests?|indicates?)\b",
        re.I,
    ),
}


def _snippet(value: str, limit: int = 1200) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _kind_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix == ".bib":
        return "bib"
    if suffix in LATEX_SUFFIXES:
        return "latex"
    return "text"


def _discover(path_value: str | Path) -> list[Path]:
    path = Path(path_value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"input does not exist: {path}")
    if path.is_file():
        return [path.resolve()]
    if not path.is_dir():
        raise ValueError(f"input is neither a file nor a directory: {path}")
    paths = sorted(
        p.resolve()
        for p in path.rglob("*")
        if p.is_file() and (p.suffix.lower() == ".pdf" or p.suffix.lower() in TEXT_SUFFIXES)
    )
    if not paths:
        raise ValueError(f"no PDF/LaTeX/text sources found under: {path}")
    return paths


def _normalise_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _pdf_literal_text(data: bytes) -> str:
    """Read uncompressed PDF text strings when pdftotext is unavailable."""
    raw = data.decode("latin-1", errors="ignore")
    values: list[str] = []
    i = 0
    while i < len(raw):
        if raw[i] == "(":
            i += 1
            depth = 1
            chars: list[str] = []
            while i < len(raw) and depth:
                char = raw[i]
                if char == "\\" and i + 1 < len(raw):
                    i += 1
                    escaped = raw[i]
                    escapes = {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f"}
                    if escaped in escapes:
                        chars.append(escapes[escaped])
                    elif escaped.isdigit() and escaped in "01234567":
                        digits = escaped
                        for _ in range(2):
                            if i + 1 < len(raw) and raw[i + 1] in "01234567":
                                i += 1
                                digits += raw[i]
                            else:
                                break
                        chars.append(chr(int(digits, 8)))
                    else:
                        chars.append(escaped)
                elif char == "(":
                    depth += 1
                    chars.append(char)
                elif char == ")":
                    depth -= 1
                    if depth:
                        chars.append(char)
                else:
                    chars.append(char)
                i += 1
            value = "".join(chars).strip()
            if value:
                values.append(value)
            continue
        if raw[i] == "<" and i + 1 < len(raw) and raw[i + 1] != "<":
            end = raw.find(">", i + 1)
            token = raw[i + 1 : end] if end >= 0 else ""
            if token and re.fullmatch(r"[0-9a-fA-F\s]+", token):
                try:
                    value = bytes.fromhex("".join(token.split())).decode("utf-8", "replace").strip()
                except ValueError:
                    value = ""
                if value:
                    values.append(value)
                i = end + 1 if end >= 0 else i + 1
                continue
        i += 1
    return "\n".join(values)


def _read_pdf(path: Path) -> tuple[str, str, list[str]]:
    warnings: list[str] = []
    tool = shutil.which("pdftotext")
    if tool:
        try:
            result = subprocess.run(
                [tool, "-layout", str(path), "-"],
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            if result.returncode == 0 and result.stdout.strip():
                return _normalise_text(result.stdout), "pdftotext", warnings
            detail = _snippet(result.stderr, 240) if result.stderr else f"exit {result.returncode}"
            warnings.append(f"pdftotext failed for {path.name}: {detail}")
        except (OSError, subprocess.TimeoutExpired) as exc:
            warnings.append(f"pdftotext unavailable for {path.name}: {exc}")
    else:
        warnings.append("pdftotext not found; used the standard-library PDF fallback")
    text = _normalise_text(_pdf_literal_text(path.read_bytes()))
    if not text.strip():
        warnings.append(f"no text extracted from {path.name}; scanned PDFs need OCR")
    return text, "stdlib-pdf-fallback", warnings


def _read_source(path: Path) -> tuple[str, str, list[str]]:
    if _kind_for(path) == "pdf":
        return _read_pdf(path)
    return _normalise_text(path.read_text(encoding="utf-8", errors="replace")), _kind_for(path), []


def _line_starts(text: str) -> list[int]:
    return [0] + [match.end() for match in re.finditer("\n", text)]


def _locate(doc: dict, offset: int) -> tuple[int, int | None]:
    line = bisect.bisect_right(doc["line_starts"], offset)
    page = doc["text"].count("\f", 0, offset) + 1 if doc["format"] == "pdf" else None
    return line, page


def _source_lines(doc: dict):
    text = doc["text"]
    if doc["format"] != "pdf":
        position = 0
        for line_number, line in enumerate(text.split("\n"), 1):
            yield line, line_number, None, position
            position += len(line) + 1
        return
    position = 0
    for page_number, page in enumerate(text.split("\f"), 1):
        page_position = 0
        for line_number, line in enumerate(page.split("\n"), 1):
            yield line, line_number, page_number, position + page_position
            page_position += len(line) + 1
        position += len(page) + 1


def _record(doc: dict, line: str, line_number: int, page: int | None, offset: int, value: str | None = None) -> dict:
    text = _snippet(value if value is not None else line)
    record = {
        "source": doc["path"],
        "line": line_number,
        "offset": offset,
        "text": text,
    }
    if page is not None:
        record["page"] = page
    context = _snippet(line)
    if value is not None and context and context != text:
        record["context"] = context
    return record


def _add(bucket: dict, seen: dict, category: str, doc: dict, line: str, line_number: int, page: int | None, offset: int, value: str | None = None) -> None:
    if not value and not line.strip():
        return
    record = _record(doc, line, line_number, page, offset, value)
    if not record["text"]:
        return
    key = (record["source"], record["offset"], record["text"])
    if key not in seen[category]:
        seen[category].add(key)
        bucket[category].append(record)


def _clean_latex(value: str) -> str:
    value = re.sub(r"\\(?:textbf|textit|emph|underline|mbox)\s*\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\[A-Za-z@]+\*?(?:\s*\[[^]]*\])?", "", value)
    return _snippet(value.replace("~", " ").replace("{", "").replace("}", ""))


def _braced_commands(text: str, names: tuple[str, ...]):
    pattern = re.compile(
        r"\\(" + "|".join(map(re.escape, names)) + r")\*?(?:\s*\[[^]]*\])?\s*\{"
    )
    for match in pattern.finditer(text):
        start = match.end() - 1
        depth = 0
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    yield match.group(1), text[start + 1 : index], match.start()
                    break


def _bib_titles(text: str):
    pattern = re.compile(r"(?im)^\s*title\s*=\s*([{\"])")
    for match in pattern.finditer(text):
        opener = match.group(1)
        start = match.end() - 1
        if opener == "{":
            depth = 0
            for index in range(start, len(text)):
                if text[index] == "{":
                    depth += 1
                elif text[index] == "}":
                    depth -= 1
                    if depth == 0:
                        yield text[start + 1 : index], match.start()
                        break
        else:
            end = start + 1
            while end < len(text) and (text[end] != '"' or text[end - 1] == "\\"):
                end += 1
            if end < len(text):
                yield text[start + 1 : end], match.start()


def _pdf_heading(value: str) -> str | None:
    value = re.sub(r"^\s*\d+(?:\.\d+)*\.?\s+", "", value).strip(" :")
    if value.casefold() in KNOWN_HEADINGS:
        return value
    return None


def _extract_document(doc: dict) -> dict[str, list[dict]]:
    evidence = {kind: [] for kind in EVIDENCE_KINDS}
    seen = {kind: set() for kind in EVIDENCE_KINDS}
    lines = list(_source_lines(doc))

    if doc["format"] in {"latex", "text"}:
        for command, value, offset in _braced_commands(
            doc["text"], ("title", "section", "subsection", "subsubsection", "paragraph", "subparagraph")
        ):
            line_number, page = _locate(doc, offset)
            line = doc["text"].splitlines()[line_number - 1] if doc["text"].splitlines() else ""
            category = "title" if command == "title" else "sections"
            _add(evidence, seen, category, doc, line, line_number, page, offset, _clean_latex(value))
    if doc["format"] == "bib":
        for value, offset in _bib_titles(doc["text"]):
            line_number, page = _locate(doc, offset)
            line = doc["text"].splitlines()[line_number - 1] if doc["text"].splitlines() else ""
            _add(evidence, seen, "title", doc, line, line_number, page, offset, _clean_latex(value))

    # ponytail: line heuristics keep the extractor dependency-free; use a real
    # document parser/OCR only when exact layout or scanned PDFs is required.
    conclusion_mode = False
    conclusion_lines = 0
    for line, line_number, page, offset in lines:
        stripped = line.strip()
        if not stripped or (doc["format"] in {"latex", "bib"} and stripped.startswith("%")):
            continue
        lower = stripped.casefold()

        heading = _pdf_heading(stripped) if doc["format"] == "pdf" else None
        if heading:
            _add(evidence, seen, "sections", doc, line, line_number, page, offset, heading)
            if "conclusion" in heading.casefold():
                conclusion_mode = True
                conclusion_lines = 0
            elif conclusion_mode:
                conclusion_mode = False
        if doc["format"] in {"latex", "text"} and re.search(
            r"\\(?:sub)*section\*?(?:\s*\[[^]]*\])?\s*\{[^}]*conclusion", stripped, re.I
        ):
            conclusion_mode = True
            conclusion_lines = 0
        elif doc["format"] in {"latex", "text"} and re.search(r"\\(?:sub)*section", stripped):
            conclusion_mode = False

        if doc["format"] == "pdf" and page == 1 and not evidence["title"]:
            if not re.match(r"^(?:arXiv|https?://|doi:|\[?\d{4}\.\d{4,})", stripped, re.I):
                _add(evidence, seen, "title", doc, line, line_number, page, offset)
        if doc["format"] != "bib":
            if re.search(r"\\begin\{(?:equation\*?|align\*?|gather\*?|multline\*?|math|displaymath)\}|\\\[|\$\$", stripped):
                _add(evidence, seen, "equations", doc, line, line_number, page, offset)
            if "&" in stripped and (r"\\" in stripped or "\\toprule" in stripped or "\\midrule" in stripped):
                _add(evidence, seen, "tables", doc, line, line_number, page, offset)
            if "\\cite" in stripped or "\\bibitem" in stripped:
                _add(evidence, seen, "citations", doc, line, line_number, page, offset)
        if doc["format"] == "pdf" and len(NUMBER.findall(stripped)) >= 2 and (
            "|" in stripped or "±" in stripped or re.search(r"\s{2,}", stripped)
        ):
            _add(evidence, seen, "tables", doc, line, line_number, page, offset)
        if doc["format"] == "pdf" and "=" in stripped and re.search(r"\s=\s|argmax|argmin|∑|\^", stripped):
            _add(evidence, seen, "equations", doc, line, line_number, page, offset)

        for category, pattern in PATTERNS.items():
            if pattern.search(stripped):
                _add(evidence, seen, category, doc, line, line_number, page, offset)
        if re.search(r"\b(?:result|improv|outperform|score|performance|accuracy)\w*\b", lower) and NUMBER.search(stripped):
            _add(evidence, seen, "numerical_results", doc, line, line_number, page, offset)
        if conclusion_mode and conclusion_lines < 12 and not heading:
            _add(evidence, seen, "conclusions", doc, line, line_number, page, offset)
            conclusion_lines += 1

    if doc["format"] not in {"pdf", "bib"} and not evidence["title"]:
        for line, line_number, page, offset in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("%") and not stripped.startswith("\\"):
                _add(evidence, seen, "title", doc, line, line_number, page, offset)
                break
    return evidence


def _query(docs: list[dict], query: str, limit: int) -> list[dict]:
    needle = query.casefold()
    matches: list[dict] = []
    for doc in docs:
        for line, line_number, page, offset in _source_lines(doc):
            if needle in line.casefold():
                matches.append(_record(doc, line, line_number, page, offset))
                if limit and len(matches) >= limit:
                    return matches
    return matches


def extract_evidence(path_value: str | Path, queries: list[str] | None = None, max_matches: int = 100) -> dict:
    if max_matches < 0:
        raise ValueError("--max-matches must be >= 0 (0 means unlimited)")
    paths = _discover(path_value)
    docs: list[dict] = []
    warnings: list[str] = []
    evidence = {kind: [] for kind in EVIDENCE_KINDS}
    for path in paths:
        text, parser, source_warnings = _read_source(path)
        doc = {
            "path": str(path),
            "format": _kind_for(path),
            "parser": parser,
            "text": text,
            "line_starts": _line_starts(text),
        }
        docs.append(doc)
        warnings.extend(source_warnings)
        for category, records in _extract_document(doc).items():
            evidence[category].extend(records)
    query_results = [
        {"query": query, "matches": _query(docs, query, max_matches)}
        for query in (queries or [])
    ]
    sources = [
        {
            "path": doc["path"],
            "format": doc["format"],
            "parser": doc["parser"],
            "characters": len(doc["text"]),
            "lines": len(doc["text"].splitlines()),
            "text": doc["text"],
        }
        for doc in docs
    ]
    return {
        "input": str(Path(path_value).expanduser().resolve()),
        "sources": sources,
        "evidence": evidence,
        "queries": query_results,
        "warnings": warnings,
    }


def self_check() -> list[str]:
    with tempfile.TemporaryDirectory(prefix="skill-paper-") as directory:
        root = Path(directory)
        tex = root / "sample.tex"
        tex.write_text(
            r"""\title{A Small Paper}
\section{Experiments}
We use the WikiText dataset and compare with the baseline.
The learning rate is 1e-4 and accuracy reaches 91\%.
\section{Conclusion}
Our results show an improvement.
""",
            encoding="utf-8",
        )
        result = extract_evidence(tex, ["wikitext"])
        assert result["sources"][0]["format"] == "latex"
        assert result["evidence"]["title"][0]["text"] == "A Small Paper"
        assert any(item["text"] == "Experiments" for item in result["evidence"]["sections"])
        assert result["queries"][0]["matches"][0]["line"] == 3
        assert result["evidence"]["datasets"]
        assert result["evidence"]["hyperparameters"]
        assert result["evidence"]["metrics"]
        assert result["evidence"]["conclusions"]

        pdf = root / "sample.pdf"
        pdf.write_bytes(b"%PDF-1.4\nBT\n(Plain PDF title) Tj\n(Accuracy 88 percent) Tj\nET\n%%EOF\n")
        pdf_result = extract_evidence(pdf, ["accuracy"])
        assert pdf_result["sources"][0]["format"] == "pdf"
        assert pdf_result["evidence"]["title"]
        assert pdf_result["queries"][0]["matches"]
    return ["LaTeX title/sections/heuristics", "case-insensitive query with location", "PDF fallback"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="local PDF, LaTeX file, or source directory")
    parser.add_argument("--query", action="append", default=[], metavar="TEXT", help="case-insensitive substring search; repeatable")
    parser.add_argument("--max-matches", type=int, default=100, help="matches per query; 0 means unlimited (default: 100)")
    parser.add_argument("--json", action="store_true", help="accepted for explicit JSON-output invocations; JSON is always emitted")
    parser.add_argument("--output", type=Path, help="also save the JSON envelope to this path")
    parser.add_argument("--self-check", action="store_true", help="run the dependency-free executable check")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.self_check:
        print(json.dumps({"self_check": "ok", "checks": self_check()}, ensure_ascii=False, indent=2))
        return 0
    if not args.path:
        parser.error("path is required unless --self-check is used")
    try:
        result = extract_evidence(args.path, args.query, args.max_matches)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
