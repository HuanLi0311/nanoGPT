# Evidence extraction protocol

`scripts/extract_evidence.py` creates a location-aware JSON index from a local
PDF, a LaTeX/BibTeX file, or a directory containing those sources. It uses
only Python's standard library. For PDFs it prefers the installed
`pdftotext -layout` command and falls back to uncompressed PDF text strings;
the fallback cannot OCR scanned pages or decode compressed font streams.

## Commands

```bash
python3 scripts/extract_evidence.py paper.pdf --json
python3 scripts/extract_evidence.py paper.tex --query "learning rate" --query baseline
python3 scripts/extract_evidence.py source-tree/ --query WikiText --max-matches 0
python3 scripts/extract_evidence.py --self-check
```

JSON is emitted to stdout by default; `--json` is an explicit, harmless alias
for callers that require a JSON flag. `--query` is a repeatable,
case-insensitive substring search over extracted source text. Each match has
`source`, `line`, `offset`, `text`, and `page` for PDF input. `--max-matches 0`
removes the per-query limit.

## Output contract

The top-level object contains:

- `input`: resolved input path;
- `sources`: one record per input file, including `format`, parser used,
  character/line counts, and normalized extracted `text`;
- `evidence`: arrays keyed by `title`, `sections`, `equations`, `assumptions`,
  `datasets`, `baselines`, `hyperparameters`, `metrics`, `tables`,
  `numerical_results`, `conclusions`, and `citations`;
- `queries`: the query string and matching location records;
- `warnings`: parser failures, empty extraction, or OCR limitations.

Evidence is heuristic and must be audited against the cited source. A
BibTeX title or a `\\cite` occurrence proves only that a local citation exists;
it does not prove that the referenced paper was read. Numerical claims should
retain the reported `source` plus `line`/`page` and `offset` in downstream
Markdown reports.
