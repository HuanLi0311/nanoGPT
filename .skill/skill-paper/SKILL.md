---
name: skill-paper
description: Parse local PDF or LaTeX papers, search an arXiv-style source library, and extract evidence-backed claims, configurations, metrics, and literature summaries. Use for paper replication planning or report citations.
---

# Paper and evidence operations

Prefer source LaTeX when it is available because tables, captions, and configuration values are easier to audit; use `pdftotext` for a PDF-only input. Run `scripts/extract_evidence.py` and keep its JSON as the machine-readable source for the Markdown analysis.

## Workflow

1. Extract title, section headings, equations/assumptions, datasets, baselines, hyperparameters, metrics, numerical tables, and conclusion snippets.
2. Attach every reported number to a source file/line or PDF page/text offset. Distinguish stated results from inferences and unresolved inconsistencies.
3. Search local `.tex`, `.bib`, and metadata files with the script's `--query`; do not turn a bibliography hit into evidence that the cited paper was read.
4. Write a replication matrix: claim, operational DLLM test, metric, baseline, pass/fail rule, and known mismatch.
5. Use the extracted evidence to write `report/paper_analysis.md` before running models.

The extractor uses only the standard library plus installed command-line tools and has a small self-check. See [references/protocol.md](references/protocol.md).
