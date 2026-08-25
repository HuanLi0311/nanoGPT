"""Keep a domain whitelist in ordinary SFT shards and archive the originals."""

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

from model.language_model.scripts.synthesis.trim_ordinary_sft import is_coding, ordinary_paths, topic


KEEP = {
    "math_statistics",
    "natural_science",
    "language_summary_text",
    "software_it_digital",
    "history_religion_philosophy",
    "general_unclear",
}
EXPECTED = {"train": {"input": 122_305, "kept": 44_703}, "test": {"input": 15_110, "kept": 5_854}}
PATTERNS = {
    "software_it_digital": r"\b(?:algorithm\w*|data structures?|comput\w*|technolog\w*|software|hardware|internet|websites?|webpages?|mobile|smartphones?|digital|blockchain|cryptocurrenc\w*|virtual reality|augmented reality|artificial intelligence|robots?|robotics|automat\w*|networks?|routers?|vpn|cyber\w*|databases?|firebase|search engines?)\b",
    "math_statistics": r"\b(?:math\w*|algebra\w*|calculus|geometr\w*|arithmetic|equations?|probabilit\w*|statistic\w*|theorems?|integers?|fractions?|matri(?:x|ces)|derivatives?|integrals?|combinator\w*|quantitative)\b",
    "natural_science": r"\b(?:physics|chem\w*|biolog\w*|genetic\w*|genom\w*|dna|rna|cells?|molecul\w*|atoms?|astronom\w*|planets?|galax\w*|solar system|spacecraft|meteors?|comets?|evolution\w*|species|organisms?|insects?|arachnids?|animals?|neuroscien\w*|brain regions?)\b",
    "environment_energy_agriculture": r"\b(?:environment\w*|climate\w*|carbon|co2|emissions?|pollution|pollutants?|renewable\w*|energy|sustainab\w*|recycl\w*|waste|plastics?|biodivers\w*|conservation|agricultur\w*|farming|farmers?|soil|erosion|ecosystems?|weather|meteorolog\w*|forests?|oceans?|compost\w*|ecolog\w*)\b",
    "health_psychology": r"\b(?:healthcare|wellness|well-being|brains?|cognit\w*|psycholog\w*|emotions?|anatom\w*|physiolog\w*|illness\w*|diseases|doctors|patients|hospitals|symptoms|diagnosis|treatments?|clinical|pharma\w*|nursing|nutritional|vitamins?|calories|fitness)\b",
    "history_religion_philosophy": r"\b(?:histor\w*|ancient|medieval|renaissance|revolution\w*|empires?|civilization\w*|warfare|world war|religio\w*|church\w*|bible|biblical|jesus|christian\w*|islam\w*|muslim\w*|hindu\w*|buddh\w*|god|faith|theolog\w*|philosoph\w*|ethics?|mytholog\w*)\b",
    "arts_literature_media": r"\b(?:artworks?|artists?|music\w*|songs?|albums?|films?|movies?|cinema|theat(?:er|re)|books?|authors?|literat\w*|literary|novels?|poems?|lyrics?|shakespeare|paintings?|photograph\w*|comics?|television|tv show\w*|video games?|board games?)\b",
    "food_cooking": r"\b(?:foods?|cuisine|culinary|ingredients?|dishes|meals?|beverages?|coffee|tea|cookies?|cheese|pasta|soup|burrito|dessert\w*|flavo(?:r|ur)\w*|baking|brew\w*|kitchen|vegetarian|vegan)\b",
    "travel_places_transport": r"\b(?:travels?|traveling|travelling|trips?|tours?|tourism|tourists?|destinations?|vacations?|hotels?|museums?|attractions?|sightseeing|transportation|hiking|cycling|trails?|local cuisine|places to (?:visit|stay)|things to do)\b",
    "business_workplace_economics": r"\b(?:businesses|companies|customers|products|sales|markets?|startups?|advertis\w*|retailers?|entrepreneur\w*|brands?|pricing|accounting|investors?|investments?|employees?|employers?|workplace|productivity|management|industr(?:y|ies)|consumers?|operations|transactions?|econom\w*|trade|budgets?|team meetings?|project management|organizations?)\b",
    "personal_family_lifestyle": r"\b(?:famil(?:y|ies)|friends?|partners?|birthday\w*|love|romantic|personal|goals?|hobbies|breakups?|marriage|wedding\w*|children|parents?|households?|daily life|self-improvement|motivation\w*)\b",
    "law_politics_military": r"\b(?:laws?|legal\w*|courts?|judges?|governments?|presidents?|senators?|congress\w*|politic\w*|elections?|campaigns?|regulations?|human rights|white house|military|army|marine corps|soldiers?|public polic\w*)\b",
    "sports_games": r"\b(?:sports?|football|soccer|basketball|baseball|hockey|tennis|golf|athletes?|coaches?|tournaments?|championship\w*|fifa|olympic\w*|track and field)\b",
    "education_careers_training": r"\b(?:schools?|students?|teachers?|universit\w*|colleges?|classrooms?|curricul\w*|academic\w*|professors?|training programs?|courses?|career\w*|jobs?|employment|professional development|workshops?)\b",
    "language_summary_text": r"\b(?:translat\w*|paraphras\w*|summari[sz]\w*|grammar|proofread\w*|rewrite|rephrase|passage|piece of text|answer according to|main points|meaning of the phrase)\b",
    "practical_howto_communication": r"\b(?:step-by-step|instructions?|manual|guide|agenda|to-do list|checklist|letter|social media post|infographic|timeline|tutorial|how (?:can|do|to)|create|design|develop|draft|provide a list)\b",
}
REGEXES = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in PATTERNS.items()}


def classify(prompt: str) -> str:
    if is_coding(prompt):
        return "coding"
    coarse = topic(prompt)
    if coarse != "other_noncoding":
        return f"coarse:{coarse}"
    scores = {name: len(regex.findall(prompt)) for name, regex in REGEXES.items()}
    best = max(scores.values(), default=0)
    return "general_unclear" if best == 0 else next(name for name in PATTERNS if scores[name] == best)


def process(path: Path, target: Path | None = None) -> dict:
    parquet = pq.ParquetFile(path)
    writer = pq.ParquetWriter(target, parquet.schema_arrow, compression="zstd") if target else None
    labels, kept = Counter(), Counter()
    try:
        for batch in parquet.iter_batches(batch_size=2048):
            row_labels = [classify(value or "") for value in batch.column("prompt").to_pylist()]
            labels.update(row_labels)
            mask = [label == "coding" or label in KEEP for label in row_labels]
            kept.update(label for label, retain in zip(row_labels, mask) if retain)
            if writer and any(mask):
                writer.write_batch(batch.filter(pa.array(mask)))
    finally:
        if writer:
            writer.close()
    return {"input": sum(labels.values()), "kept": sum(kept.values()), "removed": sum(labels.values()) - sum(kept.values()), "kept_by_label": dict(kept), "all_labels": dict(labels)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4] / "model/language_model/data/post_train/data/rendered/sft")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    paths = {split: ordinary_paths(args.root, split) for split in ("train", "test")}
    report = {"policy": "ordinary_sft_domain_whitelist_v1", "keep": ["coding", *sorted(KEEP)], "splits": {}}
    for split, split_paths in paths.items():
        files = {path.name: process(path) for path in split_paths}
        totals = {key: sum(stats[key] for stats in files.values()) for key in ("input", "kept", "removed")}
        if totals["input"] != EXPECTED[split]["input"] or totals["kept"] != EXPECTED[split]["kept"]:
            raise RuntimeError(f"{split} safety count mismatch: {totals} != {EXPECTED[split]}")
        report["splits"][split] = {"totals": totals, "files": files}
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return

    archive = args.root / "archive/ordinary_sft_domain_whitelist_20260825"
    temporary = args.root / ".ordinary_sft_domain_whitelist_tmp"
    report_path = args.root / "ordinary_sft_domain_whitelist_20260825.json"
    if archive.exists() or temporary.exists() or report_path.exists():
        raise RuntimeError("refusing to reuse whitelist output paths")
    temporary.mkdir()
    try:
        for split, split_paths in paths.items():
            for source in split_paths:
                stats = process(source, temporary / source.name)
                if stats != report["splits"][split]["files"][source.name]:
                    raise RuntimeError(f"non-deterministic classification: {source.name}")
        archive.mkdir(parents=True)
        for split_paths in paths.values():
            for source in split_paths:
                shutil.move(source, archive / source.name)
                os.replace(temporary / source.name, source)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    report["archive"] = str(archive)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "splits": {k: v["totals"] for k, v in report["splits"].items()}, "archive": str(archive), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
