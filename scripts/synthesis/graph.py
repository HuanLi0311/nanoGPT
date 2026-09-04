"""Stage 1: controlled domain/material sampling with source provenance."""

from __future__ import annotations

import hashlib
import random
import subprocess
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .schema import fingerprint, stable_json, validate_plan, write_json, write_jsonl


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())


def _leaves(plan: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for domain in plan["domains"]:
        for subdomain in domain["subdomains"]:
            for concept in subdomain["concepts"]:
                output.append({
                    "domain": domain["name"], "subdomain": subdomain["name"],
                    "concept": concept["name"], "family": f"{domain['name']}.{subdomain['name']}",
                    "source": concept["source"], "task_template": concept["task"],
                })
    return output


def _quotas(total: int, weights: dict[str, float]) -> dict[str, int]:
    if total < 1 or not weights or sum(weights.values()) <= 0 or any(value < 0 for value in weights.values()):
        raise ValueError("sample count and distribution weights must be positive")
    scale = total / sum(weights.values())
    raw = {key: value * scale for key, value in weights.items()}
    result = {key: int(value) for key, value in raw.items()}
    for key in sorted(raw, key=lambda item: (raw[item] - result[item], item), reverse=True)[: total - sum(result.values())]:
        result[key] += 1
    return result


def _sample(plan: dict[str, Any], profile_name: str, seed: int, count: int | None, weights: dict[str, float] | None) -> list[dict[str, Any]]:
    leaves = _leaves(plan)
    profiles = plan.get("profiles", {})
    profile = profiles.get(profile_name, {}) if isinstance(profiles, dict) else {}
    if profiles and profile_name not in profiles:
        raise ValueError(f"unknown profile: {profile_name}")
    total = int(count or profile.get("count") or plan.get("sample_count") or len(leaves))
    distribution = weights or profile.get("distribution") or plan.get("target_distribution")
    families = sorted({leaf["family"] for leaf in leaves})
    distribution = {name: 1.0 for name in families} if distribution is None else {
        str(name): float(value) for name, value in distribution.items()
    }
    unknown = set(distribution) - set(families)
    if unknown:
        raise ValueError(f"distribution references unknown families: {sorted(unknown)}")
    groups = {name: [leaf for leaf in leaves if leaf["family"] == name] for name in distribution}
    rng, selected = random.Random(seed), []
    for family, quota in _quotas(total, distribution).items():
        pool = groups[family]
        if quota and not pool:
            raise ValueError(f"no concepts available for {family}")
        rng.shuffle(pool)
        selected.extend({**pool[index % len(pool)], "sample_index": index} for index in range(quota))
    rng.shuffle(selected)
    return selected


def _decode(data: bytes, content_type: str = "") -> str:
    charset = "utf-8"
    if "charset=" in content_type:
        charset = content_type.split("charset=", 1)[1].split(";", 1)[0].strip()
    text = data.decode(charset, errors="replace")
    if "html" in content_type.lower() or "<html" in text[:500].lower():
        parser = _Text()
        parser.feed(text)
        return "\n".join(parser.parts) + "\n"
    return text


def _local(source: dict[str, Any], base: Path, limit: int) -> tuple[str, dict[str, Any]]:
    path = Path(str(source.get("uri", "")))
    path = path if path.is_absolute() else (base / path).resolve()
    if source["kind"] == "repo" and path.is_dir():
        paths = [item for item in sorted(path.glob(source.get("glob", "**/*"))) if item.is_file() and ".git" not in item.parts]
        paths = paths[: int(source.get("max_files", 32))]
        text = "".join(f"\n--- {item.relative_to(path)} ---\n{item.read_text(encoding='utf-8', errors='replace')}" for item in paths)
        revision = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or None
        resolved = [str(item) for item in paths]
    else:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".pdf":
            result = subprocess.run(["pdftotext", str(path), "-"], capture_output=True, check=True)
            text = result.stdout.decode("utf-8", errors="replace")
        else:
            text = path.read_text(encoding="utf-8", errors="replace")
        revision, resolved = source.get("revision"), str(path)
    if len(text.encode()) > limit:
        raise ValueError(f"retrieved material exceeds max_bytes={limit}: {path}")
    return text, {"resolved_uri": resolved, "revision": revision}


def _retrieve(source: dict[str, Any], base: Path) -> tuple[str, dict[str, Any]]:
    kind = str(source.get("kind", ""))
    limit = int(source.get("max_bytes", 1_000_000))
    if kind == "inline":
        text, extra = str(source.get("content", "")), {"resolved_uri": "inline", "revision": source.get("revision")}
    elif kind in {"document", "repo"}:
        text, extra = _local(source, base, limit)
    elif kind == "web":
        uri = str(source.get("uri", ""))
        if urlparse(uri).scheme not in {"http", "https"}:
            raise ValueError(f"unsupported web URI: {uri}")
        request = Request(uri, headers={"User-Agent": "NanoAgent-Synthesis/1.0"})
        with urlopen(request, timeout=int(source.get("timeout", 30))) as response:
            data = response.read(limit + 1)
            if len(data) > limit:
                raise ValueError(f"retrieved material exceeds max_bytes={limit}: {uri}")
            text = _decode(data, response.headers.get("Content-Type", ""))
            extra = {"resolved_uri": response.geturl(), "revision": response.headers.get("ETag") or response.headers.get("Last-Modified")}
    else:
        raise ValueError(f"source.kind must be inline, document, repo, or web: {kind!r}")
    if not text.strip():
        raise ValueError("retrieved material is empty")
    return text, extra


def build_material_graph(plan_path: Path, output: Path, *, profile: str = "default", seed: int = 0,
                         count: int | None = None, weights: dict[str, float] | None = None) -> dict[str, Any]:
    from .schema import read_json

    plan = read_json(plan_path)
    validate_plan(plan)
    selected, rows, cache = _sample(plan, profile, seed, count, weights), [], {}
    material_dir = output / "materials"
    for index, leaf in enumerate(selected):
        source_key = stable_json(leaf["source"])
        if source_key not in cache:
            cache[source_key] = _retrieve(leaf["source"], plan_path.parent)
        content, extra = cache[source_key]
        digest = hashlib.sha256(content.encode()).hexdigest()
        material_id = f"mat-{fingerprint([profile, leaf['domain'], leaf['subdomain'], leaf['concept'], index, digest])[:16]}"
        path = material_dir / f"{material_id}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        rows.append({**leaf, "material_id": material_id, "profile": profile, "content_path": str(path.relative_to(output)),
                     "provenance": {"kind": leaf["source"]["kind"], "uri": leaf["source"].get("uri", "inline"),
                                    **extra, "sha256": digest, "bytes": len(content.encode()),
                                    "license": leaf["source"].get("license"), "retrieved_at": datetime.now(timezone.utc).isoformat()}})
    counts = {family: sum(row["family"] == family for row in rows) for family in sorted({row["family"] for row in rows})}
    nodes, edges = [], []
    for leaf in _leaves(plan):
        nodes.extend([leaf["domain"], leaf["subdomain"], leaf["concept"]])
        edges.extend([{"from": leaf["domain"], "to": leaf["subdomain"], "relation": "has_subdomain"},
                      {"from": leaf["subdomain"], "to": leaf["concept"], "relation": "has_concept"}])
    write_jsonl(output / "materials.jsonl", rows)
    graph = {"version": "domain-material-v1", "profile": profile, "seed": seed, "nodes": sorted(set(nodes)),
             "edges": edges, "target_count": len(rows), "realized_distribution": counts,
             "unique_sources": len(cache), "materials": str(output / "materials.jsonl")}
    write_json(output / "domain_graph.json", graph)
    return graph
