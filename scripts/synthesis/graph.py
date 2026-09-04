"""Stage 1: controlled domain/material sampling with source provenance."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlencode, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

if __package__:
    from .schema import fingerprint, read_json, render, stable_json, validate_plan, write_json, write_jsonl
else:
    from schema import fingerprint, read_json, render, stable_json, validate_plan, write_json, write_jsonl


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())


class _SearchLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self.href: str | None = None
        self.title: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set((values.get("class") or "").split())
        if tag == "a" and "result__a" in classes and values.get("href"):
            self.href, self.title = str(values["href"]), []

    def handle_data(self, data: str) -> None:
        if self.href and data.strip():
            self.title.append(data.strip())

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.href:
            self.results.append({"url": self.href, "title": " ".join(self.title)})
            self.href, self.title = None, []


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
    total = int(count if count is not None else profile.get("count") or plan.get("sample_count") or len(leaves))
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
        # ponytail: tag stripping is enough for material discovery; add a
        # readability extractor only if measured retrieval quality needs it.
        parser = _Text()
        parser.feed(text)
        return "\n".join(parser.parts) + "\n"
    return text


def _result_url(value: str, endpoint: str) -> str | None:
    url = urljoin(endpoint, value)
    parsed = urlsplit(url)
    if parsed.hostname and parsed.hostname.endswith("duckduckgo.com"):
        target = parse_qs(parsed.query).get("uddg", [None])[0]
        if target:
            url, parsed = target, urlsplit(target)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", parsed.query, ""))


def _domain_sets(source: dict[str, Any]) -> tuple[set[str], set[str]]:
    allowed, blocked = source.get("allowed_domains", []), source.get("blocked_domains", [])
    if not isinstance(allowed, list) or not isinstance(blocked, list):
        raise ValueError("web_search domain filters must be lists")
    return ({str(item).lower().removeprefix("www.") for item in allowed if str(item).strip()},
            {str(item).lower().removeprefix("www.") for item in blocked if str(item).strip()})


def _wikipedia(source: dict[str, Any], query: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    language = str(source.get("wikipedia_language", "en")).lower()
    if not language.replace("-", "").isalnum():
        raise ValueError(f"invalid wikipedia_language: {language!r}")
    endpoint = f"https://{language}.wikipedia.org/w/rest.php/v1/search/page"
    limit = int(source.get("max_results", 10))
    byte_limit = int(source.get("max_search_bytes", 2_000_000))
    if limit < 1 or byte_limit < 1:
        raise ValueError("web_search limits must be positive")
    host = f"{language}.wikipedia.org"
    allowed, blocked = _domain_sets(source)
    if ((allowed and not any(host == name or host.endswith(f".{name}") for name in allowed))
            or any(host == name or host.endswith(f".{name}") for name in blocked)):
        raise ValueError("Wikipedia fallback is excluded by the configured domain filters")
    request_url = endpoint + "?" + urlencode({"q": query, "limit": limit})
    request = Request(request_url, headers={"User-Agent": str(source.get("user_agent", "NanoAgent-Synthesis/1.0")),
                                            "Accept": "application/json"})
    with urlopen(request, timeout=int(source.get("timeout", 30))) as response:
        data, resolved = response.read(byte_limit + 1), response.geturl()
    if len(data) > byte_limit:
        raise ValueError(f"search response exceeds max_search_bytes={byte_limit}")
    payload = json.loads(data)
    pages = payload.get("pages", []) if isinstance(payload, dict) else []
    results = [{"url": f"https://{language}.wikipedia.org/wiki/{quote(str(item['key']))}",
                "title": str(item.get("title", item["key"])), "rank": index}
               for index, item in enumerate(pages) if isinstance(item, dict) and item.get("key")]
    if not results:
        raise ValueError(f"Wikipedia search returned no usable URLs for {query!r}")
    return results, {"provider": "wikipedia", "query": query, "request_url": request_url,
                     "resolved_search_url": resolved}


def _discover(source: dict[str, Any], query: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    provider = str(source.get("provider", "auto"))
    if provider == "wikipedia":
        return _wikipedia(source, query)
    if provider == "auto" and not source.get("search_endpoint"):
        try:
            return _discover({**source, "provider": "duckduckgo_html",
                              "search_endpoint": "https://html.duckduckgo.com/html/"}, query)
        except Exception as error:
            results, info = _wikipedia(source, query)
            info["fallback_from"] = "duckduckgo_html"
            info["fallback_reason"] = f"{type(error).__name__}: {error}"
            return results, info
    if provider not in {"auto", "duckduckgo_html"} and not source.get("search_endpoint"):
        raise ValueError(f"search provider {provider!r} requires search_endpoint")
    endpoint = str(source.get("search_endpoint") or "https://html.duckduckgo.com/html/")
    if urlsplit(endpoint).scheme not in {"http", "https"}:
        raise ValueError(f"unsupported search endpoint: {endpoint}")
    params = source.get("search_params", {})
    if not isinstance(params, dict):
        raise ValueError("web_search.search_params must be an object")
    separator = "&" if "?" in endpoint else "?"
    request_url = endpoint + separator + urlencode({**params, "q": query})
    headers = {"User-Agent": str(source.get("user_agent", "NanoAgent-Synthesis/1.0")),
               "Accept": "application/json,text/html;q=0.9"}
    if key_env := source.get("api_key_env"):
        key = os.environ.get(str(key_env))
        if not key:
            raise ValueError(f"missing search API key environment variable: {key_env}")
        headers[str(source.get("api_key_header", "X-Subscription-Token"))] = key
    limit = int(source.get("max_search_bytes", 2_000_000))
    with urlopen(Request(request_url, headers=headers), timeout=int(source.get("timeout", 30))) as response:
        data = response.read(limit + 1)
        if len(data) > limit:
            raise ValueError(f"search response exceeds max_search_bytes={limit}")
        content_type = response.headers.get("Content-Type", "")
        endpoint = response.geturl()
    if "json" in content_type.lower() or data.lstrip().startswith((b"{", b"[")):
        payload = json.loads(data)
        if isinstance(payload, dict):
            web = payload.get("web", {})
            items = payload.get("results", web.get("results", []) if isinstance(web, dict) else [])
        else:
            items = payload
        raw = [{"url": item, "title": ""} if isinstance(item, str) else item for item in items]
    else:
        parser = _SearchLinks()
        parser.feed(data.decode("utf-8", errors="replace"))
        raw = parser.results
    allowed, blocked = _domain_sets(source)
    max_results = int(source.get("max_results", 10))
    if max_results < 1:
        raise ValueError("web_search.max_results must be positive")
    if not all(isinstance(item, dict) for item in raw):
        raise ValueError("search results must be URL strings or objects")
    results, seen = [], set()
    for item in raw:
        url = _result_url(str(item.get("url") or item.get("link") or ""), endpoint)
        host = (urlsplit(url).hostname or "").lower().removeprefix("www.") if url else ""
        if (not url or url in seen or (allowed and not any(host == name or host.endswith(f".{name}") for name in allowed))
                or any(host == name or host.endswith(f".{name}") for name in blocked)):
            continue
        seen.add(url)
        results.append({"url": url, "title": str(item.get("title", "")), "rank": len(results)})
        if len(results) >= max_results:
            break
    if not results:
        raise ValueError(f"web search returned no usable URLs for {query!r}")
    provider = str(source.get("provider") or ("duckduckgo_html" if "duckduckgo.com" in endpoint else "custom"))
    return results, {"provider": provider, "query": query, "request_url": request_url,
                     "resolved_search_url": endpoint}


def _search_query(source: dict[str, Any], leaf: dict[str, Any]) -> str:
    templates = source.get("queries")
    if templates is not None and (not isinstance(templates, list) or not templates):
        raise ValueError("web_search.queries must be a non-empty list")
    template = (templates[leaf["sample_index"] % len(templates)] if templates else
                source.get("query", "{{concept}} {{subdomain}} {{domain}}"))
    query = str(render(template, leaf)).strip()
    if not query:
        raise ValueError("web_search query is empty")
    return query


def _local(source: dict[str, Any], base: Path, limit: int) -> tuple[str, dict[str, Any]]:
    path = Path(str(source.get("uri", "")))
    path = path.resolve() if path.is_absolute() else (base / path).resolve()
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
        revision = source.get("revision")
        if source["kind"] == "repo" and not revision:
            revision = subprocess.run(["git", "-C", str(path.parent), "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout.strip() or None
        resolved = str(path)
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
        if urlsplit(uri).scheme not in {"http", "https"}:
            raise ValueError(f"unsupported web URI: {uri}")
        request = Request(uri, headers={"User-Agent": str(source.get("user_agent", "NanoAgent-Synthesis/1.0"))})
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
    plan = read_json(plan_path)
    validate_plan(plan)
    selected, rows, cache = _sample(plan, profile, seed, count, weights), [], {}
    search_cache: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    claimed: dict[str, set[str]] = {}
    discoveries = []
    material_dir = output / "materials"
    for index, leaf in enumerate(selected):
        source = leaf["source"]
        extra_search: dict[str, Any] = {}
        if source.get("kind") == "web_search":
            query = _search_query(source, leaf)
            discovery_key = stable_json([source, query])
            if discovery_key not in search_cache:
                candidates, search_info = _discover(source, query)
                search_cache[discovery_key] = candidates, search_info
                discoveries.append({"discovery_id": f"search-{fingerprint(discovery_key)[:16]}",
                                    "domain": leaf["domain"], "subdomain": leaf["subdomain"],
                                    "concept": leaf["concept"], **search_info, "candidates": candidates})
            candidates, search_info = search_cache[discovery_key]
            start = leaf["sample_index"] % len(candidates)
            rotated = candidates[start:] + candidates[:start]
            used = claimed.setdefault(discovery_key, set())
            ordered = [item for item in rotated if item["url"] not in used] + [item for item in rotated if item["url"] in used]
            errors = []
            for candidate in ordered:
                page_source = {"kind": "web", "uri": candidate["url"],
                               **{key: source[key] for key in ("max_bytes", "timeout", "user_agent") if key in source}}
                source_key = stable_json(page_source)
                try:
                    if source_key not in cache:
                        cache[source_key] = _retrieve(page_source, plan_path.parent)
                    content, extra = cache[source_key]
                except Exception as error:
                    errors.append(f"{candidate['url']}: {error}")
                    continue
                reused = candidate["url"] in used
                used.add(candidate["url"])
                extra_search = {"search_query": query, "search_provider": search_info["provider"],
                                "search_rank": candidate["rank"], "discovered_url": candidate["url"],
                                "reused_candidate": reused}
                break
            else:
                raise RuntimeError("all discovered pages failed: " + "; ".join(errors))
        else:
            source_key = stable_json(source)
            if source_key not in cache:
                cache[source_key] = _retrieve(source, plan_path.parent)
            content, extra = cache[source_key]
        digest = hashlib.sha256(content.encode()).hexdigest()
        material_id = f"mat-{fingerprint([profile, leaf['domain'], leaf['subdomain'], leaf['concept'], index, digest])[:16]}"
        path = material_dir / f"{material_id}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        rows.append({**leaf, "material_id": material_id, "profile": profile, "content_path": str(path.relative_to(output)),
                     "provenance": {"kind": source["kind"],
                                    "uri": source.get("uri", extra_search.get("discovered_url", "inline")),
                                    **extra, **extra_search, "sha256": digest, "bytes": len(content.encode()),
                                    "license": source.get("license"), "retrieved_at": datetime.now(timezone.utc).isoformat()}})
    counts = {family: sum(row["family"] == family for row in rows) for family in sorted({row["family"] for row in rows})}
    probabilities = [value / len(rows) for value in counts.values() if value]
    nodes, edges = {}, []
    for leaf in _leaves(plan):
        domain_id = f"domain:{leaf['domain']}"
        subdomain_id = f"subdomain:{leaf['family']}"
        concept_id = f"concept:{leaf['family']}.{leaf['concept']}"
        nodes.update({domain_id: {"id": domain_id, "kind": "domain", "name": leaf["domain"]},
                      subdomain_id: {"id": subdomain_id, "kind": "subdomain", "name": leaf["subdomain"]},
                      concept_id: {"id": concept_id, "kind": "concept", "name": leaf["concept"]}})
        edges.extend([{"from": domain_id, "to": subdomain_id, "relation": "has_subdomain"},
                      {"from": subdomain_id, "to": concept_id, "relation": "has_concept"}])
    write_jsonl(output / "materials.jsonl", rows)
    write_jsonl(output / "discovery.jsonl", discoveries)
    graph = {"version": "domain-material-v1", "profile": profile, "seed": seed,
             "nodes": sorted(nodes.values(), key=lambda item: item["id"]),
             "edges": list({stable_json(edge): edge for edge in edges}.values()),
             "target_count": len(rows), "realized_distribution": counts,
             "unique_domains": len({row["domain"] for row in rows}),
             "unique_subdomains": len({row["family"] for row in rows}),
             "unique_concepts": len({(row["family"], row["concept"]) for row in rows}),
             "effective_family_count": math.exp(-sum(value * math.log(value) for value in probabilities)),
             "unique_sources": len(cache), "unique_material_hashes": len({row["provenance"]["sha256"] for row in rows}),
             "search_queries": len(search_cache),
             "discovered_urls": len({item["url"] for row in discoveries for item in row["candidates"]}),
             "materials": str(output / "materials.jsonl"), "discovery": str(output / "discovery.jsonl")}
    write_json(output / "domain_graph.json", graph)
    return graph
