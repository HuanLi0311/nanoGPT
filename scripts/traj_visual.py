"""Render trajectory artifacts as step-by-state-lane PNGs."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from tempfile import TemporaryDirectory


LANES = {"Observation": 0, "Assistant": 1, "Action": 2, "Environment": 3, "Evaluation": 4}
COLORS = {
    "observation": "#64748b",
    "assistant": "#475569",
    "action": "#2563eb",
    "success": "#15803d",
    "failure": "#b91c1c",
}


def payload_for(event: dict) -> dict:
    payload = event.get("payload", event)
    if isinstance(event.get("message"), dict):
        payload = {**event["message"], "type": event.get("type")}
    return payload if isinstance(payload, dict) else {}


def text_content(value) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    return "".join(
        str(block.get("text", ""))
        for block in value
        if isinstance(block, dict) and block.get("type") in {"input_text", "output_text", "text"}
    )


def short(value: object, limit: int = 26) -> str:
    text = "".join(char if char.isprintable() and ord(char) <= 0xFFFF and not 0xE000 <= ord(char) <= 0xF8FF else " " for char in str(value or ""))
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[:limit - 3]}..."


def call_label(name: object, arguments: object) -> tuple[str, str]:
    parsed = arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            parsed = arguments
    if isinstance(parsed, dict):
        detail = parsed.get("cmd", parsed.get("command", parsed.get("patch", "")))
    else:
        detail = parsed
    return short(name, 22) or "tool call", short(detail, 140)


def add_node(nodes: list[dict], lane: str, title: str, detail: str, color: str, call_id: object = None, exit_code: object = None) -> None:
    nodes.append({"lane": lane, "title": title, "detail": detail, "color": color, "call_id": call_id, "exit_code": exit_code})


def nodes_for(artifact: dict) -> list[dict]:
    nodes: list[dict] = []
    for event in artifact.get("events", []):
        payload = payload_for(event)
        kind = payload.get("type")
        if kind == "message":
            role = payload.get("role")
            content = text_content(payload.get("content"))
            if role == "user":
                add_node(nodes, "Observation", "user", short(content, 180), COLORS["observation"])
            elif role == "assistant" and content.strip():
                add_node(nodes, "Assistant", "assistant", short(content, 180), COLORS["assistant"])
            for block in payload.get("content", []) if isinstance(payload.get("content"), list) else []:
                if isinstance(block, dict) and block.get("type") in {"tool_use", "function_call"}:
                    title, detail = call_label(block.get("name"), block.get("input", block.get("arguments")))
                    add_node(nodes, "Action", title, detail, COLORS["action"], block.get("id", block.get("call_id")))
        elif kind in {"function_call", "custom_tool_call"}:
            title, detail = call_label(payload.get("name"), payload.get("arguments", payload.get("input")))
            add_node(nodes, "Action", title, detail, COLORS["action"], payload.get("call_id"))
        elif kind == "exec_command_end":
            code = payload.get("exit_code")
            color = COLORS["success"] if code == 0 else COLORS["failure"]
            output = payload.get("output", payload.get("aggregated_output", payload.get("formatted_output", "")))
            add_node(nodes, "Environment", f"exit {code}", short(output or payload.get("status"), 180), color, payload.get("call_id"), code)

    classification = artifact.get("classification", {})
    bucket = classification.get("bucket", "unknown")
    color = COLORS["success"] if bucket == "success" else COLORS["failure"]
    add_node(nodes, "Evaluation", "trace result", f"{bucket} · {classification.get('reason', 'unclassified')}", color)
    return nodes


def render(artifact_path: Path, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D

    font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    nodes = nodes_for(artifact)
    count = len(nodes)
    columns = min(8, max(1, (count + 29) // 30))
    rows = (count + columns - 1) // columns
    width = min(28, max(13, 7.4 + count * 0.22))
    trajectory_height = 4.6
    detail_height = max(2.6, rows * 0.42)
    fig = plt.figure(figsize=(width, trajectory_height + detail_height + 1.1), facecolor="white")
    grid = fig.add_gridspec(2, 1, height_ratios=[trajectory_height, detail_height])
    axis = fig.add_subplot(grid[0])
    details_axis = fig.add_subplot(grid[1])

    steps = list(range(count))
    lanes = [LANES[node["lane"]] for node in nodes]
    colors = [node["color"] for node in nodes]
    for lane, y in LANES.items():
        axis.axhspan(y - 0.43, y + 0.43, color="#f8fafc" if y % 2 == 0 else "white", zorder=0)
        axis.axhline(y, color="#e2e8f0", linewidth=0.8, zorder=1)
    axis.plot(steps, lanes, color="#334155", linewidth=1.8, alpha=0.9, solid_capstyle="round", solid_joinstyle="round", zorder=2)
    axis.scatter(steps, lanes, s=38, color=colors, edgecolor="white", linewidth=0.9, zorder=3)
    turns = [step for step, lane in enumerate(lanes) if step in {0, count - 1} or lane != lanes[step - 1] or lane != lanes[step + 1]]
    axis.scatter([steps[step] for step in turns], [lanes[step] for step in turns], s=96, facecolors="none", edgecolors=[colors[step] for step in turns], linewidth=1.35, zorder=4)

    source = short(artifact.get("source_file", artifact_path.name), 76)
    episode = artifact.get("episode_index", "?")
    classification = artifact.get("classification", {})
    bucket = classification.get("bucket", "unknown")
    exits = [item.get("exit_code") for item in classification.get("exit_codes", []) if isinstance(item, dict)]
    nonzero = sum(code != 0 for code in exits)
    fig.suptitle("Agent trajectory", x=0.045, y=0.985, ha="left", fontsize=15, fontweight="bold", color="#0f172a", fontproperties=font)
    axis.set_title(f"{bucket.upper()}  ·  {source}  ·  episode {episode}", loc="left", fontsize=8.5, color=COLORS["success"] if bucket == "success" else COLORS["failure"], pad=18, fontproperties=font, parse_math=False)
    axis.text(1, 1.035, f"{count} events · {len(exits)} command exits · {nonzero} nonzero", transform=axis.transAxes, ha="right", va="bottom", fontsize=7.5, color="#475569", fontproperties=font)
    axis.set_xlabel("step", fontsize=9, fontproperties=font)
    axis.set_yticks(list(LANES.values()), list(LANES.keys()), fontproperties=font)
    tick_step = 1 if count <= 24 else 5 if count <= 80 else 20 if count <= 200 else 50
    ticks = list(range(0, count, tick_step))
    if ticks[-1] != count - 1:
        ticks.append(count - 1)
    axis.set_xticks(ticks)
    axis.tick_params(axis="x", labelsize=7, colors="#475569")
    axis.tick_params(axis="y", labelsize=8, colors="#334155")
    axis.grid(axis="x", color="#e2e8f0", linewidth=0.55)
    axis.set_xlim(-0.6, count - 0.4)
    axis.set_ylim(-0.58, len(LANES) - 0.42)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.legend(handles=[Line2D([], [], marker="o", color="none", markerfacecolor=COLORS["observation"], markeredgecolor="white", label="observation"), Line2D([], [], marker="o", color="none", markerfacecolor=COLORS["assistant"], markeredgecolor="white", label="assistant"), Line2D([], [], marker="o", color="none", markerfacecolor=COLORS["action"], markeredgecolor="white", label="tool call"), Line2D([], [], marker="o", color="none", markerfacecolor=COLORS["success"], markeredgecolor="white", label="exit 0"), Line2D([], [], marker="o", color="none", markerfacecolor=COLORS["failure"], markeredgecolor="white", label="exit nonzero")], loc="upper right", ncol=5, fontsize=6.8, frameon=False, handletextpad=0.35, columnspacing=0.9)

    wrap_width = max(24, min(90, 320 // columns))
    details_axis.set_xlim(0, columns)
    details_axis.set_ylim(rows, 0)
    details_axis.set_axis_off()
    details_axis.text(0, -0.42, "Event log", ha="left", va="top", fontsize=10, fontweight="bold", color="#0f172a", fontproperties=font, clip_on=False)
    for column in range(1, columns):
        details_axis.axvline(column, color="#cbd5e1", linewidth=0.8)
    for row in range(rows + 1):
        details_axis.axhline(row, color="#e2e8f0", linewidth=0.55)
    for step, node in enumerate(nodes):
        column, row = divmod(step, rows)
        x, y = column + 0.045, row + 0.5
        detail = short(node["detail"], wrap_width * 2)
        wrapped = "\n".join(textwrap.wrap(detail, width=wrap_width, break_long_words=True, break_on_hyphens=False)[:2]) or "--"
        details_axis.scatter(x, y, s=22, color=node["color"], edgecolor="white", linewidth=0.7, zorder=2)
        details_axis.text(x + 0.035, y - 0.34, f"{step:03d}  {node['title']}", ha="left", va="top", fontsize=6.8, fontweight="bold", color=node["color"], fontproperties=font, parse_math=False)
        details_axis.text(x + 0.035, y - 0.02, wrapped, ha="left", va="top", fontsize=6.7, color="#334155", linespacing=1.25, fontproperties=font, parse_math=False)

    fig.subplots_adjust(left=0.045, right=0.99, top=0.925, bottom=0.025, hspace=0.14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130 if count <= 120 else 110, facecolor=fig.get_facecolor())
    plt.close(fig)


def self_check() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        artifact = {
            "source_file": "sample.jsonl", "episode_index": 0,
            "classification": {"bucket": "failure", "exit_codes": [{"exit_code": 1}]},
            "events": [
                {"payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "fix it"}]}},
                {"payload": {"type": "function_call", "name": "exec_command", "arguments": "{\"cmd\":\"echo ${HOME}\"}", "call_id": "c1"}},
                {"payload": {"type": "exec_command_end", "call_id": "c1", "exit_code": 1, "status": "failed"}},
            ],
        }
        source, output = root / "sample.json", root / "sample.png"
        source.write_text(json.dumps(artifact), encoding="utf-8")
        nodes = nodes_for(artifact)
        assert nodes[1]["title"] == "exec_command"
        assert nodes[1]["detail"] == "echo ${HOME}"
        assert nodes[2]["title"] == "exit 1"
        render(source, output)
        assert output.read_bytes().startswith(b"\x89PNG")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    paths = sorted(args.input_dir.glob("*.json"))[:args.limit]
    for index, path in enumerate(paths, 1):
        render(path, args.output_dir / f"{path.stem}.png")
        if index % 50 == 0 or index == len(paths):
            print(f"rendered={index}/{len(paths)}", flush=True)


if __name__ == "__main__":
    main()
