"""Draw three implementation-specific Agent rollout pipes.

The large arrow in each lane is the semantic rollout boundary: the output or
observation produced at round ``t`` becomes the input to round ``t + 1``.
The figure is a schematic of control boundaries, not a performance plot.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath


OUT = Path(__file__).resolve().parent

# High-contrast semantic roles.  State names and labels are always present;
# color is only a redundant cue.
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
INK = "#1F2933"
MUTED = "#5B6770"
LINE = "#C9D2DA"
PANEL = "#F8FAFC"
WHITE = "#FFFFFF"
FILL = {
    "input": "#EAF2F8",
    "generate": "#E5F1F8",
    "tools": "#FBECE5",
    "output": "#E8F4EF",
}
EDGE = {"input": BLUE, "generate": BLUE, "tools": ORANGE, "output": GREEN}


def box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str = "",
    *,
    facecolor: str = WHITE,
    edgecolor: str = LINE,
    linewidth: float = 0.9,
    linestyle: str = "-",
    fontsize: float = 7.0,
    weight: str = "normal",
    color: str = INK,
    radius: float = 0.014,
    zorder: int = 2,
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if text:
        ax.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            weight=weight,
            linespacing=1.12,
            zorder=zorder + 1,
        )
    return patch


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    linewidth: float = 1.1,
    linestyle: str = "-",
    mutation_scale: float = 11,
    connectionstyle: str = "arc3",
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        connectionstyle=connectionstyle,
        zorder=5,
    )
    ax.add_patch(patch)
    return patch


def rollout_loop(ax, accent: str, label: str):
    """Draw a large, explicit output-to-next-input loop under the pipe."""
    path = MplPath(
        [
            (0.925, 0.555),  # output edge
            (0.95, 0.37),
            (0.76, 0.205),
            (0.53, 0.205),
            (0.30, 0.205),
            (0.075, 0.34),
            (0.075, 0.535),  # input edge
        ],
        [
            MplPath.MOVETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
        ],
    )
    patch = FancyArrowPatch(
        path=path,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.65,
        color=accent,
        zorder=4,
    )
    ax.add_patch(patch)
    ax.text(
        0.50,
        0.125,
        textwrap.fill(label, 75),
        ha="center",
        va="center",
        fontsize=5.4,
        color=accent,
        weight="bold",
        linespacing=1.1,
        zorder=6,
    )
    ax.text(
        0.50,
        0.065,
        "rollout boundary: output at t → input at t + 1",
        ha="center",
        va="center",
        fontsize=5.0,
        color=MUTED,
        zorder=6,
    )


def stage(ax, x, title, detail, kind, width=0.18):
    title_text = title
    if title == "PROCESSING_TOOLS":
        title_text = "PROCESSING\nTOOLS"
    box(
        ax,
        x,
        0.45,
        width,
        0.24,
        facecolor=FILL[kind],
        edgecolor=EDGE[kind],
        linewidth=1.1,
    )
    ax.text(
        x + width / 2,
        0.615,
        title_text,
        ha="center",
        va="center",
        fontsize=6.0 if title != "PROCESSING_TOOLS" else 5.7,
        color=INK,
        weight="bold",
        linespacing=1.02,
        zorder=4,
    )
    ax.text(
        x + width / 2,
        0.505,
        textwrap.fill(detail, 24),
        ha="center",
        va="center",
        fontsize=5.2,
        color=INK,
        linespacing=1.08,
        zorder=4,
    )


def draw_lane(ax, spec):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    accent = spec["accent"]
    box(ax, 0.01, 0.02, 0.98, 0.96, facecolor=PANEL, edgecolor=accent, linewidth=1.15)

    # Lane identity.
    box(
        ax,
        0.03,
        0.855,
        0.205,
        0.095,
        spec["name"],
        facecolor=accent,
        edgecolor=accent,
        linewidth=0,
        fontsize=7.8,
        weight="bold",
        color=WHITE,
    )
    ax.text(
        0.965,
        0.902,
        spec["kind"],
        ha="right",
        va="center",
        fontsize=5.8,
        color=accent,
        weight="bold",
    )

    # The outer runtime is intentionally above the inner pipe.
    box(ax, 0.03, 0.705, 0.94, 0.095, facecolor=WHITE, edgecolor=LINE, linewidth=0.8)
    ax.text(
        0.05,
        0.768,
        "OUTER RUNTIME",
        ha="left",
        va="center",
        fontsize=5.7,
        color=accent,
        weight="bold",
    )
    ax.text(
        0.18,
        0.768,
        textwrap.fill(spec["outer"], 125),
        ha="left",
        va="center",
        fontsize=5.35,
        color=INK,
    )

    ax.text(
        0.04,
        0.675,
        "INNER PIPE  /  one rollout interaction",
        ha="left",
        va="center",
        fontsize=5.7,
        color=accent,
        weight="bold",
    )

    # A separate pipe for each implementation.
    stage(ax, 0.035, "INPUT / PENDING", spec["input"], "input", width=0.16)
    stage(ax, 0.245, "GENERATING", spec["generate"], "generate", width=0.18)
    stage(ax, 0.475, "PROCESSING_TOOLS", spec["tools"], "tools", width=0.18)
    stage(ax, 0.715, "OUTPUT / TERMINATED", spec["output"], "output", width=0.21)

    # Forward pipe arrows.
    arrow(ax, (0.195, 0.57), (0.235, 0.57), color=accent)
    arrow(ax, (0.425, 0.57), (0.465, 0.57), color=accent)
    arrow(ax, (0.655, 0.57), (0.705, 0.57), color=accent)

    rollout_loop(ax, accent, spec["loop"])


def main():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )

    specs = [
        {
            "name": "Verl ToolAgentLoop",
            "kind": "TOKEN-LEVEL RL ROLLOUT",
            "accent": BLUE,
            "outer": "AgentLoopManager / Worker: rollout scheduling, batching, tokenizer/server wiring; trainer-facing AgentLoopOutput.",
            "input": "prompt_ids + tool schemas",
            "generate": "server generate\nids / logprobs",
            "tools": "Function / BaseTool\nparallel calls → tokens",
            "output": "ids + mask\nmetrics / reward boundary",
            "loop": "tool-response tokens + partial rollout become the next generation input; final output goes to the trainer.",
        },
        {
            "name": "Codex",
            "kind": "STREAMING SESSION LOOP",
            "accent": ORANGE,
            "outer": "RegularTask + SessionTask / Session: cancellation, persistence, compaction, queued input, hooks, subagents and turn scheduling.",
            "input": "history + pending\nturn context",
            "generate": "stream response\nFunction / Custom call",
            "tools": "parallel runtime\nappend outputs",
            "output": "events + transcript\nfollow-up",
            "loop": "turn result, transcript updates or queued follow-up become the next run_turn context; task verification is external.",
        },
        {
            "name": "Prime",
            "kind": "SESSION + CONTINUATION LOOP",
            "accent": GREEN,
            "outer": "AgentSession + autonomous/headless: persistence, tool hooks/events, compaction, goals, child lifecycles and quality-gated continuation.",
            "input": "session state\nprompt + registry",
            "generate": "underlying agent\nresponse + calls",
            "tools": "registry + hooks\nemit tool_result",
            "output": "events + transcript\ngate result",
            "loop": "tool events and the continuation gate become the next prompt; autonomous/headless may inject a repair turn.",
        },
    ]

    # Provisional double-column manuscript figure: 180 mm wide, with vector
    # PDF/SVG as the masters.  The target journal is not specified yet.
    fig = plt.figure(figsize=(180 / 25.4, 140 / 25.4), facecolor=WHITE)
    fig.text(
        0.04,
        0.965,
        "Three Agent loop strategies as rollout pipes",
        ha="left",
        va="top",
        fontsize=11.5,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.04,
        0.925,
        "Each implementation has its own inner interaction pipe and outer runtime. The large return arrow is the rollout boundary from output at round t to input at round t + 1.",
        ha="left",
        va="top",
        fontsize=6.4,
        color=MUTED,
    )
    fig.text(
        0.965,
        0.925,
        "schematic · vector master",
        ha="right",
        va="top",
        fontsize=5.5,
        color=MUTED,
    )

    ys = [0.64, 0.385, 0.13]
    for y, spec in zip(ys, specs):
        ax = fig.add_axes([0.04, y, 0.92, 0.225])
        draw_lane(ax, spec)

    fig.text(
        0.04,
        0.055,
        "nanoGPT placement: runner.ts currently spans the pipe plus session state/persistence and task verification. A cleaner target is inner AgentLoop → outer SessionRunner → harness verifier; Verl reward/task verification remains external unless explicitly returned as an observation.",
        ha="left",
        va="center",
        fontsize=5.3,
        color=INK,
        bbox={
            "boxstyle": "round,pad=0.40",
            "facecolor": "#F3F0F7",
            "edgecolor": PURPLE,
            "linewidth": 0.8,
        },
    )
    fig.text(
        0.04,
        0.018,
        "Color is redundant with explicit state labels; arrows encode control flow, not performance or reward magnitude.",
        ha="left",
        va="center",
        fontsize=5.0,
        color=MUTED,
    )

    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"agent_loop_strategies.{ext}", dpi=300, bbox_inches=None)
    plt.close(fig)


if __name__ == "__main__":
    main()
