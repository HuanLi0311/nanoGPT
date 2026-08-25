"""Draw the Agent loop boundary comparison used in the runtime design notes.

The figure separates the common per-episode interaction states from the
implementation-specific outer runtime.  It is a schematic of control flow,
not a measurement plot; all labels are sourced from the local Verl, Codex,
and Prime implementations inspected for this comparison.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = Path(__file__).resolve().parent

# Okabe-Ito-inspired high-contrast roles.  State identity is also written in
# text, so color is never the only encoding.
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
INK = "#1F2933"
MUTED = "#5B6770"
PANEL = "#F7F9FB"
INNER = "#FFFFFF"
STATE_FILL = {
    "PENDING": "#EAF2F8",
    "GENERATING": "#E5F1F8",
    "PROCESSING_TOOLS": "#FBECE5",
    "TERMINATED": "#E8F4EF",
}
STATE_EDGE = {
    "PENDING": BLUE,
    "GENERATING": BLUE,
    "PROCESSING_TOOLS": ORANGE,
    "TERMINATED": GREEN,
}


def add_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = "white",
    edgecolor: str = INK,
    linewidth: float = 1.0,
    fontsize: float = 8.0,
    weight: str = "normal",
    linestyle: str = "-",
    radius: float = 0.018,
    color: str = INK,
    ha: str = "center",
    va: str = "center",
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha=ha,
        va=va,
        fontsize=fontsize,
        color=color,
        weight=weight,
        linespacing=1.15,
        zorder=3,
    )
    return patch


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    linewidth: float = 1.2,
    linestyle: str = "-",
    mutation_scale: float = 12,
    connectionstyle: str = "arc3",
):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        connectionstyle=connectionstyle,
        zorder=4,
    )
    ax.add_patch(arrow)
    return arrow


def draw_common_states(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(
        ax,
        0.005,
        0.02,
        0.99,
        0.96,
        "",
        facecolor="#FBFCFD",
        edgecolor="#B8C2CC",
        linewidth=0.9,
        radius=0.012,
    )
    ax.text(
        0.025,
        0.90,
        "COMMON CONTROL STATES  |  semantic equivalence across implementations",
        ha="left",
        va="center",
        fontsize=8.3,
        color=MUTED,
        weight="bold",
    )

    labels = [
        ("PENDING", "context / prompt\nready"),
        ("GENERATING", "model response\nbeing produced"),
        ("PROCESSING_TOOLS", "execute tools +\nappend observations"),
        ("TERMINATED", "final / limit /\nstop policy"),
    ]
    xs = [0.035, 0.275, 0.515, 0.755]
    width = 0.18
    y = 0.28
    height = 0.46

    for state, detail in labels:
        add_box(
            ax,
            xs[labels.index((state, detail))],
            y,
            width,
            height,
            f"{state}\n{detail}",
            facecolor=STATE_FILL[state],
            edgecolor=STATE_EDGE[state],
            linewidth=1.4,
            fontsize=8.3,
            weight="bold" if state != "PROCESSING_TOOLS" else "bold",
        )

    # Forward flow and the explicit tool-observation loopback.
    for left, right in zip(xs[:-1], xs[1:]):
        add_arrow(ax, (left + width, 0.51), (right, 0.51), linewidth=1.25)

    add_arrow(
        ax,
        (xs[2] + 0.04, y + height - 0.08),
        (xs[1] + width - 0.04, y + height - 0.08),
        color=ORANGE,
        linewidth=1.15,
        connectionstyle="arc3,rad=0.25",
    )
    ax.text(
        0.445,
        0.80,
        "tool results become the next observation",
        ha="center",
        va="center",
        fontsize=7.1,
        color=ORANGE,
    )
    ax.text(
        0.70,
        0.20,
        "stop / no tool call",
        ha="center",
        va="center",
        fontsize=7.1,
        color=MUTED,
    )
    ax.text(
        0.025,
        0.075,
        "Solid arrows: state transition   |   curved arrow: tool-call loopback   |   the outer runtime may decide to continue after a turn",
        ha="left",
        va="center",
        fontsize=7.0,
        color=MUTED,
    )


def draw_panel(ax, spec):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    accent = spec["accent"]
    add_box(
        ax,
        0.015,
        0.025,
        0.97,
        0.95,
        "",
        facecolor=PANEL,
        edgecolor=accent,
        linewidth=1.35,
        radius=0.015,
    )
    ax.add_patch(
        Rectangle(
            (0.015, 0.875),
            0.97,
            0.10,
            facecolor=accent,
            edgecolor=accent,
            linewidth=0,
            zorder=2.2,
        )
    )
    ax.text(
        0.05,
        0.928,
        spec["name"],
        ha="left",
        va="center",
        fontsize=11.0,
        color="white",
        weight="bold",
        zorder=3,
    )
    ax.text(
        0.95,
        0.928,
        spec["kind"],
        ha="right",
        va="center",
        fontsize=7.0,
        color="white",
        zorder=3,
    )

    ax.text(
        0.05,
        0.844,
        "OUTER RUNTIME",
        ha="left",
        va="center",
        fontsize=7.0,
        color=accent,
        weight="bold",
    )
    ax.text(
        0.05,
        0.805,
        textwrap.fill(spec["outer"], 47),
        ha="left",
        va="top",
        fontsize=7.2,
        color=INK,
        linespacing=1.2,
    )

    # The dashed boundary is the per-turn/per-episode interaction loop.
    add_box(
        ax,
        0.045,
        0.205,
        0.91,
        0.535,
        "",
        facecolor=INNER,
        edgecolor=accent,
        linewidth=1.1,
        linestyle=(0, (3, 2)),
        radius=0.012,
    )
    ax.text(
        0.07,
        0.715,
        "INNER LOOP  /  one agent interaction",
        ha="left",
        va="center",
        fontsize=7.0,
        color=accent,
        weight="bold",
    )

    rows = [
        ("PENDING", spec["states"]["PENDING"]),
        ("GENERATING", spec["states"]["GENERATING"]),
        ("PROCESSING_TOOLS", spec["states"]["PROCESSING_TOOLS"]),
        ("TERMINATED", spec["states"]["TERMINATED"]),
    ]
    ys = [0.615, 0.495, 0.375, 0.255]
    state_x = 0.065
    state_w = 0.26
    detail_x = 0.355
    detail_w = 0.575
    row_h = 0.075

    for state, detail in rows:
        add_box(
            ax,
            state_x,
            ys[rows.index((state, detail))],
            state_w,
            row_h,
            state,
            facecolor=STATE_FILL[state],
            edgecolor=STATE_EDGE[state],
            linewidth=1.15,
            fontsize=7.1 if state != "PROCESSING_TOOLS" else 6.6,
            weight="bold",
        )
        add_box(
            ax,
            detail_x,
            ys[rows.index((state, detail))],
            detail_w,
            row_h,
            textwrap.fill(detail, 37),
            facecolor="#FFFFFF",
            edgecolor="#D1D9E0",
            linewidth=0.75,
            fontsize=6.8,
            color=INK,
        )

    # Main downward sequence; the generating-to-terminated path is shown as
    # a dotted shortcut because a no-tool response can end the interaction.
    for i in range(3):
        add_arrow(
            ax,
            (0.195, ys[i] - 0.004),
            (0.195, ys[i + 1] + row_h + 0.004),
            color=accent if i != 1 else INK,
            linewidth=1.0,
        )
    add_arrow(
        ax,
        (0.875, ys[1] + row_h / 2),
        (0.875, ys[3] + row_h / 2),
        color=MUTED,
        linewidth=0.9,
        linestyle=(0, (2, 2)),
        connectionstyle="arc3,rad=-0.32",
        mutation_scale=10,
    )
    ax.text(
        0.895,
        0.445,
        "no call",
        ha="center",
        va="center",
        rotation=90,
        fontsize=6.2,
        color=MUTED,
    )
    # Explicit loopback from tool processing to generation.
    add_arrow(
        ax,
        (0.91, ys[2] + row_h / 2),
        (0.91, ys[1] + row_h / 2),
        color=ORANGE,
        linewidth=1.0,
        connectionstyle="arc3,rad=0.55",
        mutation_scale=10,
    )
    ax.text(
        0.925,
        0.50,
        "observe",
        ha="center",
        va="center",
        rotation=90,
        fontsize=6.2,
        color=ORANGE,
    )

    add_box(
        ax,
        0.055,
        0.065,
        0.89,
        0.105,
        textwrap.fill(spec["boundary"], 54),
        facecolor="#FFFFFF",
        edgecolor=accent,
        linewidth=0.9,
        fontsize=6.7,
        color=INK,
    )


def main():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10,
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
            "kind": "RL ROLLOUT",
            "accent": BLUE,
            "outer": "AgentLoopManager / Worker: rollout scheduling, batching, tokenizer and server wiring.",
            "boundary": "TRAINING OUTPUT: AgentLoopOutput with prompt_ids, response_ids, response_mask, logprobs and metrics.",
            "states": {
                "PENDING": "build prompt token ids + tool schemas",
                "GENERATING": "server_manager.generate → parse token ids / logprobs",
                "PROCESSING_TOOLS": "FunctionTool / BaseTool dispatch; parallel calls → tool tokens",
                "TERMINATED": "no calls or length / turn cap → return rollout",
            },
        },
        {
            "name": "Codex",
            "kind": "SESSION TASK",
            "accent": ORANGE,
            "outer": "RegularTask + SessionTask / Session: cancellation, persistence, compaction, queued input, hooks and subagents.",
            "boundary": "SESSION OUTPUT: streaming protocol events and transcript; the outer task can schedule another turn.",
            "states": {
                "PENDING": "prepare turn context: history, hooks and prompt updates",
                "GENERATING": "stream response items: FunctionCall / CustomToolCall",
                "PROCESSING_TOOLS": "async parallel tool runtime → append tool outputs to transcript",
                "TERMINATED": "stop / follow-up decision; outer task may run the next turn",
            },
        },
        {
            "name": "Prime",
            "kind": "AGENT SESSION",
            "accent": GREEN,
            "outer": "AgentSession: persistence, tool hooks/events, compaction, goals and child lifecycles; autonomous/headless adds continuation gates.",
            "boundary": "SESSION OUTPUT: session events, tool events and transcript; quality gates decide continue, retry or idle.",
            "states": {
                "PENDING": "underlying agent receives prompt + session state and tool registry",
                "GENERATING": "provider / model response: assistant text + tool calls",
                "PROCESSING_TOOLS": "tool registry + hooks → emit tool_result and persist event",
                "TERMINATED": "agent result, then session gate decides continue or return idle",
            },
        },
    ]

    fig = plt.figure(figsize=(18, 11), facecolor="white")
    fig.text(
        0.03,
        0.975,
        "Agent loop strategies and runtime boundaries",
        ha="left",
        va="top",
        fontsize=17,
        color=INK,
        weight="bold",
    )
    fig.text(
        0.03,
        0.945,
        "The same interaction semantics are split differently: Verl is token-level and rollout-facing; Codex and Prime place lifecycle policy around the turn loop.",
        ha="left",
        va="top",
        fontsize=8.4,
        color=MUTED,
    )

    common = fig.add_axes([0.03, 0.765, 0.94, 0.145])
    draw_common_states(common)

    left = 0.03
    width = 0.30
    gap = 0.02
    for index, spec in enumerate(specs):
        ax = fig.add_axes([left + index * (width + gap), 0.08, width, 0.655])
        draw_panel(ax, spec)

    fig.text(
        0.03,
        0.040,
        "Boundary note for nanoGPT: runner.ts currently spans the inner interaction loop plus session state/persistence and task verification. A cleaner target split is inner AgentLoop → outer SessionRunner → harness verifier; Verl's reward/task verifier remains an external concern unless returned as an observation.",
        ha="left",
        va="center",
        fontsize=7.5,
        color=INK,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#F3F0F7",
            "edgecolor": PURPLE,
            "linewidth": 0.9,
        },
    )

    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"agent_loop_strategies.{ext}", dpi=300, bbox_inches=None)
    plt.close(fig)


if __name__ == "__main__":
    main()
