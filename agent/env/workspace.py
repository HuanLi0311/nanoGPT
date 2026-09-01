"""Compatibility imports; use :mod:`agent.workspace` in new code."""

from agent.workspace.snapshot import snapshot, state_delta

__all__ = ["snapshot", "state_delta"]
