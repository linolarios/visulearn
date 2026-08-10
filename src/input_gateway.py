"""Input Gateway — validate + normalize a raw topic string. See AGENT.md §4.

Maps free text to {canonical, category, family} against the taxonomy
(LeetCode patterns, GoF, CLRS). Unknown topics route to a disambiguation LLM call.
"""
from __future__ import annotations
from models import Category  # noqa: F401


def normalize_topic(raw: str) -> dict:
    """Return {'canonical': str, 'category': 'dsa'|'design_pattern', 'family': str}."""
    raise NotImplementedError("Implement per AGENT.md §4 (Input Gateway).")
