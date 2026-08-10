"""Research Agent — ground the LLM with real facts. See AGENT.md §1/§4.

Primary source: Wikipedia API (deterministic, no cloud-IP blocking).
Fallback: ddgs  (NOT duckduckgo-search, which was frozen Jul 2025).
Wrap every network call in a timeout with a graceful empty-result path.
"""
from __future__ import annotations

try:
    from ddgs import DDGS  # new package name
except ImportError:  # pragma: no cover - back-compat only
    from duckduckgo_search import DDGS  # type: ignore  # noqa: F401


def build_fact_sheet(canonical_topic: str, timeout_s: float = 8.0) -> dict:
    """Return {definition, inventor, year, use_cases, misconceptions}. Never raise on network error."""
    raise NotImplementedError("Implement per AGENT.md §4 (Research Agent).")
