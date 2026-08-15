"""Pipeline orchestration (front half): Topic -> validated Script JSON + Storyboard.

Wires the completed stages — Input Gateway, Research Agent, Script Engine, Storyboard
Generator — behind one callable so the `visulearn.py script` command (AGENT.md §6) and
future batch runs share the same path. Every LLM/network seam is injected so the
orchestrator is testable offline with stubs (AGENT.md §7); CI installs only
pydantic/pytest/requests, so this module avoids heavy renderer/TTS/assembly deps.

Seams:
  * script_provider — Script Engine provider object (stub in tests; None -> build_provider).
  * fact_sheet      — dict (used verbatim), callable canonical->dict, or None (research_agent).
  * disambiguate    — input_gateway.normalize_topic provider (None -> reject unknown topics).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Union

from input_gateway import normalize_topic
from research_agent import build_fact_sheet
from script_engine import generate_script
from storyboard_generator import plan
from models import Storyboard


def make_script(
        topic: str,
        out_path: Path,
        *,
        script_provider: Optional[Any] = None,
        fact_sheet: Optional[Union[dict, Callable[[str], dict]]] = None,
        disambiguate: Optional[Callable[[str], Optional[dict]]] = None,
        research_timeout_s: float = 8.0,
) -> tuple[Path, Storyboard]:
    """Resolve, ground, generate, and persist a validated Script.

    Returns (script_path, storyboard). Raises on any irrecoverable failure (topic
    rejection, script-engine hard-fail). A path is only written after a fully valid
    Script passes the gate, so a failure never leaves a half-baked artifact.
    """
    resolved = normalize_topic(topic, provider=disambiguate)
    canonical = resolved["canonical"]

    if isinstance(fact_sheet, dict):
        facts = fact_sheet
    elif callable(fact_sheet):
        facts = fact_sheet(canonical)
    else:
        facts = build_fact_sheet(canonical, timeout_s=research_timeout_s)

    script = generate_script(canonical, facts, provider=script_provider)
    storyboard = plan(script)  # rule-based, deterministic, no LLM

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(script.model_dump_json(indent=2))
    return out_path, storyboard
