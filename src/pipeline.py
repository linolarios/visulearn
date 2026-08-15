"""Pipeline orchestration: Topic -> validated Script JSON + Storyboard + asset PNGs.

Wires the completed stages — Input Gateway, Research Agent, Script Engine, Storyboard
Generator, and the Asset Factory — behind small callables so the `visulearn.py script`
command (AGENT.md §6) and future batch runs share one path. Every LLM/network/render
seam is injected or delegable so the orchestrator is testable offline with stubs
(AGENT.md §7); CI installs only pydantic/pytest/requests/Pillow, so this module avoids
heavy renderer/TTS/assembly deps.

Seams:
  * script_provider — Script Engine provider object (stub in tests; None -> build_provider).
  * fact_sheet      — dict (used verbatim), callable canonical->dict, or None (research_agent).
  * disambiguate    — input_gateway.normalize_topic provider (None -> reject unknown topics).
  * render_scene    — asset_factory dispatcher (monkeypatched in tests to count/fail).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional, Union

from asset_factory import render_scene
from input_gateway import normalize_topic
from research_agent import build_fact_sheet
from script_engine import generate_script
from storyboard_generator import plan
from models import Storyboard

log = logging.getLogger(__name__)


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


def make_assets(storyboard: Storyboard, out_dir: Path, *,
                code_templates: Optional[dict] = None, force: bool = False) -> dict[int, Path]:
    """Render every Scene to out_dir/<segment_id>.png (Pillow fail-soft + caching).

    code_templates (the Script's code_template dict) is threaded ONLY to code scenes, so
    the shared renderer signature stays scene/out_path. Golden Rule 5 caches existing
    non-empty PNGs unless force=True; Golden Rule 4 fails a single scene soft (the
    dispatcher already degrades to a Pillow slide) instead of the whole run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    assets: dict[int, Path] = {}
    for scene in storyboard.scenes:
        path = out_dir / f"{scene.segment_id}.png"
        if not force and path.exists() and path.stat().st_size > 0:
            assets[scene.segment_id] = path
            continue
        kwargs = {"code_templates": code_templates} if code_templates is not None else {}
        try:
            assets[scene.segment_id] = render_scene(scene, path, **kwargs)
        except Exception as exc:  # noqa: BLE001 - fail-soft per scene, never fail-whole
            log.warning("asset for scene %s failed; skipping: %s", scene.segment_id, exc)
    return assets

