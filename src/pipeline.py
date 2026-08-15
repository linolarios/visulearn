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
from assembler import assemble
from tts_pipeline import synthesize
from models import Script, Storyboard



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


def make_video(
        topic: str,
        out_path: Path,
        *,
        workspace_dir: Optional[Path] = None,
        script_provider: Optional[Any] = None,
        fact_sheet: Optional[Union[dict, Callable[[str], dict]]] = None,
        disambiguate: Optional[Callable[[str], Optional[dict]]] = None,
        research_timeout_s: float = 8.0,
        tts: Optional[Callable[..., bytes]] = None,
        encoder: Optional[Callable[..., Path]] = None,
        force: bool = False,
        dry_run: bool = False,
) -> Path:
    """Run the full Topic -> MP4 pipeline; all LLM/tts/encoder seams are injectable.

    ``out_path`` is the output MP4 path. A private workspace (``<workspace_dir>/``) holds
    the script.json, per-scene PNGs, and per-segment WAVs for idempotent caching (GR5).

    When *dry_run=True*, only the topic-resolution and fact-grounding steps run (they are
    cheap, pure-ish lookups); no artifact is written and tts/encoder are not called.
    Returns out_path without creating it.
    """
    ws = workspace_dir or out_path.parent / f".{out_path.stem}_workspace"
    script_path = ws / "script.json"
    scenes_dir = ws / "scenes"
    audio_dir = ws / "audio"

    script = None
    if not force and script_path.exists():
        try:
            script = Script.model_validate_json(script_path.read_text())
        except Exception:  # noqa: BLE001 - corrupt cache -> regenerate
            script = None

    if script is None:
        resolved = normalize_topic(topic, provider=disambiguate)
        canonical = resolved["canonical"]
        if isinstance(fact_sheet, dict):
            facts = fact_sheet
        elif callable(fact_sheet):
            facts = fact_sheet(canonical)
        else:
            facts = build_fact_sheet(canonical, timeout_s=research_timeout_s)
        script = generate_script(canonical, facts, provider=script_provider)
        if not dry_run:
            ws.mkdir(parents=True, exist_ok=True)
            script_path.write_text(script.model_dump_json(indent=2))

    if dry_run:
        print(f"[dry-run] {topic!r} -> canonical='{script.meta.topic}' "
              f"({len(script.segments)} segments)")
        return out_path

    storyboard = plan(script)
    scene_assets = make_assets(storyboard, scenes_dir,
                               code_templates=script.code_template, force=force)
    audio = synthesize(script, audio_dir, tts=tts, force=force)
    return assemble(storyboard, scene_assets, audio, out_path, encoder=encoder)

