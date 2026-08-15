"""Asset Factory — route a Scene to its renderer, failing SOFT to a Pillow slide.

AGENT.md Golden Rule 4: 'Any renderer failure falls back to a Pillow slide.' Only the
Pillow slide renderer needs Pillow and only code_renderer needs Pygments — both ship as
wheels and run in CI; graphviz/manim/mermaid renderers import lazily and are not pulled
in by this package at module load.

Design note: code_renderer needs the Script's code templates, so it is SPECIAL-CASED
here (render_scene accepts an optional `code_templates` kwarg) rather than widening the
shared render(scene, out_path) signature that every other renderer uses.

Seam: render_scene(scene, out_path, *, code_templates=None) -> Path.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from models import Renderer, Scene

log = logging.getLogger(__name__)


def _dispatchers() -> dict:
    # Delayed import keeps this package importable without optional render deps.
    from asset_factory import (animation_renderer, flowchart_renderer,
                               graph_renderer, slide_renderer)
    return {
        Renderer.PILLOW: slide_renderer.render,
        Renderer.GRAPHVIZ: graph_renderer.render,
        Renderer.MERMAID: flowchart_renderer.render,
        Renderer.MANIM: animation_renderer.render,
    }


def _slide(scene: Scene, out_path: Path) -> Path:
    from asset_factory import slide_renderer
    return slide_renderer.render(scene, out_path)


def _render_code(scene: Scene, out_path: Path, code_templates: Optional[dict]) -> Path:
    """Code is the one renderer that needs content; also fails soft to a Pillow slide."""
    from asset_factory import code_renderer
    try:
        return code_renderer.render(scene, code_templates, out_path)
    except Exception as exc:  # noqa: BLE001 - a code-render crash degrades, never kills the run
        log.warning("code renderer failed for scene %s; falling back to Pillow: %s",
                    scene.segment_id, exc)
        return _slide(scene, out_path)


def render_scene(scene: Scene, out_path: Path, *, code_templates: Optional[dict] = None) -> Path:
    """Render one Scene to an image; degrade to a Pillow slide on any failure (GR4)."""
    if scene.renderer is Renderer.PYGMENTS:
        return _render_code(scene, out_path, code_templates)

    primary = _dispatchers()[scene.renderer]
    try:
        return primary(scene, out_path)
    except Exception as exc:  # noqa: BLE001 - a renderer crash degrades, never kills the run
        log.warning("renderer %s failed for scene %s (%s); falling back to Pillow",
                    scene.renderer.value, scene.segment_id, exc)
        if scene.renderer is Renderer.PILLOW:
            raise  # Pillow IS the fallback; nothing softer to degrade to
        try:
            return _slide(scene, out_path)
        except Exception as fb_exc:  # noqa: BLE001
            raise RuntimeError(
                f"scene {scene.segment_id} ({scene.renderer.value}) failed and its "
                f"Pillow fallback also failed: {fb_exc}"
            ) from exc
