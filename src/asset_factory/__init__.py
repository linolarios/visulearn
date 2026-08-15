"""Asset Factory — route a Scene to its renderer, failing SOFT to a Pillow slide.

AGENT.md Golden Rule 4: 'Any renderer failure falls back to a Pillow slide.' Renderers
are imported lazily so importing this package never pulls an optional system dependency
(graphviz/manim/mermaid) at module load — only the Pillow slide renderer needs Pillow,
which runs in CI.

Seam: render_scene(scene, out_path) -> Path is the single public entry; tests exercise
routing + fail-soft + determinism here (AGENT.md §7 renderer tests).
"""
from __future__ import annotations

import logging
from pathlib import Path

from models import Renderer, Scene

log = logging.getLogger(__name__)


def _dispatchers() -> dict:
    # Delayed import keeps this package importable without optional render deps.
    from asset_factory import (animation_renderer, code_renderer, flowchart_renderer,
                               graph_renderer, slide_renderer)
    return {
        Renderer.PILLOW: slide_renderer.render,
        Renderer.GRAPHVIZ: graph_renderer.render,
        Renderer.MERMAID: flowchart_renderer.render,
        Renderer.MANIM: animation_renderer.render,
        Renderer.PYGMENTS: code_renderer.render,
    }


def _slide(scene: Scene, out_path: Path) -> Path:
    from asset_factory import slide_renderer
    return slide_renderer.render(scene, out_path)


def render_scene(scene: Scene, out_path: Path) -> Path:
    """Render one Scene to an image; degrade to a Pillow slide on any failure (GR4)."""
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

