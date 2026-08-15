"""Asset Factory tests. Offline: Pillow-only (no graphviz/manim/mermaid/kokoro).

Proves the dispatcher seam (AGENT.md Golden Rule 4) at the render_scene boundary:
  * slide renderer makes a real 1920x1080 PNG on disk;
  * dispatcher routes a PILLOW scene straight to the slide renderer;
  * any non-Pillow renderer whose implementation still raises (the stubs) FALLS BACK
    to a Pillow slide instead of crashing the run;
  * determinism: same scene rendered twice -> byte-identical PNGs (for caching);
  * a failing slide renderer does not recurse forever (no infinite fallback).
"""
import pytest
from PIL import Image

from asset_factory import render_scene, slide_renderer
from models import Renderer, Scene, VisualCue

W, H = 1920, 1080


def _scene(renderer: Renderer, cue: VisualCue, seg_id: int = 1, duration: int = 5) -> Scene:
    return Scene(segment_id=seg_id, visual_cue=cue, renderer=renderer, duration_seconds=duration)


def test_slide_renderer_makes_1920x1080_png(tmp_path):
    out = slide_renderer.render(_scene(Renderer.PILLOW, VisualCue.TITLE_CARD), tmp_path / "s.png")
    assert out.exists()
    with Image.open(out) as img:
        assert img.size == (W, H)
        assert img.format == "PNG"


def test_dispatcher_routes_pillow_scene_to_slide(tmp_path):
    path = render_scene(_scene(Renderer.PILLOW, VisualCue.WHAT_IS), tmp_path / "a.png")
    assert path.exists()
    with Image.open(path) as img:
        assert img.size == (W, H)


@pytest.mark.parametrize("renderer,cue", [
    (Renderer.MANIM, VisualCue.ALGORITHM_ANIM),
    (Renderer.GRAPHVIZ, VisualCue.DS_GRAPH),
    (Renderer.MERMAID, VisualCue.FLOWCHART),
    (Renderer.PYGMENTS, VisualCue.CODE_WALKTHROUGH),
])
def test_failing_renderer_falls_back_to_pillow_slide(tmp_path, renderer, cue):
    # Each renderer stub raises NotImplementedError -> degrade to a Pillow slide, not a crash.
    path = render_scene(_scene(renderer, cue), tmp_path / "fb.png")
    assert path.exists()
    with Image.open(path) as img:
        assert img.size == (W, H)


def test_same_scene_renders_byte_identical(tmp_path):
    scene = _scene(Renderer.PILLOW, VisualCue.TITLE_CARD, seg_id=3, duration=7)
    a = render_scene(scene, tmp_path / "a.png")
    b = render_scene(scene, tmp_path / "b.png")
    assert a.read_bytes() == b.read_bytes()   # deterministic output → idempotent caching


def test_slide_failure_does_not_recurse(tmp_path, monkeypatch):
    # If the Pillow slide renderer itself fails, there is nothing softer to degrade to:
    # the dispatcher must raise, not recurse forever.
    def boom(scene, out_path):
        raise RuntimeError("slide broken")

    monkeypatch.setattr(slide_renderer, "render", boom)
    with pytest.raises(RuntimeError, match="slide broken"):
        render_scene(_scene(Renderer.PILLOW, VisualCue.TITLE_CARD), tmp_path / "x.png")
