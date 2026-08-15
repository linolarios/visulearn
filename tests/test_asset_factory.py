"""Asset Factory tests. Offline: Pillow + Pygments only (no graphviz/manim/mermaid).

Proves the dispatcher seam (AGENT.md Golden Rule 4) at the render_scene boundary:
  * slide renderer makes a real 1920x1080 PNG on disk;
  * dispatcher routes a PILLOW scene straight to the slide renderer;
  * any stubbed renderer (MANIM/GRAPHVIZ/MERMAID still raise) FALLS BACK to a Pillow
    slide with VISIBLE placeholder pixels — never a blank frame, never a crash;
  * determinism: same scene rendered twice -> byte-identical PNGs;
  * a failing slide renderer does not recurse forever;
  * code_walkthrough is end-to-end routed to code_renderer (Pygments, not the fallback)
    and produces a non-blank, deterministic, highlighted slide.
"""
import pytest
from PIL import Image

from asset_factory import code_renderer, render_scene, slide_renderer
from models import Renderer, Scene, VisualCue

W, H = 1920, 1080


def _scene(renderer: Renderer, cue: VisualCue, seg_id: int = 1, duration: int = 5) -> Scene:
    return Scene(segment_id=seg_id, visual_cue=cue, renderer=renderer, duration_seconds=duration)


def _has_visible_pixels(path) -> bool:
    """A rendered fallback must contain non-background content, not a blank frame."""
    with Image.open(path) as img:
        return img.getbbox() is not None


def test_slide_renderer_makes_1920x1080_png(tmp_path):
    out = slide_renderer.render(_scene(Renderer.PILLOW, VisualCue.TITLE_CARD), tmp_path / "s.png")
    assert out.exists()
    with Image.open(out) as img:
        assert img.size == (W, H)
        assert img.format == "PNG"
    assert _has_visible_pixels(out)                      # not a blank frame


def test_dispatcher_routes_pillow_scene_to_slide(tmp_path):
    path = render_scene(_scene(Renderer.PILLOW, VisualCue.WHAT_IS), tmp_path / "a.png")
    assert path.exists()
    with Image.open(path) as img:
        assert img.size == (W, H)
    assert _has_visible_pixels(path)


@pytest.mark.parametrize("renderer,cue", [
    (Renderer.MANIM, VisualCue.ALGORITHM_ANIM),
    (Renderer.GRAPHVIZ, VisualCue.DS_GRAPH),
    (Renderer.MERMAID, VisualCue.FLOWCHART),
])
def test_failing_stubbed_renderer_falls_back_to_visible_pillow_slide(tmp_path, renderer, cue):
    # These renderers are still stubs (raise NotImplementedError) -> the dispatcher must
    # degrade to a Pillow slide that is actually VISIBLE, not a blank frame (Golden Rule 4).
    path = render_scene(_scene(renderer, cue), tmp_path / "fb.png")
    assert path.exists()
    with Image.open(path) as img:
        assert img.size == (W, H)
    assert _has_visible_pixels(path)                     # the placeholder is rendered, not blank


def test_same_scene_renders_byte_identical(tmp_path):
    scene = _scene(Renderer.PILLOW, VisualCue.TITLE_CARD, seg_id=3, duration=7)
    a = render_scene(scene, tmp_path / "a.png")
    b = render_scene(scene, tmp_path / "b.png")
    assert a.read_bytes() == b.read_bytes()              # deterministic output for caching


def test_slide_failure_does_not_recurse(tmp_path, monkeypatch):
    def boom(scene, out_path):
        raise RuntimeError("slide broken")

    monkeypatch.setattr(slide_renderer, "render", boom)
    with pytest.raises(RuntimeError, match="slide broken"):
        render_scene(_scene(Renderer.PILLOW, VisualCue.TITLE_CARD), tmp_path / "x.png")


def test_code_walkthrough_end_to_end_uses_code_renderer_not_fallback(tmp_path, monkeypatch):
    """Pygments-in-CI: a CODE_WALKTHROUGH scene must route to code_renderer, not fall back."""
    real = code_renderer.render
    calls = [0]

    def wrap(scene, code_templates, out_path):
        calls[0] += 1
        return real(scene, code_templates, out_path)

    monkeypatch.setattr(code_renderer, "render", wrap)
    scene = _scene(Renderer.PYGMENTS, VisualCue.CODE_WALKTHROUGH, seg_id=5, duration=6)
    out = tmp_path / "code.png"

    render_scene(scene, out, code_templates={"python": "def f(a):\n    return a + 1"})

    assert calls[0] == 1                                 # code renderer ran — NOT the fallback
    assert out.exists()
    with Image.open(out) as img:
        assert img.size == (W, H)
    assert _has_visible_pixels(out)


def test_code_renderer_highlighted_and_deterministic(tmp_path):
    scene = _scene(Renderer.PYGMENTS, VisualCue.CODE_WALKTHROUGH, seg_id=7)
    code = {"python": "def is_red(n):\n    return n % 2 == 0"}
    a = code_renderer.render(scene, code, tmp_path / "a.png")
    b = code_renderer.render(scene, code, tmp_path / "b.png")
    assert a.exists() and a.read_bytes() == b.read_bytes()   # deterministic (caching)
    with Image.open(a) as img:
        assert img.size == (W, H)
    assert _has_visible_pixels(a)


def test_code_renderer_empty_template_draws_placeholder(tmp_path):
    scene = _scene(Renderer.PYGMENTS, VisualCue.CODE_WALKTHROUGH, seg_id=8)
    out = code_renderer.render(scene, {}, tmp_path / "empty.png")
    assert out.exists()
    with Image.open(out) as img:
        assert img.size == (W, H)
    assert _has_visible_pixels(out)                       # "(no code template)" is visible
