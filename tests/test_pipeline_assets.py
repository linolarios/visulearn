"""Offline tests for pipeline.make_assets — Pillow-only, no LLM/network (AGENT.md §7).

Seams: the asset_factory renderer is monkeypatched to (a) count calls — proving the
idempotent cache (Golden Rule 5) and force re-render — and (b) simulate a failing
scene — proving per-scene fail-soft (Golden Rule 4). Successful renders go through the
real Pillow renderer (dispatcher degrades non-Pillow scenes to a slide).
"""
from PIL import Image
import pytest

import pipeline
from pipeline import make_assets
from models import Renderer, Scene, Storyboard, VisualCue

W, H = 1920, 1080


def _storyboard(*pairs):
    scenes = [
        Scene(segment_id=i, visual_cue=cue, renderer=ren, duration_seconds=5)
        for i, (cue, ren) in enumerate(pairs, start=1)
    ]
    return Storyboard(topic="T", scenes=scenes)


def _counting(real):
    calls = {"n": 0}

    def wrap(scene, out_path):
        calls["n"] += 1
        return real(scene, out_path)

    return wrap, calls


def test_make_assets_renders_every_scene_to_png(tmp_path, monkeypatch):
    wrap, calls = _counting(pipeline.render_scene)
    monkeypatch.setattr(pipeline, "render_scene", wrap)
    # GRAPHVIZ scene has no implementation yet -> the dispatcher degrades to Pillow.
    sb = _storyboard((VisualCue.TITLE_CARD, Renderer.PILLOW),
                     (VisualCue.DS_GRAPH, Renderer.GRAPHVIZ))

    out = tmp_path / "assets"
    assets = make_assets(sb, out)

    assert set(assets) == {1, 2}
    assert assets[1].exists() and assets[2].exists()
    with Image.open(assets[1]) as img:
        assert img.size == (W, H)
    assert calls["n"] == 2


def test_make_assets_is_idempotent_and_cached(tmp_path, monkeypatch):
    wrap, calls = _counting(pipeline.render_scene)
    monkeypatch.setattr(pipeline, "render_scene", wrap)
    sb = _storyboard((VisualCue.TITLE_CARD, Renderer.PILLOW),
                     (VisualCue.WHAT_IS, Renderer.PILLOW),
                     (VisualCue.CLOSING, Renderer.PILLOW))

    out = tmp_path / "assets"
    first = make_assets(sb, out)
    assert calls["n"] == 3

    second = make_assets(sb, out)
    assert calls["n"] == 3          # cached — zero new renders
    assert first == second          # same segment_id -> path mapping


def test_make_assets_force_rerenders(tmp_path, monkeypatch):
    wrap, calls = _counting(pipeline.render_scene)
    monkeypatch.setattr(pipeline, "render_scene", wrap)
    sb = _storyboard((VisualCue.TITLE_CARD, Renderer.PILLOW))

    out = tmp_path / "assets"
    make_assets(sb, out)
    assert calls["n"] == 1

    make_assets(sb, out, force=True)
    assert calls["n"] == 2          # force bypasses the cache


def test_make_assets_fail_soft_skips_one_scene(tmp_path, monkeypatch):
    real = pipeline.render_scene

    def wrap(scene, out_path):
        if scene.segment_id == 2:
            raise RuntimeError("renderer + Pillow fallback both failed")
        return real(scene, out_path)

    monkeypatch.setattr(pipeline, "render_scene", wrap)
    sb = _storyboard((VisualCue.TITLE_CARD, Renderer.PILLOW),
                     (VisualCue.DS_GRAPH, Renderer.GRAPHVIZ),
                     (VisualCue.CLOSING, Renderer.PILLOW))

    out = tmp_path / "assets"
    assets = make_assets(sb, out)

    assert set(assets) == {1, 3}    # scene 2 skipped, the run survives (never fail-whole)
    assert assets[1].exists() and assets[3].exists()
