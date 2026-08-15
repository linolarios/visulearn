"""Pillow renderer — deterministic 1920x1080 PNG per scene (AGENT.md §5).

v1 draws a cue label (segment id + visual_cue + renderer + duration) because a Scene
carries no narration text; text-bearing slides need Segment{id} -> narration plumbing
and are out of scope here. Deterministic by construction (no randomness) so re-runs are
byte-identical — required for idempotent caching (AGENT.md §5 / Golden Rule 5).
On failure the Asset Factory dispatcher falls back to this renderer (Golden Rule 4).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from models import Scene

WIDTH, HEIGHT = 1920, 1080
BG = (18, 18, 28)          # dark slate background
FG = (240, 240, 245)       # off-white text
ACCENT = (110, 200, 255)   # top accent bar


def render(scene: Scene, out_path: Path) -> Path:
    """Draw a deterministic cue slide and save it to out_path as a PNG."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, WIDTH, 12], fill=ACCENT)
    draw.text((64, 56), f"Scene {scene.segment_id} - {scene.visual_cue.value}", fill=FG)
    draw.text((64, 128), f"renderer: {scene.renderer.value}  duration: {scene.duration_seconds}s", fill=FG)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")
    return out_path
