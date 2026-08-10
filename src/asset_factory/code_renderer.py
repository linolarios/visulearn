"""Pygments renderer. Produces a 1920x1080 PNG (or 1080p MP4 for Manim) per scene.
See AGENT.md §5. On failure, the Asset Factory falls back to a Pillow slide (Golden Rule 4).
"""
from __future__ import annotations
from pathlib import Path
from models import Scene


def render(scene: Scene, out_path: Path) -> Path:
    raise NotImplementedError("Implement Pygments rendering per AGENT.md §5.")
