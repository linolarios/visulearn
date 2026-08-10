"""Assembly — sync scenes + audio into final MP4 via MoviePy. See AGENT.md §3/§4.

MoviePy is PINNED to <2 in requirements.txt: this module uses the 1.x API
(moviepy.editor, crossfadein/crossfadeout). If you migrate to 2.x, update BOTH the pin
and the calls (from moviepy import ...; .with_duration(); vfx.CrossFadeIn).
"""
from __future__ import annotations

from pathlib import Path

from models import Storyboard

CROSSFADE_S = 0.3


def assemble(storyboard: Storyboard, scene_assets: dict[int, Path],
             audio: dict[int, Path], out_path: Path) -> Path:
    """H.264 MP4, 1920x1080, 30fps, AAC. Scene duration = max(audio, min_display)."""
    raise NotImplementedError("Implement per AGENT.md §4 (Assembly).")
