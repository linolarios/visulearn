"""Assembly — sync scenes + audio into final MP4. See AGENT.md §3/§4.

MoviePy 1.x is pinned (`moviepy<2`). Because MoviePy needs ffmpeg and is NOT installed
in CI, the compositing work sits behind an injectable ENCODER seam — `encoder(plan,
out_path) -> Path` — so the orchestration (audio-duration reads, per-scene duration =
max(audio_duration, min_display_time), ordering, preflight validation, crossfade) is
fully tested offline with a stub. The real MoviePy driver is a follow-up; encoder=None
fails fast with a readable error rather than silently producing nothing.

Preflight (deliberate, not fail-soft): a scene missing its asset/audio (or with
unreadable audio) is a hard, legible error — at this stage there is no 'blank' to
degrade to (that happens in the Asset Factory), and silently dropping a scene would
desync the audio timeline.
"""
from __future__ import annotations

import logging
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from models import MIN_DISPLAY_TIME_SECONDS, Scene, Storyboard

log = logging.getLogger(__name__)

CROSSFADE_S = 0.3


@dataclass(frozen=True)
class ClipSpec:
    """One composited scene the encoder must produce."""
    segment_id: int
    asset: Path
    audio: Path
    duration_seconds: float
    crossfade_s: float


def _audio_duration(path: Path) -> float:
    """Seconds of audio from a WAV (stdlib). Raises wave.Error/OSError if unreadable."""
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate()) if w.getframerate() > 0 else 0.0


def _build_plan(
        storyboard: Storyboard,
        scene_assets: dict[int, Path],
        audio: dict[int, Path],
        min_display_time: int,
) -> list[ClipSpec]:
    """Ordered (by segment_id) clip plan; preflights missing/unreadable pieces."""
    missing: list[int] = []
    plan: list[ClipSpec] = []
    for scene in sorted(storyboard.scenes, key=lambda s: s.segment_id):
        asset = scene_assets.get(scene.segment_id)
        wav = audio.get(scene.segment_id)
        if (asset is None or wav is None
                or not Path(asset).is_file() or not Path(wav).is_file()):
            missing.append(scene.segment_id)
            continue
        try:
            audio_s = _audio_duration(wav)
        except (OSError, wave.Error):
            missing.append(scene.segment_id)
            continue
        duration = max(audio_s, max(scene.duration_seconds, min_display_time))
        plan.append(ClipSpec(scene.segment_id, Path(asset), Path(wav), duration, CROSSFADE_S))
    if missing:
        raise ValueError(f"assembly is missing asset/audio for segment(s): {sorted(missing)}")
    if not plan:
        raise ValueError("assembly has no scenes to render")
    return plan


def assemble(
        storyboard: Storyboard,
        scene_assets: dict[int, Path],
        audio: dict[int, Path],
        out_path: Path,
        *,
        encoder: Optional[Callable[[list[ClipSpec], Path], Path]] = None,
        min_display_time: int = MIN_DISPLAY_TIME_SECONDS,
        crossfade_s: float = CROSSFADE_S,
) -> Path:
    """Compose scenes+audio into an MP4 via the injected encoder seam.

    Scene duration = max(audio_duration, min_display_time); clips are ordered by
    segment_id; 0.3s crossfades; audio synced per segment (AGENT.md §4).
    """
    if encoder is None:
        raise NotImplementedError(
            "MoviePy encoder driver not wired yet — inject "
            "`encoder(plan: list[ClipSpec], out_path) -> Path`. "
            "(moviepy<2 is pinned in requirements.txt but needs ffmpeg; not run in CI.)"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plan = _build_plan(storyboard, scene_assets, audio, min_display_time)
    return encoder(plan, out_path)
