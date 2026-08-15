"""Assembly — sync scenes + audio into final MP4. See AGENT.md §3/§4.

MoviePy 1.x is pinned (`moviepy<2`) and is the DEFAULT encoder; because it needs ffmpeg
and is not installed in CI, the compositing path is split so the spec logic is tested
without it:

  * build_render_plan(...) is PURE and MoviePy-agnostic — it owns the AGENT.md §4 render
    contract (30fps, 1920x1080, H.264/libx264, AAC) and is fully unit-tested offline.
  * _moviepy_encode(...) is the thin MoviePy glue over that plan; it fails fast with a
    readable error if MoviePy is missing, and is only exercised by a skipif-guarded live
    test when moviepy<2 + ffmpeg are actually installed.

`encoder` remains an injectable seam for tests/stubs. Preflight (deliberate): a scene
missing its asset/audio (or unreadable audio) is a legible hard error — at this stage
there is no 'blank' to degrade to and silently dropping a scene would desync the audio
timeline.
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
WIDTH, HEIGHT = 1920, 1080
DEFAULT_FPS = 30
DEFAULT_VIDEO_CODEC = "libx264"   # H.264 (AGENT.md §4)
DEFAULT_AUDIO_CODEC = "aac"       # AAC (AGENT.md §4)


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


def build_render_plan(
        plan: list[ClipSpec],
        *,
        fps: int = DEFAULT_FPS,
        size: tuple[int, int] = (WIDTH, HEIGHT),
        codec: str = DEFAULT_VIDEO_CODEC,
        audio_codec: str = DEFAULT_AUDIO_CODEC,
) -> dict:
    """Pure, MoviePy-agnostic render contract (AGENT.md §4): 1080p/30fps/H.264/AAC."""
    return {
        "clips": [
            {
                "asset": spec.asset,
                "audio": spec.audio,
                "duration": spec.duration_seconds,
                "crossfade": spec.crossfade_s,
            }
            for spec in plan
        ],
        "fps": fps,
        "size": tuple(size),
        "video_codec": codec,
        "audio_codec": audio_codec,
    }


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


def _import_moviepy():
    """Lazy, 1.x import. Raises a readable error if MoviePy (or ffmpeg deps) are missing."""
    try:
        from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips  # 1.x
    except ImportError as exc:
        raise RuntimeError(
            "MoviePy not installed — run 'pip install \"moviepy<2\"' (requires ffmpeg) "
            "for real encoding"
        ) from exc
    return ImageClip, AudioFileClip, concatenate_videoclips


def _moviepy_encode(plan: list[ClipSpec], out_path: Path, **render_kwargs) -> Path:
    """MoviePy 1.x glue over build_render_plan. Uses crossfades (AGENT.md §4)."""
    ImageClip, AudioFileClip, concatenate = _import_moviepy()
    import numpy as np
    from PIL import Image as PilImage

    rp = build_render_plan(plan, **render_kwargs)
    clips: list = []
    n = len(rp["clips"])
    for i, c in enumerate(rp["clips"]):
        with PilImage.open(str(c["asset"])) as im:
            arr = np.array(im)
        clip = ImageClip(arr).set_duration(c["duration"]).set_fps(rp["fps"])
        clip = clip.set_audio(AudioFileClip(str(c["audio"])))
        if n > 1:
            if i == 0:
                clip = clip.crossfadeout(c["crossfade"])
            elif i == n - 1:
                clip = clip.crossfadein(c["crossfade"])
            else:
                clip = clip.crossfadein(c["crossfade"]).crossfadeout(c["crossfade"])
        clips.append(clip)

    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(
        str(out_path), fps=rp["fps"], codec=rp["video_codec"], audio_codec=rp["audio_codec"],
    )
    return out_path


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
    """Compose scenes+audio into an MP4. Default encoder = MoviePy driver.

    Scene duration = max(audio_duration, min_display_time); clips ordered by segment_id;
    0.3s crossfades; audio synced per segment (AGENT.md §4).
    """
    if encoder is None:
        encoder = _moviepy_encode
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plan = _build_plan(storyboard, scene_assets, audio, min_display_time)
    return encoder(plan, out_path)
