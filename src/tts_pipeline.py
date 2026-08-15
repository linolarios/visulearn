"""TTS pipeline — one WAV per segment. See AGENT.md §4.

Kokoro-82M outputs 24 kHz mono (KOKORO_SAMPLE_RATE); every produced WAV is validated as
such before it is kept. The synthesis call is an injected seam
`tts(text, *, voice, speed) -> wav_bytes` so the orchestrator is testable offline with a
stub (AGENT.md §7); the real Kokoro driver is a follow-up, so tts=None fails fast with a
readable error rather than silently producing nothing.

Golden Rule 5 (idempotent): an existing non-empty, valid .wav is reused unless force=True.
Golden Rule 4 (fail-soft): a segment whose synthesis OR validation fails is logged + skipped,
never a whole-run crash.
"""
from __future__ import annotations

import io
import logging
import wave
from pathlib import Path
from typing import Callable, Optional

from models import Script

log = logging.getLogger(__name__)

KOKORO_SAMPLE_RATE = 24_000
KOKORO_CHANNELS = 1


def _valid_wav(data: bytes) -> bool:
    """A produced .wav is only accepted if it really is 24 kHz mono (AGENT.md §4)."""
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            return (w.getframerate() == KOKORO_SAMPLE_RATE
                    and w.getnchannels() == KOKORO_CHANNELS
                    and w.getnframes() > 0)  # §4: non-empty — zero/zero-duration WAV rejected
    except (wave.Error, EOFError, ValueError):
        return False


def synthesize(
        script: Script,
        out_dir: Path,
        *,
        tts: Optional[Callable[..., bytes]] = None,
        voice: str = "af_bella",
        speed: float = 1.0,
        force: bool = False,
) -> dict[int, Path]:
    """One valid WAV per segment at out_dir/<segment_id>.wav; cache + fail-soft (GR 4/5)."""
    if tts is None:
        raise NotImplementedError(
            "TTS driver not wired yet — inject `tts(text, *, voice, speed) -> wav_bytes` "
            "(the Kokoro driver is a follow-up; nothing is produced silently)."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    wavs: dict[int, Path] = {}
    for seg in script.segments:
        path = out_dir / f"{seg.id}.wav"
        if not force and path.exists() and path.stat().st_size > 0:
            wavs[seg.id] = path
            continue
        try:
            data = tts(seg.narration, voice=voice, speed=speed)
            if not _valid_wav(data):
                raise ValueError(
                    f"TTS returned invalid audio (expected {KOKORO_SAMPLE_RATE} Hz mono, "
                    "non-empty)"
                )
            path.write_bytes(data)
            wavs[seg.id] = path
        except Exception as exc:  # noqa: BLE001 - fail-soft per segment, never fail-whole
            log.warning("TTS failed for segment %s; skipping: %s", seg.id, exc)
    return wavs
