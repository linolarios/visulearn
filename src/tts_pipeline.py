"""TTS pipeline — one WAV per segment. See AGENT.md §4.

Kokoro-82M outputs 24 kHz mono (KOKORO_SAMPLE_RATE); every produced WAV is validated as
such before it is kept. The synthesis call is an injectable seam
`tts(text, *, voice, speed) -> wav_bytes` so the orchestrator is testable offline (AGENT.md
§7). The DEFAULT driver, _kokoro_driver, lazily imports kokoro-onnx and fails fast with a
readable error when it (or the model files) are missing — nothing is produced silently.

Golden Rule 5 (idempotent): an existing non-empty, valid .wav is reused unless force=True.
Golden Rule 4 (fail-soft): a segment whose synthesis OR validation fails is logged + skipped,
never a whole-run crash.
"""
from __future__ import annotations

import array
import io
import logging
import os
import wave
from pathlib import Path
from typing import Callable, Optional

from models import Script

log = logging.getLogger(__name__)

KOKORO_SAMPLE_RATE = 24_000
KOKORO_CHANNELS = 1
KOKORO_MODEL_FILENAME = "kokoro-v1.0.onnx"
KOKORO_VOICES_FILENAME = "voices-v1.0.bin"

REPO_ROOT = Path(__file__).resolve().parent.parent


def _valid_wav(data: bytes) -> bool:
    """A produced .wav is only accepted if it really is 24 kHz mono (AGENT.md §4)."""
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            return (w.getframerate() == KOKORO_SAMPLE_RATE
                    and w.getnchannels() == KOKORO_CHANNELS
                    and w.getnframes() > 0)
    except (wave.Error, EOFError, ValueError):
        return False


def _samples_to_wav(samples, sample_rate: int, *,
                    target_rate: int = KOKORO_SAMPLE_RATE,
                    channels: int = KOKORO_CHANNELS) -> bytes:
    """Signed-16 PCM samples -> 24 kHz mono WAV bytes (pure, stdlib).

    Rejects a wrong sample rate rather than silently mislabelling (the 24 kHz contract).
    """
    if sample_rate != target_rate:
        raise ValueError(f"audio {sample_rate} Hz != expected {target_rate} Hz")
    if len(samples) == 0:
        raise ValueError("empty audio samples")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(array.array("h", (int(s) for s in samples)).tobytes())
    return buf.getvalue()


def resolve_kokoro_model_dir() -> Path:
    """Model dir: $VISULEARN_KOKORO_DIR override, else default assets/kokoro."""
    env = os.environ.get("VISULEARN_KOKORO_DIR")
    if env:
        return Path(env)
    return REPO_ROOT / "assets" / "kokoro"


def _require_models(model_dir: Path) -> tuple[Path, Path]:
    """Return (model.onnx, voices.bin); raise a readable error if any piece is missing."""
    model_path = model_dir / KOKORO_MODEL_FILENAME
    voices_path = model_dir / KOKORO_VOICES_FILENAME
    if not model_dir.is_dir():
        raise RuntimeError(
            f"Kokoro model directory not found: {model_dir} — place the Kokoro-82M model "
            "there or set VISULEARN_KOKORO_DIR"
        )
    missing = [p.name for p in (model_path, voices_path) if not p.is_file()]
    if missing:
        raise RuntimeError(
            f"Kokoro model files missing in {model_dir}: {sorted(missing)} — download "
            f"{KOKORO_MODEL_FILENAME} and {KOKORO_VOICES_FILENAME}"
        )
    return model_path, voices_path


def _import_kokoro():
    try:
        from kokoro_onnx import Kokoro
    except ImportError as exc:
        raise RuntimeError(
            "Kokoro not installed — 'pip install kokoro-onnx' (and download the model / "
            f"voices files; see AGENT.md §1 for the {KOKORO_SAMPLE_RATE} Hz contract)"
        ) from exc
    return Kokoro


def _kokoro_driver(voice: str = "af_bella", speed: float = 1.0,
                   model_dir: Optional[Path] = None) -> Callable[[str], bytes]:
    """Real synthesis driver: kokoro-onnx -> valid 24 kHz mono WAV bytes."""
    _Kokoro = _import_kokoro()
    model_path, voices_path = _require_models(model_dir or resolve_kokoro_model_dir())
    kokoro = _Kokoro(str(model_path), str(voices_path))

    def synth(text, *, voice=voice, speed=speed):
        samples, sample_rate = kokoro.create(text, voice=voice, speed=speed)
        pcm = [max(-32768, min(32767, int(round(float(s) * 32767)))) for s in samples]
        return _samples_to_wav(pcm, sample_rate)

    return synth


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
    synth = tts if tts is not None else _kokoro_driver(voice, speed)

    out_dir.mkdir(parents=True, exist_ok=True)
    wavs: dict[int, Path] = {}
    for seg in script.segments:
        path = out_dir / f"{seg.id}.wav"
        if not force and path.exists() and path.stat().st_size > 0:
            wavs[seg.id] = path
            continue
        try:
            data = synth(seg.narration, voice=voice, speed=speed)
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
