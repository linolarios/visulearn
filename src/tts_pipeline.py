"""TTS — one WAV per segment via Kokoro-82M. See AGENT.md §4.

Kokoro outputs 24 kHz mono (verify for your build). Normalize technical terms before
synthesis; the Script Engine gate already rejects raw 'O(n)'-style narration.
"""
from __future__ import annotations

from pathlib import Path

from models import Script

KOKORO_SAMPLE_RATE = 24_000


def synthesize(script: Script, out_dir: Path, voice: str = "af_bella", speed: float = 1.0) -> list[Path]:
    """Return one WAV path per segment, in order."""
    raise NotImplementedError("Implement per AGENT.md §4 (TTS).")
