"""Offline tests for tts_pipeline.synthesize — stdlib-only, no Kokoro/network (AGENT.md §7).

The synthesis seam is injected with a stub that produces REAL WAV bytes via the stdlib
`wave` module, so the "valid output, not just call-count" contract (AGENT.md §4 — 24 kHz
mono, non-empty) is genuinely asserted. Covers the three audio-admission cases:
  (a) valid WAV at the WRONG sample rate -> rejected/skipped (regression guard);
  (b) valid WAV header with ZERO frames -> rejected/skipped (regression guard; §4 non-empty);
  (c) INVALID audio in a PARTIAL batch -> only that segment dropped, good ones survive.
Plus: one valid WAV/segment, validity, voice/speed forwarding (explicit + defaults),
per-segment narration (matched by content), idempotent cache (GR5), force re-render,
fail-soft isolation (GR4), invalid-audio all-rejected, and the fail-fast no-seam guard.
"""
import io
import wave
from collections import Counter
from pathlib import Path

import pytest

import tts_pipeline
from models import Script
from tts_pipeline import synthesize

FIXTURE = Path(__file__).parent / "fixtures" / "rbt_script.json"


def _script() -> Script:
    return Script.model_validate_json(FIXTURE.read_text())


def _wav_bytes(rate: int = 24_000, duration_s: float = 0.5) -> bytes:
    """A structurally valid WAV; duration_s=0 yields a valid header with zero frames."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * duration_s))
    return buf.getvalue()


class _FakeTTS:
    """Injected synthesis seam. Optionally raises OR returns garbage for given narrations."""

    def __init__(self, fail_narrations=(), garbage_narrations=()):
        self.fail_narrations = set(fail_narrations)
        self.garbage_narrations = set(garbage_narrations)
        self.calls = []

    def __call__(self, text, *, voice, speed):
        self.calls.append((text, voice, speed))
        if text in self.fail_narrations:
            raise RuntimeError("synth exploded")
        if text in self.garbage_narrations:
            return b"this is not a wav"
        return _wav_bytes()


# --- happy path + validity ------------------------------------------------- #

def test_one_valid_wav_per_segment(tmp_path):
    script = _script()
    fake = _FakeTTS()
    out = tmp_path / "audio"

    wavs = synthesize(script, out, tts=fake)

    assert set(wavs) == {s.id for s in script.segments}
    for path in wavs.values():
        assert path.exists() and path.stat().st_size > 0
    assert len(fake.calls) == len(script.segments)


def test_produced_wav_is_valid_24k_mono(tmp_path):
    wavs = synthesize(_script(), tmp_path / "audio", tts=_FakeTTS())
    with wave.open(str(next(iter(wavs.values()))), "rb") as w:
        assert w.getframerate() == 24_000
        assert w.getnchannels() == 1


# --- narration + voice/speed forwarding ------------------------------------- #

def test_tts_receives_narration_and_voice_speed(tmp_path):
    script = _script()
    fake = _FakeTTS()
    synthesize(script, tmp_path / "audio", tts=fake, voice="af_bella", speed=0.9)

    text, voice, speed = fake.calls[0]
    assert text == script.segments[0].narration
    assert voice == "af_bella"
    assert speed == 0.9


def test_tts_receives_each_segments_narration(tmp_path):
    # Match by content (Counter) — order is not part of the contract.
    fake = _FakeTTS()
    synthesize(_script(), tmp_path / "audio", tts=fake)
    assert Counter(c[0] for c in fake.calls) == Counter(s.narration for s in _script().segments)


def test_default_voice_speed_flow_through(tmp_path):
    fake = _FakeTTS()
    synthesize(_script(), tmp_path / "audio", tts=fake)          # no voice/speed kwargs
    assert all(c[1] == "af_bella" and c[2] == 1.0 for c in fake.calls)


# --- idempotent cache + force (GR5) ---------------------------------------- #

def test_idempotent_cache_no_rerender(tmp_path):
    fake = _FakeTTS()
    script = _script()
    out = tmp_path / "audio"

    first = synthesize(script, out, tts=fake)
    assert len(fake.calls) == len(script.segments)

    second = synthesize(script, out, tts=fake)
    assert len(fake.calls) == len(script.segments)   # cached — zero new synth calls
    assert first == second


def test_force_rerenders(tmp_path):
    fake = _FakeTTS()
    script = _script()
    out = tmp_path / "audio"

    synthesize(script, out, tts=fake)
    assert len(fake.calls) == len(script.segments)

    synthesize(script, out, tts=fake, force=True)
    assert len(fake.calls) == 2 * len(script.segments)


# --- fail-soft isolation (GR4) --------------------------------------------- #

def test_fail_soft_skips_only_bad_segment(tmp_path):
    script = _script()
    bad = script.segments[2].narration
    fake = _FakeTTS(fail_narrations={bad})

    wavs = synthesize(script, tmp_path / "audio", tts=fake)

    expected = {s.id for s in script.segments} - {script.segments[2].id}
    assert set(wavs) == expected       # bad segment skipped, run survives
    assert all(p.exists() for p in wavs.values())


# --- (c) partial batch: invalid bytes in a mixed batch ---------------------- #

def test_invalid_audio_partial_batch_keeps_good_segments(tmp_path):
    script = _script()
    bad = script.segments[2].narration
    fake = _FakeTTS(garbage_narrations={bad})

    wavs = synthesize(script, tmp_path / "audio", tts=fake)

    expected = {s.id for s in script.segments} - {script.segments[2].id}
    assert set(wavs) == expected       # ONLY the garbage segment dropped
    assert all(p.exists() for p in wavs.values())
    assert not (tmp_path / "audio" / f"{script.segments[2].id}.wav").exists()


def test_invalid_audio_all_rejected_no_files(tmp_path):
    class _BadAudio:
        def __call__(self, text, *, voice, speed):
            return b"this is not a wav"

    out = tmp_path / "audio"
    wavs = synthesize(_script(), out, tts=_BadAudio())
    assert wavs == {}
    assert not list(out.glob("*.wav"))   # nothing silently kept


# --- (a) wrong sample rate -------------------------------------------------- #

def test_wrong_sample_rate_wav_rejected(tmp_path):
    class _OffRateTTS:
        def __call__(self, text, *, voice, speed):
            return _wav_bytes(rate=22_050, duration_s=0.5)   # valid WAV, wrong rate

    out = tmp_path / "audio"
    wavs = synthesize(_script(), out, tts=_OffRateTTS())

    # A valid-but-wrong-rate WAV must NOT silently land (24 kHz contract, AGENT.md §4).
    assert wavs == {}
    assert not list(out.glob("*.wav"))


# --- (b) zero-duration audio ----------------------------------------------- #

def test_zero_duration_wav_rejected(tmp_path):
    class _EmptyTTS:
        def __call__(self, text, *, voice, speed):
            return _wav_bytes(rate=24_000, duration_s=0)     # valid header, zero frames

    out = tmp_path / "audio"
    wavs = synthesize(_script(), out, tts=_EmptyTTS())

    # §4: audio must be non-empty (duration > 0); a zero-frame WAV is NOT accepted.
    assert wavs == {}
    assert not list(out.glob("*.wav"))


# --- config guard ----------------------------------------------------------- #

def test_default_kokoro_driver_readable_when_missing(tmp_path, monkeypatch):
    def boom():
        raise RuntimeError("Kokoro not installed — 'pip install kokoro-onnx' ...")

    monkeypatch.setattr(tts_pipeline, "_import_kokoro", boom)

    with pytest.raises(RuntimeError, match="Kokoro not installed"):
        synthesize(_script(), tmp_path / "audio")   # tts=None -> default kokoro driver


# --- pure kokoro-driver tests ------------------------------------------------- #

def _kokoro_present():
    try:
        import kokoro_onnx  # noqa: F401
        tts_pipeline._require_models(tts_pipeline.resolve_kokoro_model_dir())
        return True
    except Exception:
        return False


def test_samples_to_wav_produces_valid_24k_mono():
    data = tts_pipeline._samples_to_wav([0, 32767, -32768], 24_000)
    with wave.open(io.BytesIO(data), "rb") as w:
        assert w.getframerate() == 24_000
        assert w.getnchannels() == 1
        assert w.getnframes() == 3
    assert tts_pipeline._valid_wav(data)


def test_samples_to_wav_rejects_wrong_rate():
    with pytest.raises(ValueError, match="24000"):
        tts_pipeline._samples_to_wav([0, 1], 22_050)


def test_samples_to_wav_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        tts_pipeline._samples_to_wav([], 24_000)


def test_resolve_kokoro_model_dir_default(monkeypatch):
    monkeypatch.delenv("VISULEARN_KOKORO_DIR", raising=False)
    assert tts_pipeline.resolve_kokoro_model_dir() == tts_pipeline.REPO_ROOT / "assets" / "kokoro"


def test_resolve_kokoro_model_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("VISULEARN_KOKORO_DIR", str(tmp_path / "models"))
    assert tts_pipeline.resolve_kokoro_model_dir() == tmp_path / "models"


def test_require_models_missing_reports_readable(tmp_path):
    with pytest.raises(RuntimeError, match="model files missing"):
        tts_pipeline._require_models(tmp_path)          # empty dir


def test_require_models_missing_dir():
    with pytest.raises(RuntimeError, match="not found"):
        tts_pipeline._require_models(Path("/no/such/kokoro"))


@pytest.mark.skipif(
    not _kokoro_present(),
    reason="requires kokoro-onnx + model files (real synthesis; not in CI)",
)
def test_kokoro_driver_synthesizes_valid_wav():
    synth = tts_pipeline._kokoro_driver()
    data = synth("Hello world.")
    assert tts_pipeline._valid_wav(data)

