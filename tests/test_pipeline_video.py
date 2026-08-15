"""Offline end-to-end tests for pipeline.make_video (AGENT.md §4/§6/§7).

Every driver is injected: a stub Script-Engine provider (returns the RBT fixture), a stub
tts seam producing real WAV bytes (stdlib wave), and a recording encoder seam. The real
Pillow renderers run, so PNGs are genuinely produced; no network, no MoviePy/ffmpeg.
"""
import io
import wave
from pathlib import Path

import pytest

import pipeline
from models import Script

FIXTURE = Path(__file__).parent / "fixtures" / "rbt_script.json"
VALID = FIXTURE.read_text()


class _FakeScriptProvider:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def adapt_schema(self, schema):
        return schema

    def complete(self, messages, schema):
        idx = self.calls
        self.calls += 1
        if idx >= len(self.outputs):
            raise AssertionError("provider called more times than outputs provided")
        return self.outputs[idx]


def _wav_bytes(rate: int = 24_000, duration_s: float = 0.5) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * duration_s))
    return buf.getvalue()


def _counting_tts():
    calls = [0]

    def tts(text, *, voice, speed):
        calls[0] += 1
        return _wav_bytes()

    return tts, calls


class _Recorder:
    def __init__(self):
        self.plan = None

    def __call__(self, plan, out_path):
        self.plan = list(plan)
        out_path.write_bytes(b"MP4")
        return out_path


def test_make_video_end_to_end(tmp_path):
    out = tmp_path / "rbt.mp4"
    provider = _FakeScriptProvider([VALID])
    enc = _Recorder()

    path = pipeline.make_video("Red-Black Tree", out, script_provider=provider,
                               tts=lambda text, **kw: _wav_bytes(), encoder=enc)

    n = len(Script.model_validate_json(VALID).segments)
    ws = tmp_path / ".rbt_workspace"
    assert path == out and out.exists()
    assert (ws / "script.json").exists()
    assert provider.calls == 1
    assert enc.plan is not None and len(enc.plan) == n
    assert all(c.duration_seconds > 0 for c in enc.plan)
    assert len(list((ws / "scenes").glob("*.png"))) == n
    assert len(list((ws / "audio").glob("*.wav"))) == n


def test_make_video_reuse_is_idempotent(tmp_path):
    provider = _FakeScriptProvider([VALID])
    tts, calls = _counting_tts()
    n = len(Script.model_validate_json(VALID).segments)
    out = tmp_path / "rbt.mp4"

    pipeline.make_video("Red-Black Tree", out, script_provider=provider,
                        tts=tts, encoder=_Recorder())
    assert provider.calls == 1
    assert calls[0] == n

    pipeline.make_video("Red-Black Tree", out, script_provider=provider,
                        tts=tts, encoder=_Recorder())
    assert provider.calls == 1               # script resumed from cache
    assert calls[0] == n                     # audio reused


def test_make_video_dry_run_does_not_call_tts_or_encoder(tmp_path):
    provider = _FakeScriptProvider([VALID])
    tts_called = False

    def tts(text, **kw):
        nonlocal tts_called
        tts_called = True

    path = pipeline.make_video("Red-Black Tree", tmp_path / "rbt.mp4",
                               script_provider=provider, tts=tts,
                               encoder=_Recorder(), dry_run=True)

    assert provider.calls == 1        # resolve + script generated
    assert not tts_called             # tts NOT called — dry_run
    assert not path.exists()          # nothing written
    assert not (tmp_path / ".rbt_workspace" / "script.json").exists()


def test_make_video_missing_tts_fails_fast(tmp_path, monkeypatch):
    import tts_pipeline

    def boom():
        raise RuntimeError("Kokoro not installed — 'pip install kokoro-onnx' ...")

    monkeypatch.setattr(tts_pipeline, "_import_kokoro", boom)
    provider = _FakeScriptProvider([VALID])

    with pytest.raises(RuntimeError, match="Kokoro not installed"):
        pipeline.make_video("Red-Black Tree", tmp_path / "rbt.mp4",
                            script_provider=provider, tts=None,
                            encoder=_Recorder())


def test_make_video_injected_fact_sheet_is_used(tmp_path):
    provider = _FakeScriptProvider([VALID])
    seen = {}

    def facts(topic):
        seen["topic"] = topic
        return {"definition": "ok"}

    pipeline.make_video("red-black tree", tmp_path / "rbt.mp4",
                        script_provider=provider, fact_sheet=facts,
                        tts=lambda text, **kw: _wav_bytes(),
                        encoder=_Recorder())

    assert seen["topic"] == "Red-Black Tree"
