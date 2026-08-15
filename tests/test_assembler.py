"""Offline tests for assembler.assemble — stdlib-only, no MoviePy/ffmpeg (AGENT.md §7).

The MoviePy encoder is injected as a recording seam, so the real logic — reading real
WAV durations via the stdlib `wave` module, computing scene duration = max(audio, min_display),
ordering clips, and preflighting missing/unreadable pieces — is genuinely tested without
needing ffmpeg (which CI does not install).
"""
import wave
from pathlib import Path

import pytest

import assembler
from models import MIN_DISPLAY_TIME_SECONDS, Renderer, Scene, Storyboard, VisualCue
from assembler import ClipSpec, assemble

RATE = 24_000


def _wav(path: Path, seconds: float, rate: int = RATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return path


def _storyboard(durations=(5, 5), ids=(1, 2)):
    scenes = [
        Scene(segment_id=i, visual_cue=VisualCue.WHAT_IS, renderer=Renderer.PILLOW,
              duration_seconds=d)
        for i, d in zip(ids, durations)
    ]
    return Storyboard(topic="T", scenes=scenes)


class _Recorder:
    """Recording encoder seam: captures the plan, writes a placeholder file."""
    def __init__(self):
        self.plan = None
        self.out = None

    def __call__(self, plan, out_path):
        self.plan = list(plan)
        self.out = out_path
        out_path.write_bytes(b"FAKEMP4")
        return out_path


def _assets_audio(tmp_path, n=2):
    assets = {i: (tmp_path / f"{i}.png") for i in range(1, n + 1)}
    for p in assets.values():
        p.write_bytes(b"PNG")
    audio = {i: _wav(tmp_path / f"{i}.wav", 1.0) for i in range(1, n + 1)}
    return assets, audio


def test_audio_duration_is_read_from_wav(tmp_path):
    w = _wav(tmp_path / "a.wav", 2.0)
    assert assembler._audio_duration(w) == pytest.approx(2.0, abs=0.01)


def test_scene_duration_is_max_of_audio_and_display(tmp_path):
    # scene 1: audio 2.0s vs display 5 -> 5 ; scene 2: audio 8.0s vs display 5 -> 8
    sb = _storyboard(durations=(5, 5))
    assets, audio = _assets_audio(tmp_path, n=2)
    audio[1] = _wav(tmp_path / "1.wav", 2.0)
    audio[2] = _wav(tmp_path / "2.wav", 8.0)

    rec = _Recorder()
    assemble(sb, assets, audio, tmp_path / "out.mp4", encoder=rec)

    d = {c.segment_id: c.duration_seconds for c in rec.plan}
    assert d[1] == pytest.approx(5.0)     # audio shorter than display floor -> display wins
    assert d[2] == pytest.approx(8.0)     # audio longer -> audio wins (AGENT.md §4)


def test_encoder_receives_ordered_plan_with_fields(tmp_path):
    sb = _storyboard(ids=(2, 1))          # deliberately unordered
    assets, audio = _assets_audio(tmp_path, n=2)

    rec = _Recorder()
    out = tmp_path / "out.mp4"
    ret = assemble(sb, assets, audio, out, encoder=rec)

    assert ret == out and out.exists()                # returns/creates out_path
    assert [c.segment_id for c in rec.plan] == [1, 2]  # ordered by segment_id
    c = rec.plan[0]
    assert isinstance(c, ClipSpec)
    assert c.asset == assets[1] and c.audio == audio[1]
    assert c.crossfade_s == 0.3                        # 0.3s crossfades (AGENT.md §4)


def test_missing_artifact_is_a_legible_preflight_error(tmp_path):
    sb = _storyboard(durations=(5, 5))
    assets, audio = _assets_audio(tmp_path, n=2)
    del audio[2]                                       # scene 2 has no audio

    rec = _Recorder()
    with pytest.raises(ValueError, match=r"segment\(s\): \[2\]"):
        assemble(sb, assets, audio, tmp_path / "out.mp4", encoder=rec)
    assert rec.plan is None                            # fail fast — encoder never ran


def test_unreadable_audio_is_preflight_error(tmp_path):
    sb = _storyboard(durations=(5, 5))
    assets, _ = _assets_audio(tmp_path, n=2)
    audio = {1: tmp_path / "1.wav", 2: _wav(tmp_path / "2.wav", 1.0)}
    audio[1].write_bytes(b"not a wav")                 # present but unreadable

    with pytest.raises(ValueError, match="segment"):
        assemble(sb, assets, audio, tmp_path / "out.mp4", encoder=_Recorder())


def test_no_encoder_fails_fast_legibly(tmp_path):
    assets, audio = _assets_audio(tmp_path, n=1)
    with pytest.raises(NotImplementedError, match="encoder"):
        assemble(_storyboard(ids=(1,)), assets, audio, tmp_path / "out.mp4")


def test_min_display_time_floor_respected():
    assert MIN_DISPLAY_TIME_SECONDS >= 3                # contract: no scene shorter than this
