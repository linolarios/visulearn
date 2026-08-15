"""Offline tests for assembler.assemble + build_render_plan — stdlib-only, no MoviePy.

The MoviePy encoder is injected for the orchestration tests; build_render_plan (the pure
AGENT.md §4 render contract) is unit-tested directly; the real moviepy driver has a
deterministic guard test (readable error when absent) and a skipif live test (only when
moviepy<2 + ffmpeg are present, matching the smoke-test pattern — not in CI).
"""
import shutil
import wave
from pathlib import Path

import pytest

import assembler
from models import MIN_DISPLAY_TIME_SECONDS, Renderer, Scene, Storyboard, VisualCue
from assembler import ClipSpec, assemble

RATE = 24_000


def _moviepy_installed():
    try:
        import moviepy  # noqa: F401
        return True
    except ImportError:
        return False


def _wav(path: Path, seconds: float, rate: int = RATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return path


def _png(path: Path, color=(200, 30, 30), size=(1920, 1080)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.new("RGB", size, color).save(path)
    return path


def _storyboard(durations=(5, 5), ids=(1, 2)):
    scenes = [
        Scene(segment_id=i, visual_cue=VisualCue.WHAT_IS, renderer=Renderer.PILLOW,
              duration_seconds=d)
        for i, d in zip(ids, durations)
    ]
    return Storyboard(topic="T", scenes=scenes)


class _Recorder:
    def __init__(self):
        self.plan = None
        self.out = None

    def __call__(self, plan, out_path):
        self.plan = list(plan)
        self.out = out_path
        out_path.write_bytes(b"FAKEMP4")
        return out_path


def _assets_audio(tmp_path, n=2):
    assets = {i: _png(tmp_path / f"{i}.png") for i in range(1, n + 1)}
    audio = {i: _wav(tmp_path / f"{i}.wav", 1.0) for i in range(1, n + 1)}
    return assets, audio


# --- orchestration ---------------------------------------------------------- #

def test_audio_duration_is_read_from_wav(tmp_path):
    assert assembler._audio_duration(_wav(tmp_path / "a.wav", 2.0)) == pytest.approx(2.0, abs=0.01)


def test_scene_duration_is_max_of_audio_and_display(tmp_path):
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
    sb = _storyboard(ids=(2, 1))
    assets, audio = _assets_audio(tmp_path, n=2)

    rec = _Recorder()
    out = tmp_path / "out.mp4"
    assert assemble(sb, assets, audio, out, encoder=rec) == out and out.exists()
    assert [c.segment_id for c in rec.plan] == [1, 2]
    c = rec.plan[0]
    assert isinstance(c, ClipSpec) and c.asset == assets[1] and c.audio == audio[1]
    assert c.crossfade_s == 0.3


def test_missing_artifact_is_a_legible_preflight_error(tmp_path):
    sb = _storyboard(durations=(5, 5))
    assets, audio = _assets_audio(tmp_path, n=2)
    del audio[2]

    rec = _Recorder()
    with pytest.raises(ValueError, match=r"segment\(s\): \[2\]"):
        assemble(sb, assets, audio, tmp_path / "out.mp4", encoder=rec)
    assert rec.plan is None                            # fail fast — encoder never ran


def test_unreadable_audio_is_preflight_error(tmp_path):
    sb = _storyboard(durations=(5, 5))
    assets, _ = _assets_audio(tmp_path, n=2)
    audio = {1: tmp_path / "1.wav", 2: _wav(tmp_path / "2.wav", 1.0)}
    audio[1].write_bytes(b"not a wav")

    with pytest.raises(ValueError, match="segment"):
        assemble(sb, assets, audio, tmp_path / "out.mp4", encoder=_Recorder())


def test_min_display_time_floor_respected():
    assert MIN_DISPLAY_TIME_SECONDS >= 3                # contract: no scene shorter than this


# --- build_render_plan: the pure §4 render contract -------------------------- #

def _plan(tmp_path):
    return [
        ClipSpec(1, tmp_path / "1.png", tmp_path / "1.wav", 5.0, 0.3),
        ClipSpec(2, tmp_path / "2.png", tmp_path / "2.wav", 8.0, 0.3),
    ]


def test_build_render_plan_encodes_1080p_contract(tmp_path):
    rp = assembler.build_render_plan(_plan(tmp_path))
    # Contract values asserted from the spec (AGENT.md §4), not recomputed:
    assert rp["fps"] == 30
    assert rp["size"] == (1920, 1080)
    assert rp["video_codec"] == "libx264"               # H.264
    assert rp["audio_codec"] == "aac"                   # AAC
    assert [c["duration"] for c in rp["clips"]] == [5.0, 8.0]
    assert [c["crossfade"] for c in rp["clips"]] == [0.3, 0.3]
    assert rp["clips"][0]["asset"] == tmp_path / "1.png"
    assert rp["clips"][0]["audio"] == tmp_path / "1.wav"


# --- default moviepy driver: guard + live ------------------------------------ #

def test_default_moviepy_driver_readable_when_missing(tmp_path, monkeypatch):
    # Deterministic: simulate MoviePy being absent so the guard is testable offline.
    def boom():
        raise RuntimeError("MoviePy not installed — run 'pip install \"moviepy<2\"' (requires ffmpeg)")

    monkeypatch.setattr(assembler, "_import_moviepy", boom)
    sb = _storyboard(ids=(1,))
    assets, audio = _assets_audio(tmp_path, n=1)

    with pytest.raises(RuntimeError, match="MoviePy not installed"):
        assemble(sb, assets, audio, tmp_path / "out.mp4")   # encoder=None -> default driver


@pytest.mark.skipif(
    not (shutil.which("ffmpeg") and _moviepy_installed()),
    reason="requires moviepy<2 + ffmpeg (real encoding; not in CI)",
)
def test_moviepy_default_driver_produces_valid_mp4(tmp_path):
    sb = _storyboard(ids=(1, 2), durations=(1, 1))
    assets = {1: _png(tmp_path / "1.png", (200, 30, 30)),
              2: _png(tmp_path / "2.png", (30, 90, 200))}
    audio = {1: _wav(tmp_path / "1.wav", 1.0), 2: _wav(tmp_path / "2.wav", 1.0)}

    out = assemble(sb, assets, audio, tmp_path / "out.mp4")   # default MoviePy driver

    assert out.exists()
    assert out.stat().st_size > 1024                          # real encoded content
    assert b"ftyp" in out.read_bytes()[:64]                   # valid MP4 container, not a stub

def test_empty_storyboard_produces_legible_error(tmp_path):
    # model_construct bypasses Pydantic's min_length=1 guard (which is for normal
    # construction), so we can test the assembler's own guard for an empty storyboard.
    sb = Storyboard.model_construct(topic="Empty", scenes=[])
    rec = _Recorder()

    with pytest.raises(ValueError, match="no scenes"):
        assemble(sb, {}, {}, tmp_path / "out.mp4", encoder=rec)
    assert rec.plan is None                              # fail fast — encoder never ran