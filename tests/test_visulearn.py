"""Offline CLI tests — every pipeline driver call is stubbed (AGENT.md §7).

Covers the visulearn.main dispatch: script/video/batch invoke pipeline.make_* with the
right arguments, batch isolates per-topic failures (returning nonzero), dry-run mode,
extension validation, _slug edge cases, parent-dir creation, batch summaries, and an
unknown command exits. Nothing renders, encodes, or talks to an LLM.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pipeline  # noqa: E402
import visulearn  # noqa: E402


def _stub_make_video(monkeypatch, calls):
    def fake(topic, out_path, **kw):
        calls.append((topic, out_path, dict(kw)))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"x")
        return out_path

    monkeypatch.setattr(pipeline, "make_video", fake)
    return fake


def test_video_subcommand_dispatches(tmp_path, monkeypatch):
    calls = []
    _stub_make_video(monkeypatch, calls)

    rc = visulearn.main(["video", "Red-Black Tree", "--out", str(tmp_path / "rbt.mp4")])

    assert rc == 0
    topic, out_path, kw = calls[0]
    assert topic == "Red-Black Tree"
    assert out_path == tmp_path / "rbt.mp4"
    assert kw.get("dry_run") is False
    assert (tmp_path / "rbt.mp4").exists()


def test_video_dry_run_flag(tmp_path, monkeypatch):
    calls = []
    _stub_make_video(monkeypatch, calls)

    rc = visulearn.main(["video", "Stack", "--out", str(tmp_path / "s.mp4"), "--dry-run"])

    assert rc == 0
    _, _, kw = calls[0]
    assert kw.get("dry_run") is True


def test_video_rejects_non_mp4_extension(tmp_path):
    with pytest.raises(SystemExit):
        visulearn.main(["video", "Stack", "--out", str(tmp_path / "s.webm")])


def test_video_accepts_case_insensitive_extension(tmp_path, monkeypatch):
    calls = []

    def fake(*a, **kw):
        calls.append(a)
        a[1].parent.mkdir(parents=True, exist_ok=True)
        a[1].write_bytes(b"x")
        return a[1]

    monkeypatch.setattr(pipeline, "make_video", fake)

    rc = visulearn.main(["video", "Stack", "--out", str(tmp_path / "s.MP4")])

    assert rc == 0
    assert (tmp_path / "s.MP4").exists()


def test_script_subcommand_dispatches(tmp_path, monkeypatch):
    class _SB:
        scenes = [1, 2]

    def fake(topic, out_path):
        out_path.write_text("{}")
        return out_path, _SB()

    monkeypatch.setattr(pipeline, "make_script", fake)
    out = tmp_path / "rbt.json"

    rc = visulearn.main(["script", "Red-Black Tree", "--out", str(out)])

    assert rc == 0 and out.exists()


def test_script_rejects_non_json(tmp_path):
    with pytest.raises(SystemExit):
        visulearn.main(["script", "Stack", "--out", str(tmp_path / "s.mp4")])


def test_script_accepts_case_insensitive_json(tmp_path, monkeypatch):
    class _SB:
        scenes = [1, 2]

    def fake(t, p):
        p.write_text("{}")
        return p, _SB()

    monkeypatch.setattr(pipeline, "make_script", fake)

    rc = visulearn.main(["script", "Stack", "--out", str(tmp_path / "s.JSON")])

    assert rc == 0


def test_batch_subcommand_processes_each_line(tmp_path, monkeypatch):
    (tmp_path / "topics.txt").write_text("Stack\nMerge Sort\n")
    calls = []
    _stub_make_video(monkeypatch, calls)
    out_dir = tmp_path / "videos"

    rc = visulearn.main(["batch", str(tmp_path / "topics.txt"), "--out_dir", str(out_dir)])

    assert rc == 0
    assert [c[0] for c in calls] == ["Stack", "Merge Sort"]


def test_batch_reports_failure_and_continues(tmp_path, monkeypatch):
    (tmp_path / "topics.txt").write_text("Stack\nMerge Sort\n")

    def fake(topic, out_path, **kw):
        if topic == "Merge Sort":
            raise RuntimeError("boom")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"x")
        return out_path

    monkeypatch.setattr(pipeline, "make_video", fake)
    out_dir = tmp_path / "videos"

    rc = visulearn.main(["batch", str(tmp_path / "topics.txt"), "--out_dir", str(out_dir)])

    assert rc == 1
    assert (out_dir / "stack.mp4").exists()   # good topic still ran (slug lowercases)


def test_batch_skips_existing_output(tmp_path, monkeypatch):
    (tmp_path / "topics.txt").write_text("Stack\n")
    calls = []

    def fake(topic, out_path, **kw):
        calls.append(topic)
        return out_path

    monkeypatch.setattr(pipeline, "make_video", fake)
    out_dir = tmp_path / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stack.mp4").write_bytes(b"existing")

    rc = visulearn.main(["batch", str(tmp_path / "topics.txt"), "--out_dir", str(out_dir)])

    assert rc == 0
    assert calls == []                          # existing output -> skipped, driver not called
    assert (out_dir / "stack.mp4").read_bytes() == b"existing"   # untouched


def test_batch_summary_lists_skipped_and_failed(tmp_path, monkeypatch, capsys):
    (tmp_path / "topics.txt").write_text("A\nB\nC\n")
    calls = []

    def fake(topic, out_path, **kw):
        calls.append(topic)
        if topic == "B":
            raise RuntimeError("boom")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"x")
        return out_path

    monkeypatch.setattr(pipeline, "make_video", fake)
    out_dir = tmp_path / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "c.mp4").write_bytes(b"existing")      # -> C is skipped

    rc = visulearn.main(["batch", str(tmp_path / "topics.txt"), "--out_dir", str(out_dir)])

    out = capsys.readouterr().out
    assert rc == 1
    assert "3 total" in out and "1 skipped" in out and "1 failed" in out
    assert calls == ["A", "B"]                        # C never reached the driver


def test_script_warns_on_empty_storyboard(tmp_path, monkeypatch, capsys):
    class _SB:
        scenes = []

    monkeypatch.setattr(pipeline, "make_script", lambda t, p: (p.write_text("{}") or p, _SB()))
    out = tmp_path / "s.json"

    rc = visulearn.main(["script", "Stack", "--out", str(out)])

    assert rc == 0
    assert "no scenes" in capsys.readouterr().out


def test_video_creates_parent_dir(tmp_path, monkeypatch):
    calls = []

    def fake(*a, **kw):
        calls.append(a)
        a[1].parent.mkdir(parents=True, exist_ok=True)
        return a[1]

    monkeypatch.setattr(pipeline, "make_video", fake)
    deep = tmp_path / "a" / "b" / "rbt.mp4"

    visulearn.main(["video", "RBT", "--out", str(deep)])

    assert deep.parent.exists()


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        visulearn.main(["nope"])


def test_slug_empty_string_returns_topic():
    assert visulearn._slug("") == "topic"
    assert visulearn._slug("   ") == "topic"
    assert visulearn._slug("___") == "topic"
    assert visulearn._slug("...") == "topic"


def test_video_has_no_tts_or_encoder_flags():
    # The dead DI flags were removed; driver injection lives at the pipeline seam.
    with pytest.raises(SystemExit):
        visulearn.main(["video", "Stack", "--out", "x.mp4", "--tts", "default"])
