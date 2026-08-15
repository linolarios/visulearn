"""Offline CLI tests — every pipeline driver call is stubbed (AGENT.md §7).

Covers the visulearn.main dispatch: script/video/batch invoke pipeline.make_* with the
right arguments, batch isolates per-topic failures (returning nonzero), dry-run mode,
extension validation, and an unknown command exits. Nothing renders, encodes, or talks
to an LLM.
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

def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        visulearn.main(["nope"])
