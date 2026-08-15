"""VisuLearn CLI entry point. See AGENT.md §6.

    python visulearn.py script "Red-Black Tree" --out output/scripts/rbt.json
    python visulearn.py video  "Red-Black Tree" --out output/videos/rbt.mp4
    python visulearn.py batch  topics.txt --out_dir output/videos
"""
from __future__ import annotations

import argparse
import re
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pipeline  # noqa: E402


def _slug(topic: str) -> str:
    """Filesystem-safe, lowercased, no dots."""
    safe = re.sub(r"[^a-z0-9._-]+", "_", topic.strip().lower()) or "topic"
    return re.sub(r"[._]+", "_", safe).strip("_")


def _run_batch(topics_path: Path, out_dir: Path, *, dry_run: bool) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [ln.strip() for ln in topics_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        print(f"no topics found in {topics_path}")
        return 1
    failed = 0
    for i, topic in enumerate(lines, 1):
        out_path = out_dir / f"{_slug(topic)}.mp4"
        try:
            pipeline.make_video(topic, out_path, dry_run=dry_run)
            print(f"[{i}] {'DRYRUN' if dry_run else 'ok':8} {topic!r} -> {out_path}")
        except Exception:  # noqa: BLE001 - per-topic failure, keep the batch going
            failed += 1
            traceback.print_exc()
    return 1 if failed else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="visulearn")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_script = sub.add_parser("script", help="topic -> validated script JSON")
    p_script.add_argument("topic")
    p_script.add_argument("--out", type=Path, required=True)

    p_video = sub.add_parser("video", help="topic -> assembled MP4")
    p_video.add_argument("topic")
    p_video.add_argument("--out", type=Path, required=True)
    p_video.add_argument("--workspace", type=Path, default=None,
                         help="workspace dir (default: .<stem>_workspace)")
    p_video.add_argument("--dry-run", action="store_true",
                         help="validate topic only, don't generate")
    # DI seams — pass-through for future driver injection
    p_video.add_argument("--tts", default="default", choices=("default",),
                         help="TTS driver (default: kokoro)")
    p_video.add_argument("--encoder", default="default", choices=("default",),
                         help="encoder driver (default: moviepy)")

    p_batch = sub.add_parser("batch", help="topics file (one per line) -> one MP4 each")
    p_batch.add_argument("topics", type=Path)
    p_batch.add_argument("--out_dir", type=Path, required=True)
    p_batch.add_argument("--dry-run", action="store_true",
                         help="validate topics only without generating")

    args = parser.parse_args(argv)

    if args.cmd == "script":
        if args.out.suffix != ".json":
            parser.error(f"--out must end with .json, got {args.out.suffix}")
        path, storyboard = pipeline.make_script(args.topic, args.out)
        print(f"script -> {path} ({len(storyboard.scenes)} scenes)")
        return 0

    if args.cmd == "video":
        if args.out.suffix != ".mp4":
            parser.error(f"--out must end with .mp4, got {args.out.suffix}")
        path = pipeline.make_video(
            args.topic, args.out,
            workspace_dir=args.workspace,
            dry_run=args.dry_run,
        )
        print(f"[{'DRYRUN' if args.dry_run else 'ok':8}] video -> {path}")
        return 0

    if args.cmd == "batch":
        return _run_batch(args.topics, args.out_dir, dry_run=args.dry_run)

    return 2   # unreachable (subparsers required=True handles missing/unknown)


if __name__ == "__main__":
    raise SystemExit(main())
