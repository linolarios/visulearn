"""VisuLearn CLI entry point. See AGENT.md §6.

    python visulearn.py script "Red-Black Tree" --out output/scripts/rbt.json
    python visulearn.py video  output/scripts/rbt.json --out output/videos/rbt.mp4
    python visulearn.py --batch topics.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(prog="visulearn")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_script = sub.add_parser("script", help="topic -> validated script JSON")
    p_script.add_argument("topic")
    p_script.add_argument("--out", type=Path, required=True)

    p_video = sub.add_parser("video", help="script JSON -> MP4")
    p_video.add_argument("script_json", type=Path)
    p_video.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    raise SystemExit(f"'{args.cmd}' not implemented yet — see AGENT.md §4/§6.")


if __name__ == "__main__":
    main()
