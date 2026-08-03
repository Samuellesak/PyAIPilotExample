#!/usr/bin/env python3
"""Create a video from saved camera frames in a log/session directory.

Examples:
    python make_video_from_logs.py logs/20260803_104227
    python make_video_from_logs.py logs --fps 20
"""

import argparse
import re
import sys
from pathlib import Path

import cv2


def _frame_sort_key(path: Path):
    match = re.search(r"(\d+)", path.stem)
    if match:
        return (0, int(match.group(1)))
    return (1, path.name.lower())


def _collect_session_dirs(path: Path):
    if (path / "frames").is_dir():
        return [path]

    sessions = []
    for child in sorted(path.iterdir()):
        if child.is_dir() and (child / "frames").is_dir():
            sessions.append(child)
    return sessions


def make_video(session_dir: Path, output_path: Path, fps: float):
    frames_dir = session_dir / "frames"
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"No frames directory found in {session_dir}")

    frame_paths = sorted(frames_dir.glob("*.jpg"), key=_frame_sort_key)
    frame_paths += sorted(frames_dir.glob("*.jpeg"), key=_frame_sort_key)
    frame_paths += sorted(frames_dir.glob("*.png"), key=_frame_sort_key)
    frame_paths = sorted(set(frame_paths), key=_frame_sort_key)

    if not frame_paths:
        raise FileNotFoundError(f"No frame images found in {frames_dir}")

    first_img = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first_img is None:
        raise RuntimeError(f"Could not read first frame: {frame_paths[0]}")

    height, width = first_img.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {output_path}")

    written = 0
    for frame_path in frame_paths:
        img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] skipping unreadable frame: {frame_path}", file=sys.stderr)
            continue
        writer.write(img)
        written += 1

    writer.release()
    print(f"Wrote {written} frames to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create a video from saved frames in a log/session directory")
    parser.add_argument("path", nargs="?", default="logs", help="Session directory or logs root")
    parser.add_argument("-o", "--output", default=None, help="Optional output path for a single session")
    parser.add_argument("--fps", type=float, default=20.0, help="Video frame rate (default: 20)")
    args = parser.parse_args()

    root = Path(args.path).resolve()
    if not root.exists():
        parser.error(f"Path does not exist: {root}")

    sessions = _collect_session_dirs(root)
    if not sessions:
        parser.error(f"No session directories with a frames/ folder found under {root}")

    if args.output is not None and len(sessions) > 1:
        parser.error("--output can only be used when the input points to a single session directory")

    for session_dir in sessions:
        if args.output is not None:
            out_path = Path(args.output).resolve()
        else:
            out_path = session_dir / f"{session_dir.name}.mp4"
        make_video(session_dir, out_path, args.fps)


if __name__ == "__main__":
    main()
