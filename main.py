#!/usr/bin/env python3
"""CLI for basketball auto shot-event scanning."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

from detector import scan_video, write_tracks_sample_csv
from shots import detect_shots, write_shots_csv
from teams import cluster_teams

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def find_data_video(data_dir: Path = DATA_DIR) -> Path | None:
    """Return the first video file found under data/, sorted by name."""
    if not data_dir.is_dir():
        return None
    videos = sorted(
        p
        for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    return videos[0] if videos else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a basketball video with YOLO, estimate rim/teams, "
            "and write automatic shot-event rows to shots.csv."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Track ball/players and write shots.csv")
    scan.add_argument(
        "video",
        type=Path,
        nargs="?",
        default=None,
        help="Path to the input video (default: first video found in data/)",
    )
    scan.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ultralytics YOLO weights (default: yolov8n.pt)",
    )
    scan.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )
    scan.add_argument(
        "--sample-interval",
        type=float,
        default=0.25,
        help="Seconds between sampled frames (default: 0.25)",
    )
    scan.add_argument(
        "--merge-gap",
        type=float,
        default=1.0,
        help="Max gap (seconds) to merge nearby shot intervals (default: 1.0)",
    )
    scan.add_argument(
        "--pre-seconds",
        type=float,
        default=1.0,
        help="Seconds before release included in each shot (default: 1.0)",
    )
    scan.add_argument(
        "--post-seconds",
        type=float,
        default=1.5,
        help="Seconds after release included in each shot (default: 1.5)",
    )
    scan.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Folder for shots.csv and tracks_sample.csv (default: output)",
    )
    return parser


def cmd_scan(args: argparse.Namespace) -> int:
    video: Path | None = args.video
    if video is None:
        video = find_data_video()
        if video is None:
            print(
                f"Error: no video argument given and no video found in {DATA_DIR}",
                file=sys.stderr,
            )
            return 1
        print(f"Using video from data/: {video}")
    elif not video.exists() or not video.is_file():
        print(f"Error: video not found or not a file: {video}", file=sys.stderr)
        return 1

    # Never overwrite the input video; write only under output-dir.
    output_dir: Path = args.output_dir
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"Error: could not create output directory {output_dir}: {exc}", file=sys.stderr)
        return 1

    # Outputs go only under output_dir; the input video is never overwritten.
    print(f"Scanning {video} with model={args.model} conf={args.conf} ...")
    try:
        result = scan_video(
            video_path=video,
            model_name=args.model,
            conf=args.conf,
            sample_interval=args.sample_interval,
        )
    except Exception as exc:
        print(f"Error while scanning video: {exc}", file=sys.stderr)
        return 1

    team_assignment = cluster_teams(result.jersey_samples)
    if not team_assignment:
        print(
            "Warning: jersey clustering failed or had too few samples; "
            "team labels will be 'unknown'."
        )

    shots = detect_shots(
        observations=result.observations,
        rim_center=result.rim_center,
        team_assignment=team_assignment,
        frame_width=result.frame_width,
        frame_height=result.frame_height,
        pre_seconds=args.pre_seconds,
        post_seconds=args.post_seconds,
        merge_gap=args.merge_gap,
        duration_seconds=result.duration_seconds,
    )

    shots_path = output_dir / "shots.csv"
    tracks_path = output_dir / "tracks_sample.csv"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_shots_csv(shots_path, shots)
        for w in caught:
            print(f"Warning: {w.message}")

    write_tracks_sample_csv(tracks_path, result.observations, team_assignment)

    rim_msg = (
        f"({result.rim_center[0]:.1f}, {result.rim_center[1]:.1f})"
        if result.rim_center
        else "unavailable"
    )
    print(f"Rim estimate: {rim_msg}")
    print(f"Teams assigned to {len(team_assignment)} person tracks.")
    print(f"Detected {len(shots)} shot event(s).")
    print(f"Wrote {shots_path}")
    print(f"Wrote debug tracks to {tracks_path}")
    print(
        "Note: YOLO + heuristics are approximate. Stock models may miss "
        "basketballs/rims; review shots.csv before relying on it."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "scan":
        return cmd_scan(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
