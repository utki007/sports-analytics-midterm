"""Command-line entry point.

    python -m backend.cli demo                 # synthetic end-to-end run
    python -m backend.cli process input.mp4    # real inference
    python -m backend.cli process input.mp4 --stride 2 --max-seconds 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import OUTPUT_DIR, ProcessConfig
from .pipeline import Pipeline


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="AI Basketball Analytics pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="Run the synthetic labeled demo end-to-end")
    d.add_argument("--out", type=Path, default=OUTPUT_DIR / "demo")

    p = sub.add_parser("process", help="Process a real video")
    p.add_argument("video", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--max-seconds", type=float, default=0.0)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--no-ocr", action="store_true")

    args = ap.parse_args(argv)

    _last = [0.0]

    def prog(f):
        if f - _last[0] >= 0.05 or f >= 1.0:
            _last[0] = f
            sys.stdout.write(f"\r  progress: {f*100:5.1f}%")
            sys.stdout.flush()

    if args.cmd == "demo":
        cfg = ProcessConfig()
        out_dir = args.out
        pipe = Pipeline(cfg)
        res = pipe.run(Path("unused"), out_dir, mode="demo", progress=prog)
        print()
    else:
        if not args.video.exists():
            print(f"Error: video not found: {args.video}", file=sys.stderr)
            return 1
        cfg = ProcessConfig(model_name=args.model, frame_stride=args.stride,
                            imgsz=args.imgsz, conf=args.conf,
                            max_seconds=args.max_seconds,
                            render_video=not args.no_render,
                            enable_ocr=not args.no_ocr)
        out_dir = args.out or (OUTPUT_DIR / args.video.stem)
        pipe = Pipeline(cfg)
        res = pipe.run(args.video, out_dir, mode="real", progress=prog)
        print()

    m = res.meta
    print(f"Mode: {m['mode']}  device: {m['device']}  elapsed: {m.get('elapsed_s')}s")
    print(f"Processed {m['processed_frames']} frames @ {m['fps']:.1f} fps")
    print(f"Players: {len(res.players)}  Shots: {len(res.shots)}  "
          f"Rim: {'yes' if m['rim_detected'] else 'no'}")
    for s in res.shots:
        ph = s.physics
        extra = ""
        if ph and ph.calibrated:
            extra = f" | launch {ph.launch_angle_deg}° speed {ph.release_speed_mps}m/s [{ph.feedback_code}]"
        print(f"  Shot {s.id}: t={s.t_release:.2f}s {s.outcome} "
              f"(team {s.team}, conf {s.outcome_confidence:.2f}){extra}")
    print(f"Outputs in: {out_dir}")
    print(f"  annotated video: {out_dir}/annotated_video.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
