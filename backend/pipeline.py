"""End-to-end processing pipeline.

Single pass over the video performs detection + tracking + jersey sampling +
rim accumulation + camera motion, caching the raw per-frame detections. A second
pass renders the overlay. Downstream analysis (teams, ball timeline, shots,
physics) runs on the cached detections so iteration is fast.

Two clearly separated front-ends feed the SAME downstream analysis:
  * real  — YOLO + color-ball + rim estimation on actual footage.
  * demo  — synthetic ground-truth detections (labeled synthetic throughout).
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import __version__
from .ball import build_ball_timeline
from .config import OUTPUT_DIR, ProcessConfig
from .court import RimEstimator, rim_from_ball_apexes
from .detection import Detector, RawBall, RawPerson, detect_color_balls
from .demo import generate_demo
from .jersey import JerseyAggregator, backend_available, read_number
from .physics import analyze_shot
from .renderer import render as render_video
from .schemas import (AnalysisResult, BallDet, Frame, PersonDet, PlayerTrack,
                      RimDet, Shot)
from .shots import associate_shooter, detect_shots
from .teams import TeamClassifier, jersey_features
from .tracking import CameraMotionEstimator, TrackStats


def _video_meta(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return {"fps": fps, "width": w, "height": h, "n_frames": n,
            "duration": (n / fps) if fps else 0.0}


class Pipeline:
    def __init__(self, cfg: ProcessConfig):
        self.cfg = cfg

    # -- single-pass detection over the source video ------------------------
    def _scan_real(self, video: Path, meta: dict, progress) -> Tuple[
            Dict[int, List[RawPerson]], Dict[int, List[RawBall]],
            Dict[int, float], List[int], Optional[RimDet],
            TeamClassifier, JerseyAggregator, Dict[int, TrackStats]]:
        cap = cv2.VideoCapture(str(video))
        detector = Detector(self.cfg.model_name, self.cfg.device, self.cfg.conf,
                            self.cfg.iou, self.cfg.imgsz, self.cfg.tracker)
        rim_est = RimEstimator(meta["width"], meta["height"])
        cam = CameraMotionEstimator()
        teams = TeamClassifier(self.cfg.team_min_samples)
        jerseys = JerseyAggregator()
        ocr_on = self.cfg.enable_ocr and backend_available() is not None
        stats: Dict[int, TrackStats] = {}

        persons_by_frame: Dict[int, List[RawPerson]] = {}
        balls_by_frame: Dict[int, List[RawBall]] = {}
        frame_times: Dict[int, float] = {}
        indices: List[int] = []
        ball_pts: List[Tuple[float, float]] = []

        fi = 0
        fps = meta["fps"]
        max_frames = meta["n_frames"]
        if self.cfg.max_seconds > 0:
            max_frames = min(max_frames, int(self.cfg.max_seconds * fps))
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and fi >= max_frames):
                break
            if fi % self.cfg.frame_stride != 0:
                fi += 1
                continue
            t = fi / fps
            frame_times[fi] = t
            indices.append(fi)
            cam.update(frame)
            rim_est.update(frame)

            persons, yolo_balls = detector.detect_track(frame)
            person_boxes = [p.bbox for p in persons]
            balls = list(yolo_balls)
            if self.cfg.use_color_ball_fallback:
                balls += detect_color_balls(frame, person_boxes)
            persons_by_frame[fi] = persons
            balls_by_frame[fi] = balls
            for b in balls:
                ball_pts.append((b.cx, b.cy))

            for p in persons:
                if p.track_id is None:
                    continue
                st = stats.setdefault(p.track_id, TrackStats(p.track_id))
                cx = (p.bbox[0] + p.bbox[2]) / 2
                cy = (p.bbox[1] + p.bbox[3]) / 2
                st.add(cx, cy, p.bbox[3] - p.bbox[1], p.conf, fi)
                teams.add(p.track_id, jersey_features(frame, p.bbox))
                if ocr_on:
                    x1, y1, x2, y2 = [int(v) for v in p.bbox]
                    bh = y2 - y1
                    crop = frame[y1 + int(bh * 0.2):y1 + int(bh * 0.55),
                                 max(0, x1):x2]
                    jerseys.add(p.track_id, read_number(crop) if crop.size else None)
            fi += 1
            if progress and max_frames:
                progress(0.1 + 0.5 * fi / max_frames)
        cap.release()

        for tid, st in stats.items():
            teams.set_track_stats(tid, st.total_motion())

        rim = rim_est.estimate() or rim_from_ball_apexes(ball_pts, meta["height"])
        return (persons_by_frame, balls_by_frame, frame_times, indices, rim,
                teams, jerseys, stats)

    def _scan_demo(self, video: Path, gt, progress) -> Tuple:
        """Use synthetic ground truth but still sample real pixels for teams."""
        cap = cv2.VideoCapture(str(video))
        teams = TeamClassifier(self.cfg.team_min_samples)
        jerseys = JerseyAggregator()
        stats: Dict[int, TrackStats] = {}
        persons_by_frame: Dict[int, List[RawPerson]] = {}
        balls_by_frame: Dict[int, List[RawBall]] = {}
        frame_times: Dict[int, float] = {}
        indices: List[int] = []
        fps = gt.fps
        fi = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = fi / fps
            frame_times[fi] = t
            indices.append(fi)
            persons = []
            for (tid, bbox, _bgr) in gt.persons.get(fi, []):
                persons.append(RawPerson(track_id=tid, bbox=tuple(map(float, bbox)), conf=0.99))
            persons_by_frame[fi] = persons
            balls = []
            if fi in gt.ball:
                bx, by, br = gt.ball[fi]
                balls.append(RawBall(cx=bx, cy=by, r=br, conf=0.95, source="demo"))
            balls_by_frame[fi] = balls
            for p in persons:
                st = stats.setdefault(p.track_id, TrackStats(p.track_id))
                cx = (p.bbox[0] + p.bbox[2]) / 2
                cy = (p.bbox[1] + p.bbox[3]) / 2
                st.add(cx, cy, p.bbox[3] - p.bbox[1], p.conf, fi)
                teams.add(p.track_id, jersey_features(frame, p.bbox))
            fi += 1
            if progress:
                progress(0.1 + 0.5 * fi / max(1, gt.n_frames))
        cap.release()
        for tid, st in stats.items():
            teams.set_track_stats(tid, st.total_motion())
        rim = RimDet(cx=gt.rim[0], cy=gt.rim[1], w=gt.rim[2], h=gt.rim[3], conf=0.9)
        return (persons_by_frame, balls_by_frame, frame_times, indices, rim,
                teams, jerseys, stats)

    # -- main entry ---------------------------------------------------------
    def run(self, video: Path, job_dir: Path, mode: str = "real",
            progress: Optional[Callable[[float], None]] = None) -> AnalysisResult:
        job_dir.mkdir(parents=True, exist_ok=True)
        t_start = time.time()
        gt = None
        if mode == "demo":
            demo_video = job_dir / "demo_input.mp4"
            gt = generate_demo(demo_video)
            video = demo_video
        meta = _video_meta(video)
        if progress:
            progress(0.05)

        if mode == "demo":
            scan = self._scan_demo(video, gt, progress)
        else:
            scan = self._scan_real(video, meta, progress)
        (persons_by_frame, balls_by_frame, frame_times, indices, rim,
         teams, jerseys, stats) = scan

        # Teams + jersey resolution
        team_map, team_palette = teams.assign()
        jersey_map = jerseys.resolve()

        # Ball timeline
        dt = self.cfg.frame_stride / meta["fps"]
        rim_xy = (rim.cx, rim.cy) if rim else None
        timeline = build_ball_timeline(balls_by_frame, frame_times, rim_xy, indices, dt)
        timeline_by_frame = {p.frame: p for p in timeline}

        # Build Frame objects
        frames: List[Frame] = []
        for fi in indices:
            persons_out = []
            for p in persons_by_frame.get(fi, []):
                if p.track_id is None:
                    continue
                info = team_map.get(p.track_id, {})
                jn = jersey_map.get(p.track_id, (None, 0.0))[0]
                persons_out.append(PersonDet(
                    track_id=p.track_id, bbox=p.bbox, conf=p.conf,
                    team=info.get("team", "unknown"),
                    role=info.get("role", "player"),
                    jersey_number=jn))
            bp = timeline_by_frame.get(fi)
            ball = BallDet(bp.cx, bp.cy, bp.r, bp.conf, bp.observed) if bp else None
            frames.append(Frame(index=fi, t=frame_times[fi],
                                persons=persons_out, ball=ball, rim=rim))

        # Player tracks. "significant" = persistent track likely on-court,
        # separating real players from brief fragments / spectators.
        n_proc = max(1, len(indices))
        sig_gate = max(12, int(0.12 * n_proc))
        players: List[PlayerTrack] = []
        for tid, st in sorted(stats.items()):
            info = team_map.get(tid, {})
            jn, jconf = jersey_map.get(tid, (None, 0.0))
            role = info.get("role", "player")
            players.append(PlayerTrack(
                track_id=tid, team=info.get("team", "unknown"),
                team_confidence=info.get("team_confidence", 0.0),
                role=role,
                jersey_number=jn, jersey_confidence=jconf,
                color_rgb=tuple(info.get("color_rgb", [170, 170, 170])),
                n_frames=st.n, first_t=frame_times.get(st.first_frame, 0.0),
                last_t=frame_times.get(st.last_frame, 0.0),
                significant=(role in ("player", "referee") and st.n >= sig_gate)))
        players.sort(key=lambda p: (not p.significant, -p.n_frames))

        # Shots + physics
        if progress:
            progress(0.65)
        persons_for_assoc = {
            fi: [{"track_id": p.track_id,
                  "cx": (p.bbox[0] + p.bbox[2]) / 2,
                  "cy": (p.bbox[1] + p.bbox[3]) / 2}
                 for p in ps if p.track_id is not None]
            for fi, ps in persons_by_frame.items()}

        candidates = detect_shots(timeline, rim_xy, meta["width"], meta["height"])
        shots: List[Shot] = []
        rim_width = rim.w if rim else None
        for i, c in enumerate(candidates, 1):
            shooter = associate_shooter(persons_for_assoc, frame_times,
                                        c.t_release, c.release_xy)
            info = team_map.get(shooter, {}) if shooter else {}
            jn = jersey_map.get(shooter, (None, 0.0))[0] if shooter else None
            # Ball arc points for physics
            arc = [timeline[k] for k in c.arc_frames if 0 <= k < len(timeline)
                   and timeline[k].observed]
            if len(arc) < 3:
                arc = [timeline[k] for k in c.arc_frames if 0 <= k < len(timeline)]
            ball_r = float(np.median([p.r for p in arc])) if arc else None
            phys = analyze_shot(
                [p.t for p in arc], [p.cx for p in arc], [p.cy for p in arc],
                rim_xy=rim_xy, rim_width_px=rim_width, ball_radius_px=ball_r,
                image_height_px=meta["height"]) if len(arc) >= 3 else None
            shots.append(Shot(
                id=i, t_release=c.t_release, t_start=c.t_start, t_end=c.t_end,
                shooter_track_id=shooter, team=info.get("team", "unknown"),
                jersey_number=jn, outcome=c.outcome,
                outcome_confidence=c.outcome_confidence,
                detection_confidence=c.detection_confidence,
                release_xy=c.release_xy, rim_xy=c.rim_xy, kind=c.kind,
                physics=phys))

        # Assemble result
        teams_out = {
            "A": team_palette.get("Team A", {"color_rgb": [40, 120, 235]}),
            "B": team_palette.get("Team B", {"color_rgb": [235, 60, 60]}),
        }
        result = AnalysisResult(
            meta={
                "version": __version__,
                "mode": mode,
                "source": str(video.name),
                "fps": meta["fps"], "width": meta["width"],
                "height": meta["height"], "n_frames": meta["n_frames"],
                "duration": meta["duration"],
                "processed_frames": len(indices),
                "frame_stride": self.cfg.frame_stride,
                "device": self.cfg.device,
                "model": self.cfg.model_name if mode == "real" else "synthetic",
                "ocr_backend": backend_available() or "none",
                "rim_detected": rim is not None,
                "rim_confidence": rim.conf if rim else 0.0,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "config": self.cfg.as_dict(),
            },
            players=players, frames=frames,
            ball_track=[{
                "frame": p.frame, "t": round(p.t, 3), "cx": round(p.cx, 1),
                "cy": round(p.cy, 1), "r": round(p.r, 1),
                "conf": round(p.conf, 3), "observed": p.observed}
                for p in timeline],
            shots=shots, teams=teams_out)

        # Export JSON/CSV + clips + render
        if progress:
            progress(0.75)
        self._export(result, video, job_dir, progress)
        result.meta["elapsed_s"] = round(time.time() - t_start, 2)
        (job_dir / "analysis.json").write_text(json.dumps(result.to_dict(), indent=2))
        if progress:
            progress(1.0)
        return result

    # -- exports ------------------------------------------------------------
    def _export(self, result: AnalysisResult, video: Path, job_dir: Path, progress):
        # CSV: shots
        with (job_dir / "shots.csv").open("w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["shot_id", "t_release", "t_start", "t_end", "shooter_track_id",
                         "team", "jersey", "outcome", "outcome_conf", "kind",
                         "launch_angle_deg", "release_speed_mps", "entry_angle_deg",
                         "calibrated"])
            for s in result.shots:
                p = s.physics
                wr.writerow([s.id, f"{s.t_release:.3f}", f"{s.t_start:.3f}",
                             f"{s.t_end:.3f}", s.shooter_track_id, s.team,
                             s.jersey_number or "?", s.outcome,
                             f"{s.outcome_confidence:.2f}", s.kind,
                             (p.launch_angle_deg if p else ""),
                             (p.release_speed_mps if p else ""),
                             (p.entry_angle_deg if p else ""),
                             (p.calibrated if p else False)])
        # CSV: players
        with (job_dir / "players.csv").open("w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["track_id", "team", "team_conf", "role", "jersey",
                         "jersey_conf", "n_frames", "first_t", "last_t"])
            for pl in result.players:
                wr.writerow([pl.track_id, pl.team, f"{pl.team_confidence:.2f}",
                             pl.role, pl.jersey_number or "?",
                             f"{pl.jersey_confidence:.2f}", pl.n_frames,
                             f"{pl.first_t:.2f}", f"{pl.last_t:.2f}"])

        # Render annotated video
        if self.cfg.render_video:
            out = job_dir / "annotated_video.mp4"
            try:
                render_video(video, result, out,
                             progress=(lambda f: progress(0.75 + 0.2 * f)) if progress else None)
                if out.exists():
                    result.meta["annotated_video"] = out.name
            except Exception as e:
                result.meta["render_error"] = str(e)

        # Shot clips
        clips_dir = job_dir / "clips"
        clips_dir.mkdir(exist_ok=True)
        pad = self.cfg.clip_padding_seconds
        src_for_clip = job_dir / "annotated_video.mp4"
        if not src_for_clip.exists():
            src_for_clip = video
        if shutil.which("ffmpeg"):
            for s in result.shots:
                start = max(0.0, s.t_start - pad)
                dur = (s.t_end + pad) - start
                clip = clips_dir / f"shot_{s.id:02d}_{s.outcome}.mp4"
                cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(src_for_clip),
                       "-t", f"{dur:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-movflags", "+faststart", "-an", str(clip)]
                try:
                    subprocess.run(cmd, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=120)
                    if clip.exists():
                        s.clip_path = f"clips/{clip.name}"
                except Exception:
                    pass
