"""Synthetic, clearly-labeled basketball sequence generator.

Produces (a) a real MP4 with two teams of moving players, a rim, and a ball that
executes known parabolic shots, and (b) ground-truth per-frame detections. The
pipeline consumes (b) in demo mode so the ENTIRE downstream stack (teams, ball
tracking, shot detection, physics, rendering, exports) is exercised without any
YOLO weights or real footage.

This is explicitly synthetic and must never be presented as real inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .config import GRAVITY, RIM_DIAMETER_M
from .physics import solve_ideal_shot


@dataclass
class DemoGroundTruth:
    fps: float
    width: int
    height: int
    n_frames: int
    # frame -> list of (track_id, (x1,y1,x2,y2), team_color_bgr)
    persons: Dict[int, List[tuple]] = field(default_factory=dict)
    # frame -> (cx, cy, r) or None
    ball: Dict[int, tuple] = field(default_factory=dict)
    rim: tuple = (0, 0, 0, 0)  # cx, cy, w, h
    # true physics per shot (for validation/reference)
    shots_truth: List[dict] = field(default_factory=list)


TEAM_A_BGR = (235, 170, 40)   # blue
TEAM_B_BGR = (50, 50, 230)    # red


def generate_demo(out_path: Path, seconds: float = 14.0, fps: float = 30.0,
                  width: int = 1280, height: int = 720,
                  seed: int = 7) -> DemoGroundTruth:
    rng = np.random.default_rng(seed)
    n_frames = int(seconds * fps)
    gt = DemoGroundTruth(fps=fps, width=width, height=height, n_frames=n_frames)

    # Rim near top-center. Metric scale fixed so rim width maps to regulation.
    rim_cx, rim_cy = int(width * 0.5), int(height * 0.30)
    scale = 0.006  # m/px
    rim_w = int(RIM_DIAMETER_M / scale)   # pixels
    rim_h = int(rim_w * 0.4)
    gt.rim = (rim_cx, rim_cy, rim_w, rim_h)

    # Players: 5 per team, wandering. Track ids 1..10.
    def init_players(team_bgr, base_x, ids):
        return [{
            "id": i, "x": base_x + rng.uniform(-80, 80),
            "y": height * 0.6 + rng.uniform(-80, 80),
            "vx": rng.uniform(-1.5, 1.5), "vy": rng.uniform(-1.0, 1.0),
            "bgr": team_bgr,
        } for i in ids]

    players = init_players(TEAM_A_BGR, width * 0.35, [1, 2, 3, 4, 5]) + \
              init_players(TEAM_B_BGR, width * 0.65, [6, 7, 8, 9, 10])
    # One referee: track id 11, gray/striped.
    ref = {"id": 11, "x": width * 0.5, "y": height * 0.75, "vx": 0.6, "vy": 0.0,
           "bgr": (180, 180, 180)}

    # Schedule 3 shots. Launch params are SOLVED so the ball is a true
    # projectile that reaches the rim (computed lazily at shot start below).
    shot_schedule = [
        {"t0": 2.0, "entry": 45.0, "shooter": 1, "made": True},
        {"t0": 6.0, "entry": 30.0, "shooter": 6, "made": False},
        {"t0": 10.0, "entry": 50.0, "shooter": 3, "made": True},
    ]
    shot_traj: Dict[int, dict] = {}  # schedule idx -> solved trajectory (lazy)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    pw, ph = 36, 90  # player box size

    for fi in range(n_frames):
        t = fi / fps
        img = np.full((height, width, 3), (40, 70, 40), np.uint8)  # court green
        # court lines
        cv2.rectangle(img, (40, int(height * 0.18)), (width - 40, height - 30), (90, 120, 90), 2)
        cv2.circle(img, (width // 2, int(height * 0.6)), 70, (90, 120, 90), 2)
        # backboard + rim
        cv2.rectangle(img, (rim_cx - 70, rim_cy - 60), (rim_cx + 70, rim_cy - 10), (230, 230, 230), 2)
        cv2.ellipse(img, (rim_cx, rim_cy), (rim_w // 2, rim_h // 2), 0, 0, 360, (0, 140, 255), 4)

        persons_gt = []
        for pl in players + [ref]:
            pl["x"] += pl["vx"]
            pl["y"] += pl["vy"]
            if pl["x"] < 60 or pl["x"] > width - 60:
                pl["vx"] *= -1
            if pl["y"] < height * 0.25 or pl["y"] > height - 60:
                pl["vy"] *= -1
            x1, y1 = int(pl["x"] - pw / 2), int(pl["y"] - ph)
            x2, y2 = int(pl["x"] + pw / 2), int(pl["y"])
            # head
            cv2.circle(img, (int(pl["x"]), y1), 14, (170, 150, 140), -1)
            # torso jersey
            cv2.rectangle(img, (x1, y1 + 12), (x2, y1 + 50), pl["bgr"], -1)
            if pl["id"] == 11:  # referee stripes
                for sx in range(x1, x2, 8):
                    cv2.line(img, (sx, y1 + 12), (sx, y1 + 50), (30, 30, 30), 3)
            # legs
            cv2.rectangle(img, (x1 + 4, y1 + 50), (x2 - 4, y2), (60, 60, 90), -1)
            persons_gt.append((pl["id"], (x1, y1, x2, y2), pl["bgr"]))
        gt.persons[fi] = persons_gt

        # Ball: during a shot follow a SOLVED projectile; otherwise idle.
        ball_pos = None
        active_idx = None
        for si, s in enumerate(shot_schedule):
            if si in shot_traj:
                if shot_traj[si]["t0"] <= t <= shot_traj[si]["t_end"]:
                    active_idx = si
                    break
            elif abs(t - s["t0"]) < 0.5 / fps + 1e-9 or (t >= s["t0"] and si not in shot_traj):
                # Solve this shot's trajectory at its start, snapshotting shooter.
                shooter = next(p for p in players if p["id"] == s["shooter"])
                x0_px, y0_px = shooter["x"], height * 0.60 - ph
                # Made aims at the net; the miss is aimed long/high so it sails
                # over the rim and descends on the far side -> clear "missed".
                tgt_px = (rim_cx if s["made"] else rim_cx + rim_w * 0.35,
                          rim_cy + rim_h * 0.2 if s["made"] else rim_cy - rim_h * 0.8)
                release_m = (x0_px * scale, (height - y0_px) * scale)
                target_m = (tgt_px[0] * scale, (height - tgt_px[1]) * scale)
                sol = solve_ideal_shot(release_m, target_m, entry_angle_deg=s["entry"])
                if sol is None:
                    continue
                # Made shots continue through the net; the miss carries past the rim.
                tf = sol["flight_time_s"] * (1.25 if s["made"] else 1.15)
                shot_traj[si] = {
                    "t0": s["t0"], "t_end": s["t0"] + tf,
                    "x0_px": x0_px, "y0_px": y0_px,
                    "launch_deg": sol["launch_angle_deg"],
                    "speed": sol["release_speed_mps"],
                    "dirx": 1.0 if target_m[0] >= release_m[0] else -1.0,
                    "shooter": s["shooter"], "made": s["made"],
                }
                gt.shots_truth.append({
                    "t0": s["t0"],
                    "true_launch_angle_deg": round(sol["launch_angle_deg"], 2),
                    "true_release_speed_mps": round(sol["release_speed_mps"], 2),
                    "shooter": s["shooter"], "intended_made": s["made"],
                })
                active_idx = si
                break
        if active_idx is not None:
            tr = shot_traj[active_idx]
            tt = t - tr["t0"]
            th = math.radians(tr["launch_deg"])
            vpx = tr["speed"] / scale
            vx = vpx * math.cos(th) * tr["dirx"]
            vy = vpx * math.sin(th)
            bx = tr["x0_px"] + vx * tt
            by = tr["y0_px"] - vy * tt + 0.5 * (GRAVITY / scale) * tt * tt
            ball_pos = (float(bx), float(by), 11.0)
        # Outside a shot the ball is in possession / occluded -> not emitted,
        # which also mirrors how a real detector drops the ball during handling.

        if ball_pos:
            gt.ball[fi] = ball_pos
            cv2.circle(img, (int(ball_pos[0]), int(ball_pos[1])), int(ball_pos[2]),
                       (0, 140, 255), -1)
            cv2.circle(img, (int(ball_pos[0]), int(ball_pos[1])), int(ball_pos[2]),
                       (30, 60, 140), 2)

        cv2.putText(img, "SYNTHETIC DEMO", (width - 260, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        writer.write(img)

    writer.release()
    return gt
