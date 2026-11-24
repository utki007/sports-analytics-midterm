"""Broadcast-style annotated video renderer.

Re-reads the source video and draws, per frame: team-colored player boxes with
ID + jersey, the ball marker and trail, the rim, the current shooter highlight,
shot-event banners, and a compact stats overlay. Encodes to H.264 yuv420p via
ffmpeg so the result plays in any browser.

Only confident data is drawn boldly; uncertain items are muted or omitted so the
overlay never misleads.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .config import (BALL_COLOR_BGR, TEAM_COLORS_BGR, TRAJ_IDEAL_BGR,
                     TRAJ_OBSERVED_BGR)
from .schemas import AnalysisResult

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _team_color(team: str):
    return TEAM_COLORS_BGR.get(team, TEAM_COLORS_BGR["unknown"])


def _draw_box(img, p: dict):
    x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
    color = _team_color(p["team"])
    thick = 3 if p["role"] == "player" and p["team"] in ("Team A", "Team B") else 1
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    label = f"#{p['track_id']}"
    if p.get("jersey_number"):
        label = f"{p['jersey_number']} ({p['track_id']})"
    (tw, th), _ = cv2.getTextSize(label, _FONT, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 4), _FONT, 0.5, (20, 20, 20), 1, cv2.LINE_AA)


def _draw_ball(img, ball: dict, trail: List[tuple]):
    for k, (tx, ty) in enumerate(trail[-18:]):
        a = (k + 1) / 18.0
        cv2.circle(img, (int(tx), int(ty)), max(1, int(3 * a)),
                   TRAJ_OBSERVED_BGR, -1, cv2.LINE_AA)
    cx, cy, r = int(ball["cx"]), int(ball["cy"]), max(5, int(ball["r"]))
    col = BALL_COLOR_BGR if ball["observed"] else (120, 120, 120)
    cv2.circle(img, (cx, cy), r, col, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 2, col, -1)


def _draw_rim(img, rim: dict):
    cx, cy, w, h = int(rim["cx"]), int(rim["cy"]), int(rim["w"]), int(rim["h"])
    cv2.ellipse(img, (cx, cy), (max(8, w // 2), max(4, h // 2)), 0, 0, 360,
                (0, 215, 255), 2, cv2.LINE_AA)


def _stats_panel(img, title_lines: List[str]):
    x, y = 12, 12
    w = 280
    h = 22 * len(title_lines) + 14
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, line in enumerate(title_lines):
        cv2.putText(img, line, (x + 10, y + 24 + i * 22), _FONT, 0.55,
                    (240, 240, 240), 1, cv2.LINE_AA)


def render(video_path: Path, analysis: AnalysisResult, out_path: Path,
           progress=None) -> Optional[Path]:
    """Render annotated video. Returns the output path, or None on failure."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or analysis.meta.get("fps", 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames_by_index: Dict[int, dict] = {f["index"]: f for f in
                                        (fr.to_dict() for fr in analysis.frames)}
    shots = [s.to_dict() for s in analysis.shots]

    ffmpeg = shutil.which("ffmpeg")
    writer = None
    proc = None
    if ffmpeg:
        cmd = [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
               "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", "-preset", "veryfast", str(out_path)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    ball_trail: List[tuple] = []
    # Running team scoreboard from made shots keyed by t_end.
    team_made = {"Team A": 0, "Team B": 0}
    shots_by_endframe: Dict[int, dict] = {}
    for s in shots:
        ef = int(round(s["t_end"] * fps))
        shots_by_endframe[ef] = s

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        meta = frames_by_index.get(idx)
        active_shot = None
        if meta:
            if meta.get("rim"):
                _draw_rim(frame, meta["rim"])
            for p in meta["persons"]:
                _draw_box(frame, p)
            if meta.get("ball"):
                ball_trail.append((meta["ball"]["cx"], meta["ball"]["cy"]))
                _draw_ball(frame, meta["ball"], ball_trail)
            else:
                ball_trail = ball_trail[-18:]

        # Shot event banner + trajectory overlays while a shot is on screen.
        tcur = idx / fps
        for s in shots:
            if s["t_start"] - 0.3 <= tcur <= s["t_end"] + 0.6:
                active_shot = s
                break
        if idx in shots_by_endframe:
            s = shots_by_endframe[idx]
            if s["outcome"] == "made" and s["team"] in team_made:
                team_made[s["team"]] += 2

        if active_shot:
            s = active_shot
            label = {"made": "MADE", "missed": "MISSED",
                     "unknown": "SHOT ATTEMPT"}.get(s["outcome"], "SHOT ATTEMPT")
            if s["kind"] == "pass?":
                label = "BALL EVENT"
            bcol = {"made": (80, 220, 80), "missed": (80, 80, 230)}.get(
                s["outcome"], (0, 200, 255))
            cv2.rectangle(frame, (w // 2 - 150, 20), (w // 2 + 150, 60), (20, 20, 20), -1)
            cv2.putText(frame, f"{label}", (w // 2 - 130, 48), _FONT, 0.9, bcol, 2, cv2.LINE_AA)
            # Draw observed + ideal arcs if physics present.
            phys = s.get("physics")
            if phys:
                obs = phys.get("fitted_path") or phys.get("observed_path")
                if obs and len(obs) > 1:
                    pts = np.array(obs, np.int32).reshape(-1, 1, 2)
                    cv2.polylines(frame, [pts], False, TRAJ_OBSERVED_BGR, 2, cv2.LINE_AA)
                if phys.get("calibrated") and phys.get("ideal_path"):
                    pts = np.array(phys["ideal_path"], np.int32).reshape(-1, 1, 2)
                    cv2.polylines(frame, [pts], False, TRAJ_IDEAL_BGR, 2, cv2.LINE_AA)
            # Highlight shooter.
            if meta and s.get("shooter_track_id") is not None:
                for p in meta["persons"]:
                    if p["track_id"] == s["shooter_track_id"]:
                        x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
                        cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3),
                                      (0, 255, 255), 2)
                        cv2.putText(frame, "SHOOTER", (x1, y2 + 18), _FONT, 0.5,
                                    (0, 255, 255), 1, cv2.LINE_AA)

        n_players = len(meta["persons"]) if meta else 0
        _stats_panel(frame, [
            f"Basketball Analytics  t={tcur:5.1f}s",
            f"Players tracked: {n_players}",
            f"Score  A {team_made['Team A']} - {team_made['Team B']} B",
            f"Shots: {len(shots)}",
        ])
        if meta and analysis.meta.get("mode") == "demo":
            cv2.putText(frame, "SYNTHETIC DEMO", (w - 240, h - 20), _FONT, 0.7,
                        (0, 0, 255), 2, cv2.LINE_AA)

        if proc:
            try:
                proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                break
        else:
            writer.write(frame)
        idx += 1
        if progress and total and idx % 30 == 0:
            progress(idx / total)

    cap.release()
    if proc:
        proc.stdin.close()
        proc.wait()
    if writer:
        writer.release()
    return out_path if out_path.exists() else None
