"""Ball timeline assembly: pick one ball per frame, smooth with a constant-
acceleration Kalman filter, and interpolate short gaps.

Observed detections and interpolated/predicted points are kept distinct via the
`observed` flag so the UI and physics never confuse a guess for a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .detection import RawBall


@dataclass
class BallPoint:
    frame: int
    t: float
    cx: float
    cy: float
    r: float
    conf: float
    observed: bool


class Kalman2D:
    """Constant-acceleration Kalman filter for (x, y).

    State = [x, y, vx, vy, ax, ay]. Used to smooth detections and to predict
    through short occlusions. Gravity is not baked in (image-space, camera may
    move); acceleration is learned from data.
    """

    def __init__(self, dt: float):
        self.dt = dt
        self.x = np.zeros((6, 1))
        self.P = np.eye(6) * 1000.0
        dt2 = 0.5 * dt * dt
        self.F = np.array([
            [1, 0, dt, 0, dt2, 0],
            [0, 1, 0, dt, 0, dt2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], float)
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0]], float)
        self.Q = np.eye(6) * 0.8
        self.R = np.eye(2) * 9.0
        self._init = False

    def predict(self) -> Tuple[float, float]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0, 0]), float(self.x[1, 0])

    def correct(self, cx: float, cy: float):
        if not self._init:
            self.x[0, 0], self.x[1, 0] = cx, cy
            self._init = True
            return
        z = np.array([[cx], [cy]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P


def build_ball_timeline(
    per_frame: Dict[int, List[RawBall]],
    frame_times: Dict[int, float],
    rim_xy: Optional[Tuple[float, float]],
    frame_indices: List[int],
    dt: float,
    max_gap_frames: int = 8,
) -> List[BallPoint]:
    """Assemble a single coherent ball track.

    Strategy: greedily associate the detection nearest the Kalman prediction
    each frame (prefer high confidence and, when ambiguous, proximity to the
    rim / flight path). Fill short gaps by interpolation flagged observed=False.
    """
    kf = Kalman2D(dt)
    timeline: List[BallPoint] = []
    last_obs_frame: Optional[int] = None
    last_xy: Optional[Tuple[float, float]] = None

    for fi in frame_indices:
        t = frame_times[fi]
        cands = per_frame.get(fi, [])
        pred = kf.predict() if last_xy is not None else None

        chosen: Optional[RawBall] = None
        if cands:
            if pred is None:
                chosen = max(cands, key=lambda b: b.conf)
            else:
                def cost(b: RawBall) -> float:
                    d = np.hypot(b.cx - pred[0], b.cy - pred[1])
                    return d - 60.0 * b.conf
                chosen = min(cands, key=cost)
                # Reject wild jumps unless detection is strong.
                if np.hypot(chosen.cx - pred[0], chosen.cy - pred[1]) > 250 and chosen.conf < 0.5:
                    chosen = None

        if chosen is not None:
            kf.correct(chosen.cx, chosen.cy)
            # Interpolate across a short gap since last observation.
            if last_obs_frame is not None and last_xy is not None:
                gap = fi - last_obs_frame
                if 1 < gap <= max_gap_frames:
                    for g in range(1, gap):
                        a = g / gap
                        gi = last_obs_frame + g
                        ix = last_xy[0] + a * (chosen.cx - last_xy[0])
                        iy = last_xy[1] + a * (chosen.cy - last_xy[1])
                        timeline.append(BallPoint(
                            frame=gi, t=frame_times.get(gi, t), cx=ix, cy=iy,
                            r=chosen.r, conf=0.0, observed=False))
            timeline.append(BallPoint(
                frame=fi, t=t, cx=chosen.cx, cy=chosen.cy, r=chosen.r,
                conf=chosen.conf, observed=True))
            last_obs_frame = fi
            last_xy = (chosen.cx, chosen.cy)

    timeline.sort(key=lambda p: p.frame)
    return timeline
