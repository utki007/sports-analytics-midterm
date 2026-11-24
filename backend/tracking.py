"""Camera-motion compensation and track-level utilities.

Ultralytics ByteTrack already gives us per-frame track IDs. What it does NOT do
is account for a panning/zooming camera when we reason about *where things are*
(rim stability, ball association). This module estimates frame-to-frame global
motion so downstream geometry can be expressed in a stabilized reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CameraMotion:
    """Per-frame estimated global motion relative to the previous frame.

    dx, dy are translation in pixels; scale is zoom factor (>1 = zoom in).
    `ok` indicates a reliable estimate.
    """
    dx: float = 0.0
    dy: float = 0.0
    scale: float = 1.0
    ok: bool = False


class CameraMotionEstimator:
    """Estimate global camera motion with sparse optical flow (Lucas-Kanade).

    Lightweight and robust enough for pan/zoom compensation on gym footage.
    """

    def __init__(self, max_corners: int = 200):
        self.max_corners = max_corners
        self._prev_gray: Optional[np.ndarray] = None

    def update(self, frame_bgr: np.ndarray) -> CameraMotion:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return CameraMotion(ok=False)
        prev = self._prev_gray
        self._prev_gray = gray
        p0 = cv2.goodFeaturesToTrack(prev, maxCorners=self.max_corners,
                                     qualityLevel=0.01, minDistance=12)
        if p0 is None or len(p0) < 8:
            return CameraMotion(ok=False)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None)
        if p1 is None or st is None:
            return CameraMotion(ok=False)
        st = st.reshape(-1).astype(bool)
        a, b = p0.reshape(-1, 2)[st], p1.reshape(-1, 2)[st]
        if len(a) < 8:
            return CameraMotion(ok=False)
        # Partial affine handles translation + uniform scale + rotation.
        M, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                                 ransacReprojThreshold=3)
        if M is None:
            return CameraMotion(ok=False)
        dx, dy = float(M[0, 2]), float(M[1, 2])
        scale = float(np.hypot(M[0, 0], M[0, 1]))
        return CameraMotion(dx=dx, dy=dy, scale=scale, ok=True)


@dataclass
class TrackStats:
    """Accumulated per-track info for court/person classification."""
    track_id: int
    centers: List[Tuple[float, float]] = field(default_factory=list)
    heights: List[float] = field(default_factory=list)
    confs: List[float] = field(default_factory=list)
    first_frame: int = 0
    last_frame: int = 0
    n: int = 0

    def add(self, cx: float, cy: float, h: float, conf: float, frame: int):
        self.centers.append((cx, cy))
        self.heights.append(h)
        self.confs.append(conf)
        if self.n == 0:
            self.first_frame = frame
        self.last_frame = frame
        self.n += 1

    def total_motion(self) -> float:
        if len(self.centers) < 2:
            return 0.0
        pts = np.asarray(self.centers)
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    def median_height(self) -> float:
        return float(np.median(self.heights)) if self.heights else 0.0

    def median_center(self) -> Tuple[float, float]:
        pts = np.asarray(self.centers)
        return float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))
