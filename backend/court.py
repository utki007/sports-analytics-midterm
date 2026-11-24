"""Rim/hoop estimation and (optional) court context.

There is no COCO class for a basketball rim, and reliable rim detectors require
dedicated weights we don't assume exist. We therefore estimate the rim from the
persistent orange ring in the upper portion of the frame, accumulated across
frames so the static rim separates from the moving ball. The estimate carries a
confidence and is intentionally conservative: if nothing persistent is found we
return None rather than guessing.

We do NOT compute a court homography unless strong, stable correspondences are
available — which this heuristic pipeline does not attempt. We are explicit that
a 2D rim location is an image-space anchor, not a 3D reconstruction.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .schemas import RimDet


class RimEstimator:
    """Accumulate orange-ring evidence to localize a static rim."""

    def __init__(self, frame_w: int, frame_h: int):
        self.w = frame_w
        self.h = frame_h
        # Search band: upper-middle of the frame, away from floor and extreme edges.
        self.y1, self.y2 = int(frame_h * 0.10), int(frame_h * 0.60)
        self.x1, self.x2 = int(frame_w * 0.05), int(frame_w * 0.95)
        self._acc = np.zeros((max(1, self.y2 - self.y1),
                              max(1, self.x2 - self.x1)), np.float32)
        self._frames = 0

    def update(self, frame_bgr: np.ndarray) -> None:
        roi = frame_bgr[self.y1:self.y2, self.x1:self.x2]
        if roi.size == 0:
            return
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, np.array([3, 120, 110], np.uint8),
                        np.array([20, 255, 255], np.uint8))
        self._acc += (m > 0).astype(np.float32)
        self._frames += 1

    def estimate(self) -> Optional[RimDet]:
        if self._frames < 4:
            return None
        thresh = max(3.0, self._frames * 0.30)
        persist = (self._acc >= thresh).astype(np.uint8) * 255
        persist = cv2.morphologyEx(persist, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(persist, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best: Optional[RimDet] = None
        best_score = 0.0
        roi_h, roi_w = self._acc.shape
        # A basketball rim occupies a small, bounded span of the frame width.
        # Orange COURT LINES are long and thin (huge aspect) -> rejected here.
        min_w = max(8.0, 0.015 * self.w)
        max_w = max(min_w + 4.0, 0.12 * self.w)
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < 20 or area > 8000:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw < min_w or bw > max_w:
                continue
            aspect = bw / max(1.0, bh)
            # Rim seen head-on is wider than tall, but not a long thin line.
            if aspect < 1.2 or aspect > 6.0:
                continue
            cx = x + bw / 2 + self.x1
            cy = y + bh / 2 + self.y1
            persistence = float(self._acc[int(y):int(y + bh),
                                          int(x):int(x + bw)].mean()) / self._frames
            score = area * aspect * (0.5 + persistence)
            if score > best_score:
                best_score = score
                conf = float(np.clip(persistence * 1.5, 0.0, 1.0))
                best = RimDet(cx=cx, cy=cy, w=float(bw), h=float(bh), conf=conf)
        return best


def rim_from_ball_apexes(ball_points: List[Tuple[float, float]],
                         frame_h: int) -> Optional[RimDet]:
    """Fallback: cluster the highest ball positions as a coarse rim anchor.

    Low confidence by construction; only used when no orange ring persists.
    """
    if len(ball_points) < 6:
        return None
    arr = np.asarray(ball_points, float)
    q = np.quantile(arr[:, 1], 0.12)
    high = arr[arr[:, 1] <= q]
    if len(high) < 3:
        return None
    cx = float(np.median(high[:, 0]))
    cy = float(np.median(high[:, 1]))
    if cy > frame_h * 0.7:
        return None
    return RimDet(cx=cx, cy=cy, w=40.0, h=20.0, conf=0.2)
