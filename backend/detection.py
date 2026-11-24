"""Object detection: YOLO for people + sports-ball, plus a color-based ball
fallback because stock COCO YOLO rarely sees small basketballs in wide footage.

The model is loaded once and reused (single-pass architecture). Tracking IDs
come from Ultralytics' built-in ByteTrack via `model.track(persist=True)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

# COCO class ids
PERSON_CLS = 0
SPORTS_BALL_CLS = 32


@dataclass
class RawPerson:
    track_id: Optional[int]
    bbox: Tuple[float, float, float, float]
    conf: float


@dataclass
class RawBall:
    cx: float
    cy: float
    r: float
    conf: float
    source: str  # "yolo" | "color"


class Detector:
    """Thin wrapper around an Ultralytics YOLO model with a stateful tracker."""

    def __init__(self, model_name: str, device: str, conf: float, iou: float,
                 imgsz: int, tracker: str = "bytetrack.yaml"):
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.tracker = tracker

    def detect_track(self, frame: np.ndarray) -> Tuple[List[RawPerson], List[RawBall]]:
        """Run one tracked inference step. Returns (persons, yolo_balls)."""
        out = self.model.track(
            source=frame,
            persist=True,
            conf=self.conf,
            iou=self.iou,
            classes=[PERSON_CLS, SPORTS_BALL_CLS],
            imgsz=self.imgsz,
            tracker=self.tracker,
            device=self.device,
            verbose=False,
        )
        persons: List[RawPerson] = []
        balls: List[RawBall] = []
        if not out:
            return persons, balls
        boxes = out[0].boxes
        if boxes is None or len(boxes) == 0:
            return persons, balls
        xyxy = boxes.xyxy.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        ids = (boxes.id.cpu().numpy().astype(int)
               if boxes.id is not None else np.full(len(xyxy), -1))
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = map(float, xyxy[i])
            if clss[i] == PERSON_CLS:
                persons.append(RawPerson(
                    track_id=int(ids[i]) if ids[i] >= 0 else None,
                    bbox=(x1, y1, x2, y2), conf=float(confs[i])))
            elif clss[i] == SPORTS_BALL_CLS:
                balls.append(RawBall(
                    cx=(x1 + x2) / 2, cy=(y1 + y2) / 2,
                    r=max(x2 - x1, y2 - y1) / 2,
                    conf=float(confs[i]), source="yolo"))
        return persons, balls


# ---------------------------------------------------------------------------
# Color-based basketball fallback
# ---------------------------------------------------------------------------
def _orange_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array([5, 110, 110], np.uint8),
                    np.array([22, 255, 255], np.uint8))
    m |= cv2.inRange(hsv, np.array([0, 140, 130], np.uint8),
                     np.array([6, 255, 255], np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m


def detect_color_balls(frame: np.ndarray,
                       person_boxes: Optional[List[Tuple[float, float, float, float]]] = None
                       ) -> List[RawBall]:
    """Find round orange blobs that plausibly are a basketball.

    Area gates scale with resolution. Circularity filters out beams/jerseys.
    Returns up to 5 candidates, best first.
    """
    h, w = frame.shape[:2]
    mask = _orange_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(6.0, (w * h) / 2_500_000)
    max_area = max(250.0, (w * h) / 20_000)  # a basketball is small

    def in_torso(cx, cy) -> bool:
        if not person_boxes:
            return False
        for (x1, y1, x2, y2) in person_boxes:
            bh = y2 - y1
            # upper-torso band where a jersey would show
            if x1 <= cx <= x2 and (y1 + 0.1 * bh) <= cy <= (y1 + 0.55 * bh):
                return True
        return False

    cands: List[RawBall] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        circ = 4.0 * np.pi * area / (peri * peri)
        if circ < 0.70:  # basketball is round; jersey/line blobs are not
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        if cy < h * 0.08 or cy > h * 0.97:
            continue
        if in_torso(cx, cy):
            continue  # orange jersey, not the ball
        score = circ * float(np.sqrt(area))
        cands.append(RawBall(cx=float(cx), cy=float(cy), r=float(r),
                             conf=float(min(0.8, score / 24.0)), source="color"))
    cands.sort(key=lambda b: b.conf, reverse=True)
    return cands[:4]
