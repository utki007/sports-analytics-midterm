"""Video IO, YOLO person tracking, color ball/rim estimates, observation stream."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from teams import extract_jersey_color

# COCO class ids
PERSON_CLS = 0
SPORTS_BALL_CLS = 32


@dataclass
class DetectionObs:
    time_seconds: float
    object_type: str  # "person" | "ball" | "rim"
    track_id: Optional[int]
    cx: float
    cy: float
    bbox: Optional[Tuple[float, float, float, float]] = None
    confidence: float = 0.0


@dataclass
class ScanResult:
    observations: List[DetectionObs] = field(default_factory=list)
    jersey_samples: List[Tuple[int, float, float]] = field(default_factory=list)
    rim_center: Optional[Tuple[float, float]] = None
    fps: float = 0.0
    frame_width: int = 0
    frame_height: int = 0
    duration_seconds: float = 0.0


def _open_video(video_path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return cap


def _orange_mask(frame_bgr: np.ndarray, strict: bool = False) -> np.ndarray:
    """Basketball / rim orange. `strict` reduces wooden-beam false positives for ball search."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    if strict:
        mask = cv2.inRange(hsv, np.array([6, 130, 130], dtype=np.uint8), np.array([20, 255, 255], dtype=np.uint8))
        mask |= cv2.inRange(hsv, np.array([0, 160, 150], dtype=np.uint8), np.array([6, 255, 255], dtype=np.uint8))
    else:
        mask = cv2.inRange(hsv, np.array([5, 100, 100], dtype=np.uint8), np.array([22, 255, 255], dtype=np.uint8))
        mask |= cv2.inRange(hsv, np.array([0, 140, 140], dtype=np.uint8), np.array([8, 255, 255], dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def _rim_roi_bounds(frame_h: int, frame_w: int) -> Tuple[int, int, int, int]:
    """Far-hoop search window (avoids scoreboard / near-camera clutter)."""
    y1, y2 = int(frame_h * 0.22), int(frame_h * 0.52)
    x1, x2 = int(frame_w * 0.48), int(frame_w * 0.70)
    return y1, y2, x1, x2


def estimate_rim_from_frame(frame_bgr: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Find orange rim in the far-court region (upper-middle), not the scoreboard.
    """
    h, w = frame_bgr.shape[:2]
    y1, y2, x1, x2 = _rim_roi_bounds(h, w)
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    mask = _orange_mask(roi)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0.0
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        # Rim is a thin ring / small orange arc — keep modest areas.
        if area < 40 or area > 4000:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        circularity = 4.0 * np.pi * area / (peri * peri)
        if circularity < 0.25:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        # Prefer mid-sized circular-ish blobs near horizontal center of ROI.
        center_bias = 1.0 - abs((cx / max(1, roi.shape[1])) - 0.55)
        score = area * max(circularity, 0.3) * (0.5 + center_bias) * (1.0 + 0.02 * radius)
        if score > best_score:
            best_score = score
            best = (float(cx + x1), float(cy + y1))
    return best


def estimate_rim_persistent(
    orange_acc: np.ndarray,
    roi_origin: Tuple[int, int],
    frames_used: int,
) -> Optional[Tuple[float, float]]:
    """
    Rim from orange pixels that persist across frames (static rim vs moving ball).
    `orange_acc` is a float accumulator of orange masks inside the rim ROI.
    """
    if frames_used < 5 or orange_acc.size == 0:
        return None
    thresh = max(3.0, frames_used * 0.25)
    persist = (orange_acc >= thresh).astype(np.uint8) * 255
    persist = cv2.morphologyEx(persist, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(persist, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    y0, x0 = roi_origin
    roi_h, roi_w = orange_acc.shape[:2]
    best = None
    best_score = 0.0
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < 30 or area > 5000:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        circularity = 4.0 * np.pi * area / (peri * peri)
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        # Real rim sits in the lower portion of the far-wall ROI; reject window/beam noise up top.
        if cy < roi_h * 0.45:
            continue
        if radius < 4 or radius > 80:
            continue
        lower_bias = 0.5 + (cy / max(1.0, roi_h))
        center_bias = 1.0 - abs((cx / max(1.0, roi_w)) - 0.55)
        score = (
            area
            * max(circularity, 0.2)
            * (1.0 + 0.03 * radius)
            * lower_bias
            * (0.5 + center_bias)
        )
        if score > best_score:
            best_score = score
            best = (float(cx + x0), float(cy + y0))
    return best


def estimate_rim_from_ball_positions(
    ball_points: Sequence[Tuple[float, float]],
    frame_height: int,
) -> Optional[Tuple[float, float]]:
    """Fallback: spatial mode of the highest ball positions."""
    if len(ball_points) < 5:
        return None
    arr = np.array(ball_points, dtype=np.float64)
    q = np.quantile(arr[:, 1], 0.15)
    high = arr[arr[:, 1] <= q]
    if len(high) == 0:
        high = arr[np.argsort(arr[:, 1])[: max(1, len(arr) // 10)]]
    cx = float(np.median(high[:, 0]))
    cy = float(np.median(high[:, 1]))
    if cy > frame_height * 0.65:
        return None
    return cx, cy


def detect_orange_balls(
    frame_bgr: np.ndarray,
    rim_center: Optional[Tuple[float, float]] = None,
) -> List[Tuple[float, float, float, float]]:
    """
    Color-based basketball candidates.

    Returns list of (cx, cy, radius, score), best first.
    Stock YOLO often misses tiny basketballs in wide 4K shots; orange blobs help.
    """
    h, w = frame_bgr.shape[:2]
    mask = _orange_mask(frame_bgr, strict=True)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Scale area gates with resolution (tuned on ~4K gym footage).
    min_area = max(12.0, (w * h) / 1_200_000)
    max_area = max(900.0, (w * h) / 8_000)

    cands: List[Tuple[float, float, float, float]] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        circularity = 4.0 * np.pi * area / (peri * peri)
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        # Skip ceiling / extreme edges / likely scoreboard LEDs.
        if cy < h * 0.12 or cy > h * 0.95:
            continue
        if cx < w * 0.02 or cx > w * 0.98:
            continue

        near_rim = False
        dist = None
        if rim_center is not None:
            dist = float(np.hypot(cx - rim_center[0], cy - rim_center[1]))
            near_rim = dist < max(200.0, 0.08 * w)
            # Ignore orange rim metal itself; keep ball-in-net slightly below.
            if dist < 55 and cy <= rim_center[1] + 12:
                continue
        # Ball in the net is often partially occluded → allow lower circularity near rim.
        min_circ = 0.30 if near_rim else 0.45
        if circularity < min_circ:
            continue

        score = circularity * np.sqrt(area)
        if rim_center is not None and dist is not None:
            # Boost candidates in the net / near-rim flight path.
            if dist < max(140.0, 0.06 * w) and cy >= rim_center[1] - 20:
                score += 40.0
            elif dist < max(240.0, 0.10 * w):
                score += 12.0
        cands.append((float(cx), float(cy), float(radius), float(score)))

    cands.sort(key=lambda x: x[3], reverse=True)
    return cands[:5]


class _BallTracker:
    """Simple nearest-neighbor tracker for color ball detections."""

    def __init__(self, max_dist: float = 180.0) -> None:
        self.max_dist = max_dist
        self._next_id = 1
        self._prev: Dict[int, Tuple[float, float]] = {}

    def update(self, detections: Sequence[Tuple[float, float, float, float]]) -> List[Tuple[int, float, float, float]]:
        """Match detections to track ids. Returns (track_id, cx, cy, score)."""
        if not detections:
            self._prev = {}
            return []

        assigned: List[Tuple[int, float, float, float]] = []
        used_prev: set[int] = set()
        used_det: set[int] = set()

        # Greedy match by distance.
        pairs: List[Tuple[float, int, int]] = []
        prev_items = list(self._prev.items())
        for di, (cx, cy, _r, score) in enumerate(detections):
            for pi, (tid, (px, py)) in enumerate(prev_items):
                dist = float(np.hypot(cx - px, cy - py))
                if dist <= self.max_dist:
                    pairs.append((dist, di, pi))
        pairs.sort()
        for _dist, di, pi in pairs:
            if di in used_det or pi in used_prev:
                continue
            tid = prev_items[pi][0]
            cx, cy, _r, score = detections[di]
            assigned.append((tid, cx, cy, score))
            used_det.add(di)
            used_prev.add(pi)

        new_prev: Dict[int, Tuple[float, float]] = {}
        for tid, cx, cy, score in assigned:
            new_prev[tid] = (cx, cy)

        for di, (cx, cy, _r, score) in enumerate(detections):
            if di in used_det:
                continue
            tid = self._next_id
            self._next_id += 1
            assigned.append((tid, cx, cy, score))
            new_prev[tid] = (cx, cy)

        self._prev = new_prev
        return assigned


def scan_video(
    video_path: Path,
    model_name: str = "yolov8n.pt",
    conf: float = 0.25,
    sample_interval: float = 0.25,
    max_jersey_samples: int = 3000,
    rim_probe_frames: int = 60,
) -> ScanResult:
    """
    Run YOLO person tracking + color ball detection on sampled frames.
    """
    from ultralytics import YOLO

    video_path = Path(video_path)
    cap = _open_video(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-3:
        fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if frame_count > 0 else 0.0

    model = YOLO(model_name)
    step = max(1, int(round(sample_interval * fps)))
    # Wider imgsz helps tiny objects a bit; ball still mainly from color.
    imgsz = 1280 if max(width, height) >= 1920 else 640

    result = ScanResult(
        fps=fps,
        frame_width=width,
        frame_height=height,
        duration_seconds=duration,
    )

    rim_votes: List[Tuple[float, float]] = []
    ball_points: List[Tuple[float, float]] = []
    ball_tracker = _BallTracker(max_dist=max(150.0, 0.05 * max(width, height)))
    rim_center: Optional[Tuple[float, float]] = None
    roi_y1, roi_y2, roi_x1, roi_x2 = _rim_roi_bounds(height, width)
    orange_acc = np.zeros((max(1, roi_y2 - roi_y1), max(1, roi_x2 - roi_x1)), dtype=np.float32)
    rim_acc_frames = 0
    frame_idx = 0
    probed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        time_seconds = frame_idx / fps

        if probed < rim_probe_frames:
            rim = estimate_rim_from_frame(frame)
            if rim is not None:
                rim_votes.append(rim)
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            if roi.size:
                orange_acc += (_orange_mask(roi) > 0).astype(np.float32)
                rim_acc_frames += 1
            probed += 1
            # Prefer persistent rim once enough frames accumulated.
            if rim_center is None and rim_acc_frames >= 8:
                rim_center = estimate_rim_persistent(
                    orange_acc, (roi_y1, roi_x1), rim_acc_frames
                )
            if rim_center is None and len(rim_votes) >= 8:
                arr = np.array(rim_votes, dtype=np.float64)
                rim_center = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))

        # Color ball detections (primary for this footage).
        color_balls = detect_orange_balls(frame, rim_center=rim_center)
        tracked = ball_tracker.update(color_balls)
        for tid, cx, cy, score in tracked:
            # Keep strongest track per frame for shot logic; still log all.
            result.observations.append(
                DetectionObs(
                    time_seconds=time_seconds,
                    object_type="ball",
                    track_id=tid,
                    cx=cx,
                    cy=cy,
                    bbox=None,
                    confidence=min(1.0, score / 40.0),
                )
            )
            ball_points.append((cx, cy))

        # YOLO: persons (+ sports ball if any).
        outputs = model.track(
            source=frame,
            persist=True,
            conf=conf,
            classes=[PERSON_CLS, SPORTS_BALL_CLS],
            imgsz=imgsz,
            verbose=False,
        )
        if outputs:
            out = outputs[0]
            boxes = out.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                if boxes.id is not None:
                    ids = boxes.id.cpu().numpy().astype(int)
                else:
                    ids = np.full(len(xyxy), -1, dtype=int)

                for i in range(len(xyxy)):
                    bx1, by1, bx2, by2 = map(float, xyxy[i])
                    cx = (bx1 + bx2) / 2.0
                    cy = (by1 + by2) / 2.0
                    cls_id = int(clss[i])
                    tid = int(ids[i]) if ids[i] >= 0 else None
                    conf_i = float(confs[i])

                    if cls_id == SPORTS_BALL_CLS:
                        result.observations.append(
                            DetectionObs(
                                time_seconds=time_seconds,
                                object_type="ball",
                                track_id=tid,
                                cx=cx,
                                cy=cy,
                                bbox=(bx1, by1, bx2, by2),
                                confidence=conf_i,
                            )
                        )
                        ball_points.append((cx, cy))
                    elif cls_id == PERSON_CLS:
                        result.observations.append(
                            DetectionObs(
                                time_seconds=time_seconds,
                                object_type="person",
                                track_id=tid,
                                cx=cx,
                                cy=cy,
                                bbox=(bx1, by1, bx2, by2),
                                confidence=conf_i,
                            )
                        )
                        if tid is not None and len(result.jersey_samples) < max_jersey_samples:
                            color = extract_jersey_color(frame, (bx1, by1, bx2, by2))
                            if color is not None:
                                result.jersey_samples.append((tid, color[0], color[1]))

        frame_idx += 1

    cap.release()

    persistent = estimate_rim_persistent(orange_acc, (roi_y1, roi_x1), rim_acc_frames)
    if persistent is not None:
        result.rim_center = persistent
    elif rim_votes:
        arr = np.array(rim_votes, dtype=np.float64)
        result.rim_center = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))
    else:
        result.rim_center = estimate_rim_from_ball_positions(ball_points, height)

    if result.rim_center is not None:
        rx, ry = result.rim_center
        result.observations.append(
            DetectionObs(
                time_seconds=0.0,
                object_type="rim",
                track_id=None,
                cx=rx,
                cy=ry,
                bbox=None,
                confidence=1.0,
            )
        )

    return result


def write_tracks_sample_csv(
    path: Path,
    observations: Sequence[DetectionObs],
    team_assignment: Dict[int, str],
) -> None:
    """Sparse debug CSV: time_seconds,object,track_id,team,cx,cy"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_seconds", "object", "track_id", "team", "cx", "cy"])
        for obs in observations:
            if obs.object_type == "rim":
                team = ""
                tid: object = ""
            elif obs.object_type == "person" and obs.track_id is not None:
                team = team_assignment.get(obs.track_id, "unknown")
                tid = obs.track_id
            else:
                team = ""
                tid = "" if obs.track_id is None else obs.track_id
            writer.writerow(
                [
                    f"{obs.time_seconds:.3f}",
                    obs.object_type,
                    tid,
                    team,
                    f"{obs.cx:.2f}",
                    f"{obs.cy:.2f}",
                ]
            )
