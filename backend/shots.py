"""Shot candidate detection, shooter association, and made/missed classification.

Works on the assembled ball timeline (image space) plus the rim anchor. The
core logic is a small, unit-testable state machine over the ball's relationship
to the rim, so it can be validated with synthetic tracks (tests/test_pipeline.py)
without any video.

We are explicit about uncertainty: when outcome evidence is weak we return
"unknown" rather than guessing, and a candidate with weak upward/approach
evidence is marked kind="pass?".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ball import BallPoint


@dataclass
class ShotCandidate:
    t_release: float
    t_start: float
    t_end: float
    release_xy: Tuple[float, float]
    rim_xy: Tuple[float, float]
    outcome: str
    outcome_confidence: float
    detection_confidence: float
    kind: str
    # indices into the ball timeline covering the shot arc
    arc_frames: List[int]


def _smooth(v: np.ndarray, w: int = 3) -> np.ndarray:
    if len(v) < 3 or w <= 1:
        return v.astype(float)
    w = min(w, len(v))
    k = np.ones(w) / w
    pad = w // 2
    return np.convolve(np.pad(v.astype(float), (pad, pad), mode="edge"), k, "valid")[:len(v)]


def classify_outcome(
    xs: np.ndarray, ys: np.ndarray, rim: Tuple[float, float], rim_radius: float
) -> Tuple[str, float]:
    """Decide made/missed/unknown from ball points around the rim.

    made   : ball passes through the net zone (just below rim, within width).
    missed : ball reaches rim neighborhood then departs upward or far sideways.
    unknown: neither pattern is clear.
    """
    rx, ry = rim
    d = np.hypot(xs - rx, ys - ry)
    near = d <= rim_radius * 1.8
    if not near.any():
        return "unknown", 0.2
    # Net zone: below rim center, within ~1.1 rim radius horizontally.
    net = ((ys >= ry + 0.15 * rim_radius) & (ys <= ry + 1.6 * rim_radius)
           & (np.abs(xs - rx) <= 1.1 * rim_radius))
    if int(net.sum()) >= 2:
        # Confidence grows with how centered and how many net-zone points.
        centered = 1.0 - float(np.min(np.abs(xs[net] - rx)) / (rim_radius + 1e-6))
        conf = float(np.clip(0.5 + 0.4 * centered, 0.5, 0.95))
        return "made", conf
    # Missed: near approach then leaves up or wide.
    idx_near = np.where(near)[0]
    after = idx_near[-1]
    tail = slice(after, len(ys))
    if np.any(ys[tail] < ry - 0.5 * rim_radius) or np.any(np.abs(xs[tail] - rx) > 2.2 * rim_radius):
        return "missed", 0.55
    return "unknown", 0.3


def _segment_timeline(timeline: List[BallPoint], max_gap_s: float) -> List[Tuple[int, int]]:
    """Split the timeline into temporally-contiguous segments (index ranges).

    The ball track is sparse: a flight is a dense run, and possession/occlusion
    leaves time gaps. Each segment is a candidate flight. This avoids reasoning
    across gaps using list indices (which would conflate separate possessions).
    """
    if not timeline:
        return []
    segs = []
    start = 0
    for k in range(1, len(timeline)):
        if timeline[k].t - timeline[k - 1].t > max_gap_s:
            segs.append((start, k))
            start = k
    segs.append((start, len(timeline)))
    return segs


def detect_shots(
    timeline: List[BallPoint],
    rim_xy: Optional[Tuple[float, float]],
    frame_w: int,
    frame_h: int,
    min_approach_px: Optional[float] = None,
) -> List[ShotCandidate]:
    """Detect shot candidates: one per temporally-contiguous ball flight whose
    closest approach reaches the rim neighborhood, after an approach from
    farther away. Outcome is classified around the closest approach.
    """
    if rim_xy is None or len(timeline) < 3:
        return []
    rx, ry = rim_xy
    rim_radius = max(50.0, 0.035 * max(frame_w, frame_h))
    approach_gate = min_approach_px if min_approach_px is not None else rim_radius * 2.2

    t_all = np.array([p.t for p in timeline])
    dt_med = float(np.median(np.diff(t_all))) if len(t_all) > 1 else 0.05
    max_gap_s = max(0.3, dt_med * 4)

    candidates: List[ShotCandidate] = []
    for (s0, s1) in _segment_timeline(timeline, max_gap_s):
        if s1 - s0 < 3:
            continue
        seg = timeline[s0:s1]
        t = np.array([p.t for p in seg])
        xs = _smooth(np.array([p.cx for p in seg]))
        ys = _smooth(np.array([p.cy for p in seg]))
        conf = np.array([p.conf for p in seg])
        d = np.hypot(xs - rx, ys - ry)
        if float(d.min()) > rim_radius * 1.8:
            continue  # ball never reached the rim in this flight

        entry = int(np.argmin(d))
        approached = float(d[:entry + 1].max()) >= approach_gate if entry > 0 else False
        rose = bool(np.any(np.diff(ys[:entry + 1]) < -2)) if entry > 1 else False
        # Reject non-shots: a real shot approaches from distance AND the ball
        # traces a vertical arc (rise/fall). A near-stationary orange blob does
        # not, so it must not become a confident shot/made.
        vert_span = float(ys.max() - ys.min())
        n_obs = int(np.sum([1 for p in seg if p.observed]))
        if not approached or vert_span < rim_radius * 0.8 or n_obs < 4:
            continue
        outcome, oconf = classify_outcome(xs, ys, rim_xy, rim_radius)

        # Release: walk back from entry while the ball is moving (a flight).
        step = np.hypot(np.diff(xs), np.diff(ys))
        speed_gate = max(3.0, 0.06 * rim_radius)
        rel = entry
        while rel - 1 >= 0 and step[rel - 1] >= speed_gate:
            rel -= 1
        release_xy = (float(xs[rel]), float(ys[rel]))
        t_release = float(t[rel])
        det_conf = float(np.clip(0.3 + 0.5 * (approached + rose) / 2
                                 + 0.2 * float(conf[max(0, entry - 3):entry + 1].mean()),
                                 0.2, 0.95))
        kind = "shot" if (approached and rose) else "pass?"
        arc_frames = list(range(s0 + rel, s0 + min(len(seg), entry + 4)))
        candidates.append(ShotCandidate(
            t_release=t_release, t_start=max(0.0, t_release), t_end=float(t[-1]),
            release_xy=release_xy, rim_xy=(rx, ry), outcome=outcome,
            outcome_confidence=oconf, detection_confidence=det_conf,
            kind=kind, arc_frames=arc_frames))

    candidates.sort(key=lambda c: c.t_start)
    return candidates


def associate_shooter(
    persons_by_frame: Dict[int, List[dict]],
    frame_times: Dict[int, float],
    t_release: float,
    release_xy: Tuple[float, float],
    time_window: float = 0.7,
) -> Optional[int]:
    """Nearest persistent person at/under the ball around release time.

    `persons_by_frame[frame] = [{track_id, cx, cy}, ...]`.
    Prefers people below the ball (shooter on court, not spectators behind rim).
    """
    bx, by = release_xy
    best_tid, best_score = None, float("inf")
    for fi, persons in persons_by_frame.items():
        if abs(frame_times.get(fi, -1e9) - t_release) > time_window:
            continue
        for p in persons:
            dx = p["cx"] - bx
            dy = p["cy"] - by
            dist = dx * dx + dy * dy
            below_penalty = 0.0 if p["cy"] >= by - 40 else 1e7
            score = dist + below_penalty
            if score < best_score:
                best_score = score
                best_tid = p["track_id"]
    return best_tid
