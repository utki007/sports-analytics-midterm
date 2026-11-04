"""Derive shot events from ball/person tracks and write shots.csv."""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from detector import DetectionObs
from teams import team_for_track


@dataclass
class ShotEvent:
    shot_number: int
    start_time_seconds: float
    end_time_seconds: float
    outcome: str  # made | missed | unknown
    team: str


def _smooth_series(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) == 0:
        return values
    window = max(1, min(window, len(values)))
    if window == 1:
        return values.astype(np.float64)
    kernel = np.ones(window, dtype=np.float64) / window
    pad = window // 2
    padded = np.pad(values.astype(np.float64), (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def _moving_ball_points(
    observations: Sequence[DetectionObs],
    rim: Optional[Tuple[float, float]],
    min_speed: float = 25.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a ball timeline from moving tracks only.

    Static orange noise (jerseys, seats, rim metal) is dropped by requiring
    per-track displacement. When multiple balls share a timestamp, prefer the
    one nearer the rim among *moving* candidates.
    """
    by_track: Dict[int, List[DetectionObs]] = {}
    for o in observations:
        if o.object_type != "ball" or o.track_id is None:
            continue
        by_track.setdefault(o.track_id, []).append(o)

    moving: List[DetectionObs] = []
    for _tid, pts in by_track.items():
        pts = sorted(pts, key=lambda p: p.time_seconds)
        if len(pts) == 1:
            # Single hit near rim can still be a ball-in-net flash.
            if rim is not None:
                d = float(np.hypot(pts[0].cx - rim[0], pts[0].cy - rim[1]))
                if d < 220:
                    moving.extend(pts)
            continue
        total = 0.0
        for a, b in zip(pts, pts[1:]):
            total += float(np.hypot(b.cx - a.cx, b.cy - a.cy))
        span = max(1e-3, pts[-1].time_seconds - pts[0].time_seconds)
        speed = total / span
        if speed >= min_speed or total >= 80.0:
            moving.extend(pts)
        elif rim is not None:
            # Keep slow tracks that clearly pass through the net zone (below rim).
            for p in pts:
                if p.cy >= rim[1] + 12 and float(np.hypot(p.cx - rim[0], p.cy - rim[1])) < 180:
                    moving.append(p)

    by_t: Dict[float, List[DetectionObs]] = {}
    for o in moving:
        by_t.setdefault(round(o.time_seconds, 3), []).append(o)

    times: List[float] = []
    cxs: List[float] = []
    cys: List[float] = []
    for t in sorted(by_t):
        group = by_t[t]
        if rim is None:
            pick = max(group, key=lambda o: o.confidence)
        else:
            # Prefer high-confidence points near the rim (shot), else best confidence.
            def key(o: DetectionObs) -> Tuple[float, float]:
                dist = float(np.hypot(o.cx - rim[0], o.cy - rim[1]))
                return (dist, -o.confidence)

            near = [o for o in group if float(np.hypot(o.cx - rim[0], o.cy - rim[1])) < 260]
            pick = min(near, key=key) if near else max(group, key=lambda o: o.confidence)
        times.append(float(t))
        cxs.append(pick.cx)
        cys.append(pick.cy)
    return (
        np.array(times, dtype=np.float64),
        np.array(cxs, dtype=np.float64),
        np.array(cys, dtype=np.float64),
    )


def _nearest_person_track(
    observations: Sequence[DetectionObs],
    time_seconds: float,
    ball_xy: Tuple[float, float],
    time_window: float = 1.0,
) -> Optional[int]:
    persons = [
        o
        for o in observations
        if o.object_type == "person"
        and o.track_id is not None
        and abs(o.time_seconds - time_seconds) <= time_window
    ]
    if not persons:
        return None
    bx, by = ball_xy
    # Prefer people below the ball (shooter on court), not spectators behind rim.
    def score(o: DetectionObs) -> float:
        dist = (o.cx - bx) ** 2 + (o.cy - by) ** 2
        below_bonus = 0.0 if o.cy >= by else 1e6
        return dist + below_bonus

    best = min(persons, key=score)
    return best.track_id


def _classify_near_rim(
    t: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    idx: int,
    rim: Tuple[float, float],
    rim_radius: float,
) -> str:
    """Classify outcome using ball points around a near-rim contact."""
    rx, ry = rim
    t0 = t[idx]
    window = np.where((t >= t0 - 0.5) & (t <= t0 + 1.2))[0]
    if len(window) == 0:
        return "unknown"

    dists = np.sqrt((cx[window] - rx) ** 2 + (cy[window] - ry) ** 2)
    near = window[dists <= rim_radius * 1.6]
    if len(near) == 0:
        return "unknown"

    # Ball below rim (through net) while still near in x → made.
    through_net = near[
        (cy[near] >= ry + 12)
        & (cy[near] < ry + rim_radius * 1.8)
        & (np.abs(cx[near] - rx) < rim_radius * 1.1)
    ]
    if len(through_net) > 0:
        return "made"

    # Close approach then leave upward / sideways without settling below → miss.
    after = window[t[window] >= t0]
    if len(after) >= 2:
        if np.any(cy[after] < ry - rim_radius * 0.4):
            return "missed"
        if np.any(np.abs(cx[after] - rx) > rim_radius * 2.0):
            return "missed"
    return "unknown"


def detect_shots(
    observations: Sequence[DetectionObs],
    rim_center: Optional[Tuple[float, float]],
    team_assignment: Dict[int, str],
    frame_width: int,
    frame_height: int,
    pre_seconds: float = 1.0,
    post_seconds: float = 1.5,
    merge_gap: float = 1.0,
    duration_seconds: float = 0.0,
) -> List[ShotEvent]:
    """
    Heuristic shot detection: ball enters the rim neighborhood.

    Works with sparse/noisy color ball tracks (YOLO often misses tiny balls).
    """
    if rim_center is None:
        return []

    t, cx, cy = _moving_ball_points(observations, rim_center)
    if len(t) < 2:
        return []

    cx_s = _smooth_series(cx, window=3)
    cy_s = _smooth_series(cy, window=3)
    rx, ry = rim_center
    rim_radius = max(70.0, 0.04 * max(frame_width, frame_height))

    dists = np.sqrt((cx_s - rx) ** 2 + (cy_s - ry) ** 2)
    # Require a real approach into the rim/net neighborhood.
    near_mask = dists <= rim_radius * 1.6

    getting_closer = np.zeros(len(t), dtype=bool)
    if len(t) >= 2:
        getting_closer[1:] = dists[1:] + 15 < dists[:-1]

    # Net zone: just under the rim (through the net). Keep this tight so
    # orange jerseys under the basket are not treated as ball-in-net.
    in_net = (
        (np.abs(cx_s - rx) <= rim_radius * 1.15)
        & (cy_s >= ry + 10)
        & (cy_s <= ry + rim_radius * 1.15)
        & (dists <= rim_radius * 1.35)
    )

    # Use net-zone hits as the shot anchors (avoids long merges from noisy approach points).
    candidate_idx = np.where(in_net)[0]
    if len(candidate_idx) == 0:
        return []

    # Cluster candidate times into shot events (tight gap so separate shots stay separate).
    cluster_gap = min(0.6, max(0.35, merge_gap * 0.5))
    events: List[Tuple[float, float, str, str]] = []
    cluster: List[int] = [int(candidate_idx[0])]
    for idx in candidate_idx[1:]:
        idx = int(idx)
        if t[idx] - t[cluster[-1]] <= cluster_gap:
            cluster.append(idx)
        else:
            events.append(
                _cluster_to_event(
                    cluster,
                    t,
                    cx_s,
                    cy_s,
                    observations,
                    team_assignment,
                    rim_center,
                    rim_radius,
                    pre_seconds,
                    post_seconds,
                    duration_seconds,
                )
            )
            cluster = [idx]
    events.append(
        _cluster_to_event(
            cluster,
            t,
            cx_s,
            cy_s,
            observations,
            team_assignment,
            rim_center,
            rim_radius,
            pre_seconds,
            post_seconds,
            duration_seconds,
        )
    )

    # Drop weak clusters: need a real approach and/or clear net-zone presence.
    filtered: List[Tuple[float, float, str, str]] = []
    for start, end, outcome, team in events:
        in_iv = np.where((t >= start) & (t <= end))[0]
        if len(in_iv) == 0:
            continue
        min_d = float(dists[in_iv].min())
        if min_d > rim_radius * 1.8:
            continue

        # Look for approach: earlier ball farther from rim.
        pre = np.where((t >= start - 2.0) & (t < start + 0.1))[0]
        approached = False
        if len(pre) > 0:
            approached = float(dists[pre].max()) >= rim_radius * 1.8

        net_hits = int(
            np.sum(
                (np.abs(cx_s[in_iv] - rx) <= rim_radius * 1.15)
                & (cy_s[in_iv] >= ry + 10)
                & (cy_s[in_iv] <= ry + rim_radius * 1.15)
                & (dists[in_iv] <= rim_radius * 1.35)
            )
        )
        # Require ball under the rim AND an approach from farther away.
        if net_hits < 1 or not approached:
            continue
        filtered.append((start, end, outcome, team))

    # Merge overlapping.
    if not filtered:
        return []
    filtered.sort(key=lambda e: e[0])
    merged: List[Tuple[float, float, str, str]] = [filtered[0]]
    for start, end, outcome, team in filtered[1:]:
        ps, pe, po, pt = merged[-1]
        if start <= pe + merge_gap:
            new_outcome = po if po != "unknown" else outcome
            if outcome == "made" or (po != "made" and outcome == "missed"):
                new_outcome = outcome if outcome != "unknown" else po
            if po == "made":
                new_outcome = "made"
            new_team = pt if pt != "unknown" else team
            merged[-1] = (ps, max(pe, end), new_outcome, new_team)
        else:
            merged.append((start, end, outcome, team))

    shots: List[ShotEvent] = []
    for idx, (start, end, outcome, team) in enumerate(merged, start=1):
        shots.append(
            ShotEvent(
                shot_number=idx,
                start_time_seconds=round(start, 3),
                end_time_seconds=round(end, 3),
                outcome=outcome,
                team=team,
            )
        )
    return shots


def _cluster_to_event(
    cluster: List[int],
    t: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    observations: Sequence[DetectionObs],
    team_assignment: Dict[int, str],
    rim: Tuple[float, float],
    rim_radius: float,
    pre_seconds: float,
    post_seconds: float,
    duration_seconds: float,
) -> Tuple[float, float, str, str]:
    # Peak = closest approach to rim in the cluster; interval covers the whole cluster.
    rx, ry = rim
    dists = np.sqrt((cx[cluster] - rx) ** 2 + (cy[cluster] - ry) ** 2)
    local = int(cluster[int(np.argmin(dists))])
    outcome = _classify_near_rim(t, cx, cy, local, rim, rim_radius)
    release_t = float(t[local])
    shooter = _nearest_person_track(observations, release_t, (float(cx[local]), float(cy[local])))
    team = team_for_track(shooter, team_assignment)
    start = max(0.0, float(t[cluster[0]]) - pre_seconds)
    end = float(t[cluster[-1]]) + post_seconds
    if duration_seconds > 0:
        end = min(end, duration_seconds)
    return start, end, outcome, team


def write_shots_csv(path: Path, shots: Sequence[ShotEvent]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "shot_number",
                "start_time_seconds",
                "end_time_seconds",
                "outcome",
                "team",
            ]
        )
        for s in shots:
            writer.writerow(
                [
                    s.shot_number,
                    f"{s.start_time_seconds:.3f}",
                    f"{s.end_time_seconds:.3f}",
                    s.outcome,
                    s.team,
                ]
            )
    if not shots:
        warnings.warn(
            "No shot events detected. Check lighting, ball visibility, and rim estimate.",
            UserWarning,
            stacklevel=2,
        )
