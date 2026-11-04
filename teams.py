"""Jersey/kit color clustering into Team A / Team B."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.cluster import KMeans
except ImportError:  # pragma: no cover
    KMeans = None  # type: ignore


JerseySample = Tuple[int, float, float]  # track_id, hue, sat


def extract_jersey_color(
    frame_bgr: np.ndarray,
    bbox: Sequence[float],
    torso_fraction: float = 0.4,
) -> Optional[Tuple[float, float]]:
    """
    Sample median HSV hue/sat from the upper torso region of a person bbox.

    Returns (hue, saturation) or None if the crop is empty / too filtered.
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    box_h = y2 - y1
    torso_y2 = y1 + max(1, int(box_h * torso_fraction))
    crop = frame_bgr[y1:torso_y2, x1:x2]
    if crop.size == 0:
        return None

    import cv2

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    # Keep colorful mid-bright pixels; drop skin-ish and dim/gray court noise.
    mask = (
        (sat >= 40)
        & (val >= 40)
        & (val <= 245)
        & ~((hue >= 5) & (hue <= 25) & (sat < 120))  # rough skin filter
        & ~((hue >= 35) & (hue <= 85) & (sat < 80))  # dull greenish court
    )
    if int(mask.sum()) < 20:
        return None

    return float(np.median(hue[mask])), float(np.median(sat[mask]))


def cluster_teams(
    samples: Iterable[JerseySample],
    min_samples: int = 30,
) -> Dict[int, str]:
    """
    Cluster jersey colors with k=2 and map each person track_id to a team.

    Team A = cluster with lower hue centroid (stable tie-break).
    Returns track_id -> 'Team A' | 'Team B'. Empty dict if clustering fails.
    """
    samples_list = list(samples)
    if KMeans is None or len(samples_list) < min_samples:
        return {}

    track_ids = np.array([s[0] for s in samples_list], dtype=np.int64)
    features = np.array([[s[1], s[2]] for s in samples_list], dtype=np.float64)

    try:
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        labels = kmeans.fit_predict(features)
    except Exception:
        return {}

    centroids = kmeans.cluster_centers_
    # Lower hue centroid -> Team A
    order = np.argsort(centroids[:, 0])
    label_to_team = {
        int(order[0]): "Team A",
        int(order[1]): "Team B",
    }

    per_track: Dict[int, List[str]] = defaultdict(list)
    for tid, lab in zip(track_ids, labels):
        per_track[int(tid)].append(label_to_team[int(lab)])

    assignment: Dict[int, str] = {}
    for tid, votes in per_track.items():
        assignment[tid] = Counter(votes).most_common(1)[0][0]
    return assignment


def team_for_track(track_id: Optional[int], assignment: Dict[int, str]) -> str:
    if track_id is None:
        return "unknown"
    return assignment.get(int(track_id), "unknown")
