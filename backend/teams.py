"""Team identification + on-court / referee / spectator filtering.

Approach:
  * Sample upper-torso color (HSV + Lab) per person crop, masking out skin,
    court, and background.
  * Cluster track-median colors into 2 teams with KMeans.
  * Referees: detect striped / low-saturation (black-white) torsos separately.
  * On-court vs spectators/coaches: use track persistence and motion — people
    who appear briefly at frame edges with little movement are likely not
    on-court players. When uncertain we label role/team "unknown" rather than
    guess.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from sklearn.cluster import KMeans
except Exception:  # pragma: no cover
    KMeans = None


def torso_crop(frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    bh = y2 - y1
    # Upper torso: skip the head (top 15%), take down to 50%.
    ty1 = y1 + int(bh * 0.15)
    ty2 = y1 + int(bh * 0.50)
    crop = frame[ty1:ty2, x1:x2]
    return crop if crop.size else None


def jersey_features(frame: np.ndarray, bbox) -> Optional[dict]:
    """Return color features + a referee flag for one torso crop."""
    crop = torso_crop(frame, bbox)
    if crop is None:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    hue, sat, val = hsv[..., 0].astype(float), hsv[..., 1].astype(float), hsv[..., 2].astype(float)
    # Keep colorful, mid-bright pixels; drop skin and dull court.
    mask = ((sat >= 45) & (val >= 40) & (val <= 245)
            & ~((hue >= 5) & (hue <= 25) & (sat < 110))   # skin
            & ~((hue >= 35) & (hue <= 85) & (sat < 70)))   # court green/tan
    n = int(mask.sum())
    total = crop.shape[0] * crop.shape[1]
    if n < max(15, total * 0.05):
        # Mostly low-saturation (black/white/gray): candidate referee.
        gray_frac = float(((sat < 55)).mean())
        return {"valid": False, "ref_score": gray_frac, "n": n}
    a_med = float(np.median(lab[..., 1][mask]))
    b_med = float(np.median(lab[..., 2][mask]))
    bgr_med = [float(np.median(crop[..., c][mask])) for c in range(3)]
    # Referee stripe heuristic: high contrast in value within the crop.
    val_std = float(val.std())
    return {
        "valid": True,
        "hue": float(np.median(hue[mask])),
        "sat": float(np.median(sat[mask])),
        "lab_a": a_med,
        "lab_b": b_med,
        "rgb": [bgr_med[2], bgr_med[1], bgr_med[0]],
        "ref_score": float((sat < 55).mean()) + (0.3 if val_std > 70 else 0.0),
        "n": n,
    }


class TeamClassifier:
    """Accumulate per-track features then assign teams/roles."""

    def __init__(self, min_samples: int = 20):
        self.min_samples = min_samples
        self._feats: Dict[int, List[dict]] = defaultdict(list)
        self._motion: Dict[int, float] = {}
        self._nframes: Dict[int, int] = defaultdict(int)

    def add(self, track_id: int, feat: Optional[dict]):
        self._nframes[track_id] += 1
        if feat is not None:
            self._feats[track_id].append(feat)

    def set_track_stats(self, track_id: int, total_motion: float):
        self._motion[track_id] = total_motion

    def _track_color(self, track_id: int) -> Optional[Tuple[float, float, float, List[float]]]:
        valid = [f for f in self._feats.get(track_id, []) if f.get("valid")]
        if len(valid) < 3:
            return None
        hue = float(np.median([f["hue"] for f in valid]))
        a = float(np.median([f["lab_a"] for f in valid]))
        b = float(np.median([f["lab_b"] for f in valid]))
        rgb = [float(np.median([f["rgb"][c] for f in valid])) for c in range(3)]
        return hue, a, b, rgb

    def _ref_score(self, track_id: int) -> float:
        feats = self._feats.get(track_id, [])
        if not feats:
            return 0.0
        return float(np.median([f.get("ref_score", 0.0) for f in feats]))

    def assign(self) -> Tuple[Dict[int, dict], Dict[str, dict]]:
        """Return per-track {team, team_confidence, role, color_rgb} and team palette."""
        track_ids = list(self._nframes.keys())
        colors: Dict[int, Tuple[float, float, float, List[float]]] = {}
        for tid in track_ids:
            c = self._track_color(tid)
            if c is not None:
                colors[tid] = c

        result: Dict[int, dict] = {}
        # Referee detection first (strong gray/stripe + colorless torso).
        ref_ids = set()
        for tid in track_ids:
            if tid not in colors and self._ref_score(tid) > 0.6 and self._nframes[tid] >= 3:
                ref_ids.add(tid)

        clusterable = [tid for tid in colors if tid not in ref_ids]
        team_palette = {"Team A": {}, "Team B": {}}
        label_to_team: Dict[int, str] = {}
        centers = None
        if KMeans is not None and len(clusterable) >= 2:
            feat = np.array([[colors[t][1], colors[t][2]] for t in clusterable])  # lab a,b
            k = 2 if len(clusterable) >= 2 else 1
            km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(feat)
            centers = km.cluster_centers_
            # Team A = lower lab_a (greener/bluer), stable tie-break.
            order = np.argsort(centers[:, 0])
            label_to_team = {int(order[0]): "Team A", int(order[1 % k]): "Team B"}
            labels = km.labels_
            # Build palette from mean rgb of each team.
            team_rgb: Dict[str, List[List[float]]] = defaultdict(list)
            for tid, lab in zip(clusterable, labels):
                team = label_to_team[int(lab)]
                team_rgb[team].append(colors[tid][3])
            for team, rgbs in team_rgb.items():
                arr = np.array(rgbs)
                team_palette[team] = {"color_rgb": [int(v) for v in arr.mean(axis=0)]}

        # Compose per-track results.
        median_motion = np.median([m for m in self._motion.values()]) if self._motion else 0.0
        for tid in track_ids:
            n = self._nframes[tid]
            motion = self._motion.get(tid, 0.0)
            rgb = [int(v) for v in colors[tid][3]] if tid in colors else [170, 170, 170]
            if tid in ref_ids:
                result[tid] = {"team": "referee", "team_confidence": 0.5,
                               "role": "referee", "color_rgb": rgb}
                continue
            if centers is not None and tid in clusterable:
                feat = np.array([colors[tid][1], colors[tid][2]])
                d = np.linalg.norm(centers - feat, axis=1)
                lab = int(np.argmin(d))
                team = label_to_team[lab]
                # Confidence: separation between the two cluster distances.
                conf = 0.5
                if len(d) == 2:
                    sep = abs(d[0] - d[1]) / (d.max() + 1e-6)
                    conf = float(np.clip(0.4 + 0.6 * sep, 0.3, 0.98))
                # Low-persistence, low-motion tracks at edges -> likely spectator.
                role = "player"
                if n < 3 or (motion < 0.3 * median_motion and n < 6):
                    role = "unknown"
                    conf *= 0.6
                result[tid] = {"team": team if role == "player" else "unknown",
                               "team_confidence": round(conf, 3),
                               "role": role, "color_rgb": rgb}
            else:
                result[tid] = {"team": "unknown", "team_confidence": 0.0,
                               "role": "unknown", "color_rgb": rgb}
        return result, team_palette
