"""Typed data contract shared by the pipeline, renderer, API, and frontend.

These dataclasses serialize to the JSON that the dashboard reads, so field
names here are the frontend's API. Everything is plain dataclasses (no pydantic
dependency in the hot path) with explicit `to_dict` for stable JSON.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2 in source-video pixels


# ---------------------------------------------------------------------------
# Low-level per-frame detections
# ---------------------------------------------------------------------------
@dataclass
class PersonDet:
    track_id: int
    bbox: BBox
    conf: float
    team: str = "unknown"            # "Team A" | "Team B" | "referee" | "unknown"
    role: str = "player"             # "player" | "referee" | "unknown"
    jersey_number: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "bbox": [round(v, 1) for v in self.bbox],
            "conf": round(self.conf, 3),
            "team": self.team,
            "role": self.role,
            "jersey_number": self.jersey_number,
        }


@dataclass
class BallDet:
    cx: float
    cy: float
    r: float
    conf: float
    observed: bool = True  # True = real detection, False = interpolated/predicted

    def to_dict(self) -> dict:
        return {
            "cx": round(self.cx, 1),
            "cy": round(self.cy, 1),
            "r": round(self.r, 1),
            "conf": round(self.conf, 3),
            "observed": self.observed,
        }


@dataclass
class RimDet:
    cx: float
    cy: float
    w: float
    h: float
    conf: float

    def to_dict(self) -> dict:
        return {
            "cx": round(self.cx, 1),
            "cy": round(self.cy, 1),
            "w": round(self.w, 1),
            "h": round(self.h, 1),
            "conf": round(self.conf, 3),
        }


@dataclass
class Frame:
    index: int          # frame index in the source video
    t: float            # timestamp in seconds (index / native_fps)
    persons: List[PersonDet] = field(default_factory=list)
    ball: Optional[BallDet] = None
    rim: Optional[RimDet] = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "t": round(self.t, 3),
            "persons": [p.to_dict() for p in self.persons],
            "ball": self.ball.to_dict() if self.ball else None,
            "rim": self.rim.to_dict() if self.rim else None,
        }


# ---------------------------------------------------------------------------
# Track-level aggregates
# ---------------------------------------------------------------------------
@dataclass
class PlayerTrack:
    track_id: int
    team: str = "unknown"
    team_confidence: float = 0.0
    role: str = "player"
    jersey_number: Optional[str] = None
    jersey_confidence: float = 0.0
    color_rgb: Tuple[int, int, int] = (170, 170, 170)
    n_frames: int = 0
    first_t: float = 0.0
    last_t: float = 0.0
    significant: bool = False  # persistent on-court track vs. transient fragment

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "team": self.team,
            "team_confidence": round(self.team_confidence, 3),
            "role": self.role,
            "jersey_number": self.jersey_number,
            "jersey_confidence": round(self.jersey_confidence, 3),
            "color_rgb": list(self.color_rgb),
            "n_frames": self.n_frames,
            "first_t": round(self.first_t, 3),
            "last_t": round(self.last_t, 3),
            "significant": self.significant,
        }


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
@dataclass
class PhysicsResult:
    """All physical estimates carry explicit confidence and assumptions.

    `calibrated` is False when we lack the scene geometry to make physically
    meaningful 3D claims; in that case only the 2D image-space fit and
    qualitative feedback are valid.
    """

    calibrated: bool = False
    scale_m_per_px: Optional[float] = None
    scale_source: str = "none"              # "rim_width" | "ball_size" | "none"

    # 2D image-space parabola fit (always available if enough points)
    fit_2d: Optional[dict] = None           # {coeffs, r2, n_points}

    # Physical estimates (only when calibrated) — each may be None
    launch_angle_deg: Optional[float] = None
    release_speed_mps: Optional[float] = None
    apex_height_m: Optional[float] = None
    entry_angle_deg: Optional[float] = None
    miss_distance_m: Optional[float] = None

    # Uncertainty (1-sigma) for the above, same units
    uncertainty: Dict[str, float] = field(default_factory=dict)

    # Trajectories for plotting, all in source-video image pixels
    observed_path: List[Tuple[float, float]] = field(default_factory=list)
    fitted_path: List[Tuple[float, float]] = field(default_factory=list)
    ideal_path: List[Tuple[float, float]] = field(default_factory=list)

    # Ideal (reference) shot physical params toward the rim
    ideal_launch_angle_deg: Optional[float] = None
    ideal_release_speed_mps: Optional[float] = None

    feedback: str = "insufficient calibration"
    feedback_code: str = "insufficient_calibration"
    assumptions: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["observed_path"] = [[round(x, 1), round(y, 1)] for x, y in self.observed_path]
        d["fitted_path"] = [[round(x, 1), round(y, 1)] for x, y in self.fitted_path]
        d["ideal_path"] = [[round(x, 1), round(y, 1)] for x, y in self.ideal_path]
        return d


@dataclass
class Shot:
    id: int
    t_release: float
    t_start: float
    t_end: float
    shooter_track_id: Optional[int]
    team: str
    jersey_number: Optional[str]
    outcome: str                 # "made" | "missed" | "unknown"
    outcome_confidence: float
    detection_confidence: float
    release_xy: Optional[Tuple[float, float]]
    rim_xy: Optional[Tuple[float, float]]
    kind: str = "shot"           # "shot" | "pass?" (low-confidence)
    physics: Optional[PhysicsResult] = None
    clip_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "t_release": round(self.t_release, 3),
            "t_start": round(self.t_start, 3),
            "t_end": round(self.t_end, 3),
            "shooter_track_id": self.shooter_track_id,
            "team": self.team,
            "jersey_number": self.jersey_number,
            "outcome": self.outcome,
            "outcome_confidence": round(self.outcome_confidence, 3),
            "detection_confidence": round(self.detection_confidence, 3),
            "release_xy": list(self.release_xy) if self.release_xy else None,
            "rim_xy": list(self.rim_xy) if self.rim_xy else None,
            "kind": self.kind,
            "physics": self.physics.to_dict() if self.physics else None,
            "clip": self.clip_path,
        }


@dataclass
class AnalysisResult:
    meta: dict
    players: List[PlayerTrack] = field(default_factory=list)
    frames: List[Frame] = field(default_factory=list)
    ball_track: List[dict] = field(default_factory=list)  # flat timeline incl. interpolated
    shots: List[Shot] = field(default_factory=list)
    teams: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "meta": self.meta,
            "players": [p.to_dict() for p in self.players],
            "frames": [f.to_dict() for f in self.frames],
            "ball_track": self.ball_track,
            "shots": [s.to_dict() for s in self.shots],
            "teams": self.teams,
        }
