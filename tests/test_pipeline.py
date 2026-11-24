"""Pipeline, schema, tracking, and shot-state-machine tests.

Most tests are synthetic and need no model weights or real video. One test
generates a tiny MP4 to validate video IO + timestamp mapping. Failure cases
(no rim, no ball, occlusion gaps) are covered explicitly.
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.ball import BallPoint, Kalman2D, build_ball_timeline
from backend.detection import RawBall
from backend.schemas import AnalysisResult, BallDet, Frame, PersonDet, Shot
from backend.shots import classify_outcome, detect_shots
from backend.teams import TeamClassifier
from backend.tracking import CameraMotionEstimator


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def test_schema_roundtrip():
    f = Frame(index=0, t=0.0,
              persons=[PersonDet(1, (0, 0, 10, 20), 0.9, team="Team A")],
              ball=BallDet(5, 5, 3, 0.8, True))
    d = f.to_dict()
    assert d["persons"][0]["team"] == "Team A"
    assert d["ball"]["observed"] is True
    ar = AnalysisResult(meta={"mode": "demo"}, frames=[f])
    assert ar.to_dict()["meta"]["mode"] == "demo"


# ---------------------------------------------------------------------------
# Video IO + timestamp mapping
# ---------------------------------------------------------------------------
def test_video_meta_and_timestamps(tmp_path):
    from backend.pipeline import _video_meta
    path = tmp_path / "tiny.mp4"
    fps = 20.0
    w, h, n = 64, 48, 40
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        vw.write(np.full((h, w, 3), i % 255, np.uint8))
    vw.release()
    meta = _video_meta(path)
    assert meta["width"] == w and meta["height"] == h
    assert abs(meta["fps"] - fps) < 1.0
    # timestamp of frame k = k / fps
    assert abs((10 / meta["fps"]) - 0.5) < 0.05


# ---------------------------------------------------------------------------
# Ball timeline: Kalman + interpolation
# ---------------------------------------------------------------------------
def test_kalman_predicts_forward():
    kf = Kalman2D(dt=0.1)
    for i in range(5):
        kf.correct(float(i), float(2 * i))
    px, py = kf.predict()
    assert px == pytest.approx(5, abs=3)


def test_ball_timeline_interpolates_gaps():
    # Ball observed at frames 0,1,2 then gap, then 6,7 -> frames 3,4,5 interpolated.
    per_frame = {i: [] for i in range(8)}
    for i in [0, 1, 2]:
        per_frame[i] = [RawBall(cx=float(i * 10), cy=100.0, r=5, conf=0.9, source="yolo")]
    for i in [6, 7]:
        per_frame[i] = [RawBall(cx=float(i * 10), cy=100.0, r=5, conf=0.9, source="yolo")]
    times = {i: i * 0.1 for i in range(8)}
    tl = build_ball_timeline(per_frame, times, (100, 50), list(range(8)), dt=0.1)
    observed = [p for p in tl if p.observed]
    interp = [p for p in tl if not p.observed]
    assert len(observed) == 5
    assert len(interp) == 3  # frames 3,4,5
    # interpolation is monotonic between 20 and 60
    xs = [p.cx for p in sorted(tl, key=lambda q: q.frame)]
    assert xs == sorted(xs)


# ---------------------------------------------------------------------------
# Camera motion
# ---------------------------------------------------------------------------
def test_camera_motion_detects_pan():
    est = CameraMotionEstimator()
    base = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    est.update(base)
    shifted = np.roll(base, 8, axis=1)  # pan right by 8 px
    m = est.update(shifted)
    assert m.ok
    assert abs(m.dx - 8) < 4  # recovered translation ~ 8


# ---------------------------------------------------------------------------
# Shot state machine on synthetic ball tracks
# ---------------------------------------------------------------------------
def _parabola_track(rim, n=18, made=True, t0=0.0, fps=30.0):
    """Synthetic ascending->descending ball flight ending near the rim."""
    rx, ry = rim
    pts = []
    x0, y0 = rx - 160, ry + 160  # start low-left
    vx = 160 / (n / fps)
    for k in range(n):
        t = k / fps
        a = k / (n - 1)
        x = x0 + vx * t
        # parabola: up then down to rim height
        y = y0 - 260 * a + 260 * a * a * (1.0 if made else 0.4)
        pts.append(BallPoint(frame=int((t0 + t) * fps), t=t0 + t,
                             cx=float(x), cy=float(y), r=6, conf=0.8, observed=True))
    if made:
        # add through-net points below rim
        for j in range(1, 4):
            t = (n - 1 + j) / fps
            pts.append(BallPoint(frame=int((t0 + t) * fps), t=t0 + t,
                                 cx=float(rx), cy=float(ry + 10 * j),
                                 r=6, conf=0.8, observed=True))
    return pts


def test_detect_made_shot():
    rim = (320, 200)
    tl = _parabola_track(rim, made=True)
    shots = detect_shots(tl, rim, 640, 360)
    assert len(shots) == 1
    assert shots[0].outcome == "made"
    assert shots[0].kind == "shot"


def test_detect_no_rim_returns_empty():
    rim = (320, 200)
    tl = _parabola_track(rim, made=True)
    assert detect_shots(tl, None, 640, 360) == []


def test_detect_empty_ball_returns_empty():
    assert detect_shots([], (320, 200), 640, 360) == []


def test_two_separate_shots_not_merged():
    rim = (320, 200)
    tl = _parabola_track(rim, made=True, t0=0.0) + _parabola_track(rim, made=True, t0=5.0)
    shots = detect_shots(tl, rim, 640, 360)
    assert len(shots) == 2
    assert shots[1].t_release - shots[0].t_release > 4.0


def test_classify_outcome_unknown_when_ambiguous():
    rim = (320, 200)
    # ball stays far from rim
    xs = np.array([50, 60, 70, 80.0])
    ys = np.array([300, 300, 300, 300.0])
    outcome, conf = classify_outcome(xs, ys, rim, 40)
    assert outcome == "unknown"


# ---------------------------------------------------------------------------
# Team classifier stability
# ---------------------------------------------------------------------------
def test_team_classifier_assigns_two_teams():
    tc = TeamClassifier(min_samples=5)
    rng = np.random.default_rng(1)
    # team A: blue-ish (low lab_a), team B: red-ish (high lab_a)
    for tid in range(1, 6):
        for _ in range(8):
            tc.add(tid, {"valid": True, "hue": 110, "sat": 180,
                         "lab_a": 110 + rng.normal(0, 2), "lab_b": 120,
                         "rgb": [40, 60, 200], "ref_score": 0.1, "n": 100})
        tc.set_track_stats(tid, 500)
    for tid in range(6, 11):
        for _ in range(8):
            tc.add(tid, {"valid": True, "hue": 5, "sat": 180,
                         "lab_a": 180 + rng.normal(0, 2), "lab_b": 150,
                         "rgb": [200, 40, 40], "ref_score": 0.1, "n": 100})
        tc.set_track_stats(tid, 500)
    result, palette = tc.assign()
    teams_a = {result[t]["team"] for t in range(1, 6)}
    teams_b = {result[t]["team"] for t in range(6, 11)}
    # each group is internally consistent and the two groups differ
    assert len(teams_a) == 1 and len(teams_b) == 1
    assert teams_a != teams_b


# ---------------------------------------------------------------------------
# API smoke test (optional: needs httpx/starlette TestClient)
# ---------------------------------------------------------------------------
def test_api_health():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from backend.main import app
    client = TestClient(app)
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["ok"] is True
    # unknown job
    assert client.get("/api/status/nope").status_code == 404
