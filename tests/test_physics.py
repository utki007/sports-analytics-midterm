"""Validate projectile equations and reconstruction against synthetic data.

These tests do NOT need any video, model weights, or OCR. They prove that the
physics layer is mathematically correct under its stated assumptions.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import physics
from backend.config import GRAVITY


def test_fit_parabola_recovers_coeffs():
    xs = np.linspace(0, 100, 30)
    ys = 0.01 * xs**2 - 2.0 * xs + 50.0
    fit = physics.fit_parabola_2d(xs, ys)
    assert fit is not None
    assert fit["r2"] > 0.999
    assert abs(fit["coeffs"][0] - 0.01) < 1e-4


def test_fit_parabola_insufficient_points():
    assert physics.fit_parabola_2d([1, 2], [1, 2]) is None
    # no horizontal spread
    assert physics.fit_parabola_2d([5, 5, 5], [1, 2, 3]) is None


def test_scale_from_rim_and_ball():
    # rim 100px wide -> 0.4572/100 m/px
    s = physics.scale_from_rim(100.0)
    assert abs(s - 0.004572) < 1e-6
    assert physics.scale_from_rim(2.0) is None
    s2 = physics.scale_from_ball(30.0)  # radius px
    assert s2 and s2 > 0


@pytest.mark.parametrize("angle,speed", [(52.0, 8.0), (45.0, 9.0), (60.0, 7.5)])
def test_reconstruction_recovers_launch(angle, speed):
    scale = 0.005
    H = 720
    traj = physics.generate_synthetic_trajectory(
        release_m=(2.0, 2.0),
        launch_angle_deg=angle,
        release_speed_mps=speed,
        scale_m_per_px=scale,
        image_height_px=H,
        n_points=25,
        duration_s=1.0,
    )
    recon = physics.reconstruct_from_image(
        traj["times"], traj["xs_px"], traj["ys_px"], scale, H
    )
    assert recon is not None
    # Clean synthetic data -> should recover very closely
    assert abs(recon["launch_angle_deg"] - angle) < 1.0
    assert abs(recon["release_speed_mps"] - speed) < 0.2
    # fitted vertical acceleration should match -g
    assert abs(abs(recon["accel_fit"]) - GRAVITY) < 0.3
    assert recon["confidence"] > 0.9


def test_reconstruction_robust_to_noise():
    scale = 0.005
    H = 720
    traj = physics.generate_synthetic_trajectory(
        release_m=(2.0, 2.0),
        launch_angle_deg=50.0,
        release_speed_mps=8.0,
        scale_m_per_px=scale,
        image_height_px=H,
        n_points=30,
        duration_s=1.1,
        noise_px=2.0,
        seed=7,
    )
    recon = physics.reconstruct_from_image(
        traj["times"], traj["xs_px"], traj["ys_px"], scale, H
    )
    assert recon is not None
    assert abs(recon["launch_angle_deg"] - 50.0) < 6.0
    assert abs(recon["release_speed_mps"] - 8.0) < 1.5


def test_ideal_shot_reaches_target():
    release = (0.0, 2.0)
    rim = (4.0, 3.05)
    sol = physics.solve_ideal_shot(release, rim, entry_angle_deg=45.0)
    assert sol is not None
    # Forward-simulate and confirm it passes through the rim.
    th = math.radians(sol["launch_angle_deg"])
    v = sol["release_speed_mps"]
    vx = v * math.cos(th)
    vy = v * math.sin(th)
    t = (rim[0] - release[0]) / vx
    y_at_rim = release[1] + vy * t - 0.5 * GRAVITY * t * t
    assert abs(y_at_rim - rim[1]) < 0.02
    # Entry angle should match the requested 45 deg
    vy_entry = vy - GRAVITY * t
    entry = math.degrees(math.atan2(abs(vy_entry), abs(vx)))
    assert abs(entry - 45.0) < 1.0


def test_analyze_shot_calibrated_end_to_end():
    scale = 0.005
    H = 720
    traj = physics.generate_synthetic_trajectory(
        release_m=(1.0, 2.0),
        launch_angle_deg=52.0,
        release_speed_mps=8.2,
        scale_m_per_px=scale,
        image_height_px=H,
        n_points=24,
        duration_s=1.0,
    )
    rim_width_px = physics.RIM_DIAMETER_M / scale  # consistent rim width
    rim_xy = (traj["xs_px"][-1], traj["ys_px"][-1])
    res = physics.analyze_shot(
        traj["times"], traj["xs_px"], traj["ys_px"],
        rim_xy=rim_xy, rim_width_px=rim_width_px,
        ball_radius_px=10.0, image_height_px=H,
    )
    assert res.calibrated is True
    assert res.launch_angle_deg is not None
    assert abs(res.launch_angle_deg - 52.0) < 1.5
    assert res.ideal_launch_angle_deg is not None
    assert res.feedback_code in {
        "on_target", "more_lift", "less_lift", "more_speed",
        "less_speed", "lateral_adjustment", "no_reference",
    }
    assert len(res.ideal_path) > 0


def test_analyze_shot_uncalibrated():
    # No scale available -> must not fabricate physical numbers.
    xs = [100, 150, 200, 250, 300]
    ys = [400, 300, 250, 300, 400]
    ts = [0.0, 0.1, 0.2, 0.3, 0.4]
    res = physics.analyze_shot(
        ts, xs, ys, rim_xy=None, rim_width_px=None,
        ball_radius_px=None, image_height_px=720,
    )
    assert res.calibrated is False
    assert res.launch_angle_deg is None
    assert res.feedback_code == "insufficient_calibration"
    # but the 2D fit still exists for visualization
    assert res.fit_2d is not None
