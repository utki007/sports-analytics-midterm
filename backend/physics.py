"""Shot trajectory and projectile physics.

Design philosophy (important — this is the research-sensitive module):

Monocular video from a moving/zooming camera CANNOT recover 3D ball motion
unambiguously. We never pretend otherwise. Instead we expose two clearly
separated layers:

  1. A 2D image-space parabola fit. Always available when enough ball points
     exist. Used only for visualization and qualitative feedback. Pixel
     displacement is NOT treated as physical height or velocity.

  2. A physically-constrained estimate, produced ONLY when we have a metric
     scale (meters-per-pixel, derived from the detected rim width against the
     known regulation rim diameter) AND we accept one explicit modeling
     assumption: that the shot arc lies in a vertical plane roughly parallel to
     the image plane at that scale. This is stated in `assumptions` and every
     derived number carries a 1-sigma uncertainty. When the assumption cannot
     be justified we return `calibrated=False` and give qualitative feedback.

The projectile equation solver and the reconstruction are validated against
synthetic trajectories in tests/test_physics.py.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import BALL_DIAMETER_M, GRAVITY, RIM_DIAMETER_M
from .schemas import PhysicsResult


# ---------------------------------------------------------------------------
# 2D image-space fit (visualization only)
# ---------------------------------------------------------------------------
def fit_parabola_2d(
    xs: Sequence[float], ys: Sequence[float]
) -> Optional[dict]:
    """Fit y = a*x^2 + b*x + c in image coordinates (y points down).

    Returns dict with coeffs, r2, n_points, and x-range, or None if the points
    don't span enough horizontally to define a parabola.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) < 3:
        return None
    if xs.max() - xs.min() < 1e-3:
        return None
    coeffs = np.polyfit(xs, ys, 2)
    pred = np.polyval(coeffs, xs)
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2)) + 1e-9
    r2 = 1.0 - ss_res / ss_tot
    return {
        "coeffs": [float(c) for c in coeffs],
        "r2": round(r2, 4),
        "n_points": int(len(xs)),
        "x_min": float(xs.min()),
        "x_max": float(xs.max()),
    }


def sample_parabola_2d(fit: dict, n: int = 40) -> List[Tuple[float, float]]:
    """Dense (x, y) samples of a 2D fit for drawing a smooth curve."""
    a, b, c = fit["coeffs"]
    xs = np.linspace(fit["x_min"], fit["x_max"], n)
    return [(float(x), float(a * x * x + b * x + c)) for x in xs]


# ---------------------------------------------------------------------------
# Metric scale estimation
# ---------------------------------------------------------------------------
def scale_from_rim(rim_width_px: Optional[float]) -> Optional[float]:
    """meters-per-pixel from the detected rim width vs. regulation diameter.

    Valid only near the rim's depth plane. Returns None if unusable.
    """
    if rim_width_px is None or rim_width_px < 5:
        return None
    return RIM_DIAMETER_M / float(rim_width_px)


def scale_from_ball(ball_radius_px: Optional[float]) -> Optional[float]:
    """meters-per-pixel from the ball diameter. Secondary scale source."""
    if ball_radius_px is None or ball_radius_px < 2:
        return None
    return BALL_DIAMETER_M / (2.0 * float(ball_radius_px))


# ---------------------------------------------------------------------------
# Ideal / reference projectile toward the rim
# ---------------------------------------------------------------------------
def solve_ideal_shot(
    release_m: Tuple[float, float],
    rim_m: Tuple[float, float],
    g: float = GRAVITY,
    entry_angle_deg: float = 45.0,
) -> Optional[dict]:
    """Solve a feasible projectile from release to rim with a target entry angle.

    Coordinates are metric, y up. We pick the launch so the ball arrives at the
    rim descending at `entry_angle_deg` below horizontal — the canonical "soft"
    shooting target. Returns launch angle, speed, flight time, and a sampled
    path, or None if no physical solution exists.

    Derivation: with horizontal range R = x_rim - x_release and height gain
    H = y_rim - y_release, the entry angle theta_e fixes the vertical/horizontal
    velocity ratio at the target. Using energy/kinematics the required launch
    speed and angle follow in closed form.
    """
    x0, y0 = release_m
    xr, yr = rim_m
    R = xr - x0
    H = yr - y0
    if abs(R) < 1e-3:
        return None
    direction = math.copysign(1.0, R)
    R = abs(R)

    theta_e = math.radians(entry_angle_deg)
    # Velocity components at the rim: vx (horizontal, toward rim), vy_entry<0.
    # Relationship from projectile kinematics given R, H and entry angle:
    #   tan(theta_launch) + tan(theta_entry) = 2H/R   (standard identity, y up,
    #   entry descending so its tangent is negative) -> solve launch angle.
    tan_entry = -math.tan(theta_e)  # descending
    tan_launch = (2.0 * H / R) - tan_entry
    theta_l = math.atan(tan_launch)
    if theta_l <= 0:
        return None
    # Launch speed from range equation with unequal endpoint heights:
    #   H = R*tan(theta_l) - g*R^2 / (2 * v^2 * cos^2(theta_l))
    denom = 2.0 * (R * math.tan(theta_l) - H) * (math.cos(theta_l) ** 2)
    if denom <= 1e-9:
        return None
    v = math.sqrt(g * R * R / denom)
    vx = v * math.cos(theta_l) * direction
    vy = v * math.sin(theta_l)
    # Flight time to reach x_rim
    t_flight = R / (v * math.cos(theta_l))
    # Sample path (metric, y up)
    ts = np.linspace(0, t_flight, 40)
    path = [
        (x0 + vx * t, y0 + vy * t - 0.5 * g * t * t) for t in ts
    ]
    return {
        "launch_angle_deg": math.degrees(theta_l),
        "release_speed_mps": v,
        "flight_time_s": t_flight,
        "path_m": path,
    }


# ---------------------------------------------------------------------------
# Constrained 3D reconstruction (vertical-plane assumption)
# ---------------------------------------------------------------------------
def reconstruct_from_image(
    times: Sequence[float],
    xs_px: Sequence[float],
    ys_px: Sequence[float],
    scale_m_per_px: float,
    image_height_px: int,
    g: float = GRAVITY,
) -> Optional[dict]:
    """Estimate launch angle, speed, apex, and entry angle under the stated
    vertical-plane assumption.

    Pixels -> meters: X = x_px * s ; Y = (H_img - y_px) * s  (y up).
    Horizontal: fit X(t) linear -> vx. Vertical: fit Y(t) quadratic; we do NOT
    force gravity, we fit it and compare to -g as a sanity/confidence signal.

    Returns a dict of estimates + per-quantity 1-sigma uncertainties, or None.
    """
    t = np.asarray(times, dtype=float)
    x = np.asarray(xs_px, dtype=float) * scale_m_per_px
    y = (image_height_px - np.asarray(ys_px, dtype=float)) * scale_m_per_px
    n = len(t)
    if n < 4 or (t.max() - t.min()) < 1e-2:
        return None
    t = t - t[0]

    # Horizontal: X = x0 + vx t
    xc, xcov = np.polyfit(t, x, 1, cov=True)
    vx = float(xc[0])
    vx_sd = float(math.sqrt(max(xcov[0, 0], 0.0)))

    # Vertical: Y = y0 + vy t + 0.5 a t^2  (a should be ~ -g)
    yc, ycov = np.polyfit(t, y, 2, cov=True)
    a_fit = 2.0 * float(yc[0])
    vy = float(yc[1])
    y0 = float(yc[2])
    vy_sd = float(math.sqrt(max(ycov[1, 1], 0.0)))
    a_sd = 2.0 * float(math.sqrt(max(ycov[0, 0], 0.0)))

    speed = math.hypot(vx, vy)
    # Guard against a stationary / degenerate solution
    if speed < 1e-3:
        return None
    launch = math.degrees(math.atan2(vy, abs(vx)))

    # Apex height above the release point (only meaningful if vy>0)
    apex_rel = (vy * vy) / (2.0 * g) if vy > 0 else 0.0

    # Entry angle: velocity direction at the last observed point
    t_end = t[-1]
    vy_end = vy + a_fit * t_end
    entry = math.degrees(math.atan2(abs(vy_end), abs(vx))) if abs(vx) > 1e-6 else 90.0

    # Uncertainty propagation (first order)
    speed_sd = math.hypot(vx * vx_sd, vy * vy_sd) / max(speed, 1e-6)
    # launch angle sd from vx, vy sds
    d = vx * vx + vy * vy
    launch_sd = math.degrees(math.hypot(vx * vy_sd, vy * vx_sd) / max(d, 1e-6))
    apex_sd = (vy * vy_sd / g) if vy > 0 else 0.0

    # Confidence: how close fitted vertical accel is to -g, plus fit residual.
    g_consistency = max(0.0, 1.0 - abs(abs(a_fit) - g) / g)
    pred_y = np.polyval(yc, t)
    ss_res = float(np.sum((y - pred_y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-9
    r2 = 1.0 - ss_res / ss_tot
    confidence = float(np.clip(0.5 * g_consistency + 0.5 * max(r2, 0.0), 0.0, 1.0))

    # Physical plausibility gate: a genuine basketball shot has a launch speed
    # of roughly 3-16 m/s and an upward launch angle. Reconstructions outside
    # this (noise, mis-scaled rim, non-shot orange blobs) are NOT credible, so
    # we collapse confidence and let the caller fall back to "uncalibrated".
    plausible = (3.0 <= speed <= 16.0) and (10.0 <= launch <= 82.0) and vy > 0
    if not plausible:
        confidence *= 0.15

    return {
        "vx": vx,
        "vy": vy,
        "y0": y0,
        "accel_fit": a_fit,
        "launch_angle_deg": launch,
        "release_speed_mps": speed,
        "apex_height_m": apex_rel,
        "entry_angle_deg": entry,
        "uncertainty": {
            "launch_angle_deg": launch_sd,
            "release_speed_mps": speed_sd,
            "apex_height_m": apex_sd,
            "accel_fit": a_sd,
        },
        "g_consistency": g_consistency,
        "vertical_r2": r2,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------
def classify_feedback(
    observed_launch: float,
    ideal_launch: float,
    observed_speed: float,
    ideal_speed: float,
    lateral_miss_m: Optional[float],
    confidence: float,
) -> Tuple[str, str]:
    """Map the gap between observed and ideal shot to actionable feedback.

    Only emitted when reconstruction confidence is adequate; otherwise the
    caller substitutes qualitative feedback.
    Returns (human_text, code).
    """
    if lateral_miss_m is not None and abs(lateral_miss_m) > 0.30:
        side = "right" if lateral_miss_m > 0 else "left"
        return (
            f"Aim ~{abs(lateral_miss_m):.2f} m further {('left' if side=='right' else 'right')}; "
            f"the release drifted {side} of the rim.",
            "lateral_adjustment",
        )
    dangle = observed_launch - ideal_launch
    dspeed = observed_speed - ideal_speed
    # Prefer the larger normalized deviation as the primary note.
    angle_dev = abs(dangle) / 10.0
    speed_dev = abs(dspeed) / 1.0
    if max(angle_dev, speed_dev) < 0.6:
        return ("Release angle and speed are close to an efficient arc. Good shot.", "on_target")
    if angle_dev >= speed_dev:
        if dangle < 0:
            return (f"More lift: raise the release angle ~{abs(dangle):.0f}° for a softer arc.", "more_lift")
        return (f"Less lift: flatten the release ~{abs(dangle):.0f}°; the arc is too high.", "less_lift")
    else:
        if dspeed < 0:
            return (f"More speed: add ~{abs(dspeed):.2f} m/s; the shot is short.", "more_speed")
        return (f"Less speed: ease off ~{abs(dspeed):.2f} m/s; the shot is long.", "less_speed")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def analyze_shot(
    times: Sequence[float],
    xs_px: Sequence[float],
    ys_px: Sequence[float],
    rim_xy: Optional[Tuple[float, float]],
    rim_width_px: Optional[float],
    ball_radius_px: Optional[float],
    image_height_px: int,
) -> PhysicsResult:
    """Produce a full PhysicsResult for one shot's ball path.

    `times`, `xs_px`, `ys_px` are the observed ball points (image pixels).
    """
    res = PhysicsResult()
    res.observed_path = [(float(x), float(y)) for x, y in zip(xs_px, ys_px)]

    # --- Layer 1: 2D fit (always attempt) ---
    fit = fit_parabola_2d(xs_px, ys_px)
    if fit is not None:
        res.fit_2d = fit
        res.fitted_path = sample_parabola_2d(fit)

    # --- Metric scale ---
    scale = scale_from_rim(rim_width_px)
    if scale is not None:
        res.scale_source = "rim_width"
    else:
        scale = scale_from_ball(ball_radius_px)
        if scale is not None:
            res.scale_source = "ball_size"
    res.scale_m_per_px = scale

    # --- Layer 2: constrained reconstruction ---
    recon = None
    if scale is not None and len(times) >= 4:
        recon = reconstruct_from_image(times, xs_px, ys_px, scale, image_height_px)

    if recon is None or recon["confidence"] < 0.35:
        # Not enough to make physical claims. Give qualitative visual feedback.
        res.calibrated = False
        res.confidence = recon["confidence"] if recon else 0.0
        res.assumptions = [
            "No reliable metric scale or arc fit; 3D physics not identifiable "
            "from this monocular clip.",
        ]
        if fit is not None:
            a = fit["coeffs"][0]
            # In image space y is down, so a>0 is a downward-opening-in-world arc.
            res.feedback = (
                "Visual arc only (uncalibrated). The ball follows a "
                + ("high" if a > 0 else "flat")
                + " path in the image; enable a clearer rim view for metric feedback."
            )
        else:
            res.feedback = "Insufficient ball detections to analyze this shot's arc."
        res.feedback_code = "insufficient_calibration"
        return res

    # Calibrated path
    res.calibrated = True
    res.launch_angle_deg = round(recon["launch_angle_deg"], 2)
    res.release_speed_mps = round(recon["release_speed_mps"], 2)
    res.apex_height_m = round(recon["apex_height_m"], 2)
    res.entry_angle_deg = round(recon["entry_angle_deg"], 2)
    res.uncertainty = {k: round(v, 3) for k, v in recon["uncertainty"].items()}
    res.confidence = round(recon["confidence"], 3)
    res.assumptions = [
        "Shot arc assumed to lie in a vertical plane parallel to the image "
        "plane at the rim's metric scale.",
        f"Metric scale {scale:.5f} m/px from {res.scale_source}.",
        "Pixel y inverted to world-up; apex is height above the release point.",
    ]

    # --- Ideal reference shot + feedback ---
    lateral_miss_m = None
    if rim_xy is not None and scale is not None:
        x0_px, y0_px = res.observed_path[0]
        rx, ry = rim_xy
        release_m = (x0_px * scale, (image_height_px - y0_px) * scale)
        rim_m = (rx * scale, (image_height_px - ry) * scale)
        ideal = solve_ideal_shot(release_m, rim_m)
        if ideal is not None:
            res.ideal_launch_angle_deg = round(ideal["launch_angle_deg"], 2)
            res.ideal_release_speed_mps = round(ideal["release_speed_mps"], 2)
            # Convert ideal metric path back to image pixels for plotting
            res.ideal_path = [
                (xm / scale, image_height_px - ym / scale)
                for xm, ym in ideal["path_m"]
            ]
            # Lateral miss: where the observed fit crosses rim height vs. rim x
            if res.fit_2d is not None:
                a, b, c = res.fit_2d["coeffs"]
                # Solve a x^2 + b x + c = ry for x near rim
                disc = b * b - 4 * a * (c - ry)
                if disc >= 0 and abs(a) > 1e-9:
                    sq = math.sqrt(disc)
                    x_candidates = [(-b + sq) / (2 * a), (-b - sq) / (2 * a)]
                    x_cross = min(x_candidates, key=lambda xx: abs(xx - rx))
                    lateral_miss_m = (x_cross - rx) * scale

            text, code = classify_feedback(
                res.launch_angle_deg,
                res.ideal_launch_angle_deg,
                res.release_speed_mps,
                res.ideal_release_speed_mps,
                lateral_miss_m,
                res.confidence,
            )
            res.feedback = text
            res.feedback_code = code
            if lateral_miss_m is not None:
                res.miss_distance_m = round(lateral_miss_m, 3)

    if res.feedback_code == "insufficient_calibration":
        res.feedback = (
            "Calibrated arc estimated, but no rim geometry to compare against a "
            "reference shot. Showing measured launch/speed with uncertainty."
        )
        res.feedback_code = "no_reference"
    return res


# ---------------------------------------------------------------------------
# Synthetic trajectory generator (used by tests and demo)
# ---------------------------------------------------------------------------
def generate_synthetic_trajectory(
    release_m: Tuple[float, float],
    launch_angle_deg: float,
    release_speed_mps: float,
    scale_m_per_px: float,
    image_height_px: int,
    n_points: int = 20,
    duration_s: float = 1.2,
    g: float = GRAVITY,
    noise_px: float = 0.0,
    seed: int = 0,
) -> dict:
    """Create a ground-truth 3D projectile and project it to image pixels under
    the fronto-parallel scale model. Returns times, xs_px, ys_px, and the true
    params. Used to validate reconstruct_from_image/solve_ideal_shot.
    """
    rng = np.random.default_rng(seed)
    x0, y0 = release_m
    th = math.radians(launch_angle_deg)
    vx = release_speed_mps * math.cos(th)
    vy = release_speed_mps * math.sin(th)
    ts = np.linspace(0, duration_s, n_points)
    xs_m = x0 + vx * ts
    ys_m = y0 + vy * ts - 0.5 * g * ts * ts
    xs_px = xs_m / scale_m_per_px
    ys_px = image_height_px - ys_m / scale_m_per_px
    if noise_px > 0:
        xs_px = xs_px + rng.normal(0, noise_px, size=xs_px.shape)
        ys_px = ys_px + rng.normal(0, noise_px, size=ys_px.shape)
    return {
        "times": ts.tolist(),
        "xs_px": xs_px.tolist(),
        "ys_px": ys_px.tolist(),
        "true_launch_angle_deg": launch_angle_deg,
        "true_release_speed_mps": release_speed_mps,
        "true_vx": vx,
        "true_vy": vy,
    }
