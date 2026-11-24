"""Central configuration: paths, device selection, and tunable constants.

Everything that a user might reasonably want to change lives here or is
overridable via the ProcessConfig passed into the pipeline. No magic numbers
are buried in the algorithm modules without a named home.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "backend"
DATA_DIR = ROOT / "data"
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
CACHE_DIR = ROOT / "cache"
WEIGHTS_DIR = ROOT / "weights"

for _d in (DATA_DIR, UPLOAD_DIR, OUTPUT_DIR, CACHE_DIR, WEIGHTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

# ---------------------------------------------------------------------------
# Physical constants (SI units)
# ---------------------------------------------------------------------------
GRAVITY = 9.81  # m/s^2
RIM_DIAMETER_M = 0.4572  # regulation hoop inner diameter (18 in)
BALL_DIAMETER_M = 0.239  # men's size 7 basketball
RIM_HEIGHT_M = 3.048  # 10 ft


def select_device() -> str:
    """Pick the best available torch device. Honors BBALL_DEVICE override."""
    override = os.environ.get("BBALL_DEVICE")
    if override:
        return override
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@dataclass
class ProcessConfig:
    """Per-run processing knobs. Sensible defaults for a 30s clip on CPU/MPS."""

    # Detection / tracking
    model_name: str = "yolov8n.pt"  # auto-downloaded by ultralytics on first use
    device: str = field(default_factory=select_device)
    conf: float = 0.25
    iou: float = 0.5
    # Process every Nth frame. 1 = every frame. Timestamps always map back to
    # the original video via frame index / native fps.
    frame_stride: int = 1
    # Inference resolution. Larger helps small balls; capped for speed.
    imgsz: int = 1280
    tracker: str = "bytetrack.yaml"  # ultralytics built-in ByteTrack

    # Ball detection
    use_color_ball_fallback: bool = True

    # Teams
    team_min_samples: int = 20

    # Jersey OCR
    enable_ocr: bool = True  # degrades gracefully if no OCR backend installed

    # Shot detection
    shot_pre_seconds: float = 1.2
    shot_post_seconds: float = 1.6
    clip_padding_seconds: float = 1.5

    # Output
    render_video: bool = True
    max_seconds: float = 0.0  # 0 => whole clip

    def as_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "conf": self.conf,
            "iou": self.iou,
            "frame_stride": self.frame_stride,
            "imgsz": self.imgsz,
            "tracker": self.tracker,
            "use_color_ball_fallback": self.use_color_ball_fallback,
            "team_min_samples": self.team_min_samples,
            "enable_ocr": self.enable_ocr,
            "shot_pre_seconds": self.shot_pre_seconds,
            "shot_post_seconds": self.shot_post_seconds,
            "clip_padding_seconds": self.clip_padding_seconds,
            "render_video": self.render_video,
            "max_seconds": self.max_seconds,
        }


# Consistent team / overlay colors (BGR for OpenCV, hex for the dashboard).
TEAM_COLORS_BGR = {
    "Team A": (235, 180, 40),   # blue-cyan
    "Team B": (60, 60, 235),    # red
    "referee": (40, 220, 240),  # yellow
    "unknown": (170, 170, 170), # gray
}
TEAM_COLORS_HEX = {
    "Team A": "#28b4eb",
    "Team B": "#eb3c3c",
    "referee": "#f0dc28",
    "unknown": "#aaaaaa",
}
BALL_COLOR_BGR = (0, 140, 255)      # orange
TRAJ_OBSERVED_BGR = (0, 215, 255)   # yellow-orange
TRAJ_IDEAL_BGR = (120, 255, 120)    # green
