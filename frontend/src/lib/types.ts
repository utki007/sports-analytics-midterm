// Mirrors backend/schemas.py analysis.json contract.

export interface PersonDet {
  track_id: number;
  bbox: [number, number, number, number];
  conf: number;
  team: string;
  role: string;
  jersey_number: string | null;
}

export interface BallDet {
  cx: number; cy: number; r: number; conf: number; observed: boolean;
}

export interface RimDet { cx: number; cy: number; w: number; h: number; conf: number; }

export interface FrameData {
  index: number;
  t: number;
  persons: PersonDet[];
  ball: BallDet | null;
  rim: RimDet | null;
}

export interface PlayerTrack {
  track_id: number;
  team: string;
  team_confidence: number;
  role: string;
  jersey_number: string | null;
  jersey_confidence: number;
  color_rgb: [number, number, number];
  n_frames: number;
  first_t: number;
  last_t: number;
  significant: boolean;
}

export interface Physics {
  calibrated: boolean;
  scale_m_per_px: number | null;
  scale_source: string;
  fit_2d: { coeffs: number[]; r2: number; n_points: number } | null;
  launch_angle_deg: number | null;
  release_speed_mps: number | null;
  apex_height_m: number | null;
  entry_angle_deg: number | null;
  miss_distance_m: number | null;
  uncertainty: Record<string, number>;
  observed_path: [number, number][];
  fitted_path: [number, number][];
  ideal_path: [number, number][];
  ideal_launch_angle_deg: number | null;
  ideal_release_speed_mps: number | null;
  feedback: string;
  feedback_code: string;
  assumptions: string[];
  confidence: number;
}

export interface Shot {
  id: number;
  t_release: number;
  t_start: number;
  t_end: number;
  shooter_track_id: number | null;
  team: string;
  jersey_number: string | null;
  outcome: string;
  outcome_confidence: number;
  detection_confidence: number;
  release_xy: [number, number] | null;
  rim_xy: [number, number] | null;
  kind: string;
  physics: Physics | null;
  clip: string | null;
}

export interface Analysis {
  meta: any;
  players: PlayerTrack[];
  frames: FrameData[];
  ball_track: any[];
  shots: Shot[];
  teams: { A: { color_rgb: number[] }; B: { color_rgb: number[] } };
}

export const TEAM_HEX: Record<string, string> = {
  "Team A": "#28b4eb",
  "Team B": "#eb3c3c",
  referee: "#f0dc28",
  unknown: "#aaaaaa",
};
