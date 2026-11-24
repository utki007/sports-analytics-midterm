"""Tests for the team/player shooting rollup (backend/analytics.py)."""
from backend.analytics import compute_team_analytics
from backend.schemas import PlayerTrack, Shot, PhysicsResult


def _shot(sid, team, shooter, outcome, calib=False, la=None, ea=None):
    ph = PhysicsResult(calibrated=calib, launch_angle_deg=la,
                       entry_angle_deg=ea, release_speed_mps=5.0) if calib else None
    return Shot(id=sid, t_release=0.0, t_start=0.0, t_end=1.0,
                shooter_track_id=shooter, team=team, jersey_number=None,
                outcome=outcome, outcome_confidence=0.8, detection_confidence=0.8,
                release_xy=None, rim_xy=None, physics=ph)


def _players():
    return [PlayerTrack(track_id=3, team="Team A", role="player", significant=True),
            PlayerTrack(track_id=4, team="Team A", role="player", significant=True),
            PlayerTrack(track_id=8, team="Team B", role="player", significant=True),
            PlayerTrack(track_id=99, team="Team B", role="referee", significant=True)]


def test_fg_pct_excludes_unknown_but_conservative_counts_it():
    shots = [_shot(1, "Team A", 3, "made", calib=True, la=52, ea=45),
             _shot(2, "Team A", 3, "missed"),
             _shot(3, "Team A", 4, "unknown")]
    out = compute_team_analytics(_players(), shots, duration_s=120.0)
    a = next(t for t in out["teams"] if t["team"] == "Team A")
    assert a["made"] == 1 and a["missed"] == 1 and a["unknown"] == 1
    assert a["fg_pct"] == 50.0                 # 1 / (1+1), unknown excluded
    assert a["fg_pct_conservative"] == 33.3    # 1 / 3, unknown as miss
    assert a["points"] == 2                     # 2 per made


def test_referees_not_counted_as_team_players():
    out = compute_team_analytics(_players(), [], duration_s=60.0)
    b = next(t for t in out["teams"] if t["team"] == "Team B")
    assert b["players"] == 1                     # referee #99 excluded


def test_arc_averages_only_over_calibrated_shots():
    shots = [_shot(1, "Team A", 3, "made", calib=True, la=50, ea=46),
             _shot(2, "Team A", 3, "made", calib=False)]  # no physics
    out = compute_team_analytics(_players(), shots, duration_s=60.0)
    a = next(t for t in out["teams"] if t["team"] == "Team A")
    assert a["calibrated_shots"] == 1
    assert a["avg_launch_angle_deg"] == 50.0


def test_empty_inputs_are_safe():
    out = compute_team_analytics([], [], duration_s=0.0)
    assert out["game"]["total_shots"] == 0
    assert out["game"]["shots_per_minute"] is None
    a = next(t for t in out["teams"] if t["team"] == "Team A")
    assert a["fg_pct"] is None and a["attempts"] == 0


def test_player_rows_ranked_and_margin():
    shots = [_shot(1, "Team A", 3, "made"), _shot(2, "Team A", 3, "made"),
             _shot(3, "Team B", 8, "missed")]
    out = compute_team_analytics(_players(), shots, duration_s=60.0)
    assert out["players"][0]["track_id"] == 3     # top scorer first
    assert out["players"][0]["points"] == 4
    assert out["game"]["margin"] == 4             # A 4 - B 0
