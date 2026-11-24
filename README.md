# 🏀 AI Basketball Analytics

End-to-end computer-vision + physics pipeline and dashboard for short basketball
clips. Give it a 30-second game video (xBotGo-style wide gym footage works) and
it produces:

- **Player detection & tracking** (YOLO + ByteTrack) with persistent IDs, trails,
  and on-court / referee / spectator filtering.
- **Team identification** by jersey color (HSV + Lab clustering, track-level voting).
- **Jersey numbers** via optional OCR, aggregated across frames (shows `?` when unreadable).
- **Ball detection & tracking** (YOLO sports-ball + color fallback, Kalman smoothing,
  gap interpolation) with observed vs. interpolated points kept distinct.
- **Rim/hoop estimation** from the persistent orange ring, with camera-motion compensation.
- **Shot detection** (candidate → shooter association → made/missed/**unknown**) with
  per-shot clips.
- **Trajectory physics**: an always-available 2D image-space arc, plus a *physically
  constrained* 3D estimate (launch angle, release speed, apex, entry angle, miss
  distance) **only when a metric scale and a stated assumption hold** — each number
  carries an uncertainty, and the app says "uncalibrated" when 3D is not identifiable.
- **Broadcast-style annotated MP4** and an **interactive React dashboard** with live
  canvas overlays, toggles, a trajectory chart, manual corrections, and JSON/CSV/clip exports.

A **synthetic demo mode** runs the entire pipeline without any real footage or model
weights, and is clearly labeled synthetic throughout.

---

## What it honestly does and does not claim

Monocular video from a moving/zooming camera **cannot** recover 3D ball motion
unambiguously. This project is built around that fact:

- 2D parabola fitting is used **only** for visualization — pixel displacement is never
  treated as physical height or velocity.
- 3D physical estimates are produced **only** when the detected rim gives a metric scale
  (meters/pixel vs. the regulation 0.4572 m rim) **and** we accept one explicit
  assumption: the arc lies in a vertical plane parallel to the image plane at that scale.
  That assumption and the scale source are reported with every estimate, along with a
  1-sigma uncertainty and a confidence.
- A physical-plausibility gate rejects implausible reconstructions (e.g. a 1 m/s "shot"
  from mis-scaled rim / orange-court noise) and falls back to "insufficient calibration".
- We never claim to measure muscular force, spin, or shooting power.

---

## Architecture

```
backend/
  config.py      paths, device selection, physical constants, tunable ProcessConfig
  schemas.py     dataclasses = the JSON contract shared with the frontend
  detection.py   YOLO person/ball detection + color-ball fallback
  tracking.py    camera-motion compensation (optical flow) + track stats
  court.py       rim/hoop estimation (persistent orange ring)
  ball.py        ball timeline: Kalman filter + gap interpolation
  teams.py       jersey-color clustering + referee / on-court filtering
  jersey.py      optional OCR (EasyOCR/PaddleOCR) with graceful degradation
  shots.py       shot state machine + shooter association + made/missed/unknown
  physics.py     2D fit, metric scale, constrained 3D reconstruction, ideal arc, feedback
  renderer.py    broadcast-style annotated MP4 (ffmpeg H.264)
  demo.py        synthetic labeled sequence (video + ground-truth detections)
  pipeline.py    single-pass orchestration + exports
  main.py        FastAPI: upload / process / status / analysis / media / export / correct
  cli.py         command-line entry point
frontend/        React + Vite + TypeScript + Tailwind + Recharts dashboard
tests/           test_physics.py, test_pipeline.py
```

Processing is a single pass over the video (detection + tracking + jersey sampling +
rim accumulation + camera motion), followed by analysis (teams, ball timeline, shots,
physics) and a second pass for rendering. Timestamps always map back to the source via
`frame_index / native_fps`.

---

## Install

Requires **Python 3.11+**, **Node 18+**, and **ffmpeg** on `PATH`.

```bash
./scripts/setup.sh          # installs Python + frontend deps
# or manually:
python3 -m pip install -r requirements.txt
cd frontend && npm install
```

YOLO weights (`yolov8n.pt`, ~6 MB) download automatically on first real inference.
Jersey OCR is optional — uncomment `easyocr` or `paddleocr` in `requirements.txt` to enable it.

---

## Run

### Web dashboard (recommended)

```bash
./scripts/run.sh            # builds the UI, serves API + dashboard at http://127.0.0.1:8000
# or for hot-reload development (UI on :5173, API on :8000):
./scripts/dev.sh
```

Then: **Run synthetic demo** for an instant labeled run, or **Upload MP4** to process your
own clip. Options (frame stride, inference size, max seconds, OCR) apply to the next upload.

### Command line

```bash
python3 -m backend.cli demo                              # synthetic end-to-end
python3 -m backend.cli process data/clip_30s.mp4 \
        --stride 2 --imgsz 960 --max-seconds 30          # real inference
```

Outputs land in `outputs/<name>/`: `analysis.json`, `annotated_video.mp4`,
`shots.csv`, `players.csv`, and `clips/shot_*.mp4`.

---

## API

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/upload` | upload an MP4, returns `job_id` |
| POST | `/api/process/{job_id}` | start processing (stride, imgsz, conf, max_seconds, render, ocr) |
| POST | `/api/demo` | run the synthetic demo |
| GET  | `/api/status/{job_id}` | state + progress |
| GET  | `/api/analysis/{job_id}` | full analysis JSON |
| GET  | `/api/media/{job_id}/{name}` | serve video / clips |
| GET  | `/api/export/{job_id}/{kind}` | `json` / `shots` / `players` / `video` |
| POST | `/api/correct/{job_id}` | manual team / jersey correction for a track |

---

## Models & sources

- **Detection / tracking**: [Ultralytics YOLOv8](https://docs.ultralytics.com) (COCO
  `person` + `sports ball`) with the built-in **ByteTrack** tracker. Weights are
  auto-downloaded by Ultralytics; no URLs are hardcoded.
- **Ball fallback**: HSV orange-blob detection with circularity + torso-exclusion gates.
- **Rim**: persistent orange-ring heuristic (no dedicated hoop weights assumed).
- **OCR (optional)**: EasyOCR or PaddleOCR.
- **Physics**: closed-form projectile equations; constrained reconstruction validated
  against synthetic trajectories in `tests/test_physics.py`.

---

## Tests

```bash
python3 -m pytest tests/ -q
```

Covers: projectile math & reconstruction vs. synthetic trajectories, the ideal-shot
solver, 2D fit, metric scale, video IO + timestamp mapping, Kalman prediction, ball-gap
interpolation, camera-motion estimation, the shot state machine (made / no-rim / empty /
separate shots / ambiguous→unknown), team-clustering stability, schema round-trip, and an
API health check. (22 tests.)

---

## Known limitations

- **Wide amateur footage is hard.** Small, fast balls and orange court markings make ball
  tracking noisy; the color fallback can lock onto non-ball orange. Shots/outcomes on such
  clips are approximate and the physics will usually report *uncalibrated* — by design.
- **Rim estimation** is a heuristic; on courts with lots of orange it may mislocate. A
  dedicated hoop detector would improve this and is the highest-value next step.
- **Track fragmentation**: a panning camera and tiny players produce ID switches, so the
  raw track count can be large. The dashboard foregrounds *significant* (persistent)
  tracks; "show all tracks" reveals the rest.
- **Made/missed** is inferred from 2D ball–rim geometry; ambiguous cases are labeled
  **unknown** rather than guessed.
- **Jersey OCR** needs a backend installed and legible numbers; otherwise `?`.

## Troubleshooting

- *`ffmpeg not found`* → install it (`brew install ffmpeg` / `apt install ffmpeg`). Without
  it the renderer falls back to OpenCV `mp4v`, which some browsers won't play.
- *Slow on CPU* → increase `--stride`, lower `--imgsz`, or set a short `--max-seconds`.
  Set `BBALL_DEVICE=cpu|mps|cuda` to override device auto-selection.
- *Video won't play in the dashboard* → the annotated MP4 is H.264/yuv420p via ffmpeg;
  confirm ffmpeg is installed so re-encoding succeeds.
- *No shots detected* → the rim may be unlocated or the ball unseen; check the meta bar
  (`rim: yes/no`) and try a clip with a clearer hoop view.

---

## Next highest-value improvement

Swap the heuristic rim/ball steps for a **small basketball-specific detector** (fine-tuned
YOLO on rim + ball). Reliable rim geometry is the single biggest unlock: it both fixes
the metric scale (making 3D physics trustworthy on real clips) and sharpens shot
localization and made/missed classification.
