# Basketball Auto Shot-Event Scanner

Small CLI that scans a full basketball video with Ultralytics YOLO, estimates the rim, clusters players into Team A / Team B by jersey color, and writes automatic shot-event rows to CSV.

Clip trimming is deferred; this version only produces CSVs.

## What it does

1. Samples frames across the whole video (no quarter detection).
2. Tracks people with YOLO + ByteTrack.
3. Detects the ball primarily with orange color + circularity (YOLO `sports ball` is unreliable on wide 4K gym shots).
4. Estimates rim location in the far-court region (orange rim), with a high-ball fallback.
5. Clusters jersey colors with K-means (k=2) → Team A / Team B.
6. Infers shot intervals when the ball enters the rim neighborhood and writes `output/shots.csv`.

## Limitations

These are **heuristics**, not a production shot tracker:

- Stock COCO YOLO rarely sees tiny basketballs in wide gym video; color fallback helps but can also pick up orange noise.
- Made / missed labels can be wrong under occlusions, glare behind the glass, or odd camera angles.
- Team labels depend on distinct jersey colors; similar kits may collapse to one cluster.
- Always review `shots.csv` (and optionally `tracks_sample.csv`) before using the times for clipping.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads YOLO weights (default `yolov8n.pt`).

## Usage

### Scan a video

Place a video in `data/` (e.g. `data/compressed_FAAM_Video_2.mp4`), then:

```bash
python main.py scan
```

Or pass an explicit path:

```bash
python main.py scan /path/to/game.mp4
```

Useful options:

| Option | Default | Meaning |
|--------|---------|---------|
| `--model` | `yolov8n.pt` | Ultralytics weights |
| `--conf` | `0.25` | Confidence threshold |
| `--sample-interval` | `0.25` | Seconds between sampled frames |
| `--merge-gap` | `1.0` | Merge nearby shot intervals |
| `--pre-seconds` | `1.0` | Seconds before release in each event |
| `--post-seconds` | `1.5` | Seconds after release in each event |
| `--output-dir` | `output` | Where CSVs are written |

Example:

```bash
python main.py scan game.mp4 --model yolov8n.pt --conf 0.25 --sample-interval 0.25 --output-dir output
```

### Outputs

**`output/shots.csv`** (primary):

| Column | Description |
|--------|-------------|
| `shot_number` | 1-based index |
| `start_time_seconds` | Interval start |
| `end_time_seconds` | Interval end |
| `outcome` | `made`, `missed`, or `unknown` |
| `team` | `Team A`, `Team B`, or `unknown` |

**`output/tracks_sample.csv`** (debug): sparse track points (`time_seconds,object,track_id,team,cx,cy`).

### Team labels

Jersey colors are sampled from the upper torso of each person box, then clustered with k=2. **Team A** is the cluster with the lower median hue; **Team B** is the other. Tracks without enough color samples stay `unknown`.

### Inspecting / correcting later

1. Open `tracks_sample.csv` to see whether the ball and players were detected.
2. Edit `shots.csv` times / `outcome` / `team` by hand if needed.
3. Clip export can be added later from those confirmed times.

## Project layout

- `main.py` — CLI (`scan`)
- `detector.py` — video IO, YOLO track, rim estimate
- `teams.py` — jersey color clustering
- `shots.py` — shot-event heuristics + CSV writer
- `requirements.txt`
- `README.md`
