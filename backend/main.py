"""FastAPI backend: upload, process (background), status, metadata, media, export,
and manual team/jersey correction.

Jobs run in a thread pool; status/progress is tracked in-memory and persisted to
each job directory. No database or auth (MVP) — everything lives on disk under
outputs/<job_id>/.
"""

from __future__ import annotations

import json
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .config import OUTPUT_DIR, UPLOAD_DIR, VIDEO_EXTENSIONS, ProcessConfig
from .pipeline import Pipeline

app = FastAPI(title="AI Basketball Analytics", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=2)
_jobs: Dict[str, dict] = {}
_lock = threading.Lock()


def _job_dir(job_id: str) -> Path:
    return OUTPUT_DIR / job_id


def _set(job_id: str, **kw):
    with _lock:
        _jobs.setdefault(job_id, {}).update(kw)
        (_job_dir(job_id)).mkdir(parents=True, exist_ok=True)
        try:
            (_job_dir(job_id) / "status.json").write_text(json.dumps(_jobs[job_id]))
        except Exception:
            pass


def _get(job_id: str) -> Optional[dict]:
    with _lock:
        if job_id in _jobs:
            return dict(_jobs[job_id])
    sp = _job_dir(job_id) / "status.json"
    if sp.exists():
        try:
            return json.loads(sp.read_text())
        except Exception:
            return None
    return None


class ProcessRequest(BaseModel):
    stride: int = 1
    imgsz: int = 1280
    conf: float = 0.25
    max_seconds: float = 0.0
    render: bool = True
    ocr: bool = True


class Correction(BaseModel):
    track_id: int
    team: Optional[str] = None
    jersey_number: Optional[str] = None


def _run_job(job_id: str, video: Path, mode: str, cfg: ProcessConfig):
    def progress(f):
        _set(job_id, state="processing", progress=round(float(f), 3))
    try:
        _set(job_id, state="processing", progress=0.0, error=None)
        pipe = Pipeline(cfg)
        result = pipe.run(video, _job_dir(job_id), mode=mode, progress=progress)
        _set(job_id, state="done", progress=1.0,
             shots=len(result.shots), players=len(result.players),
             meta=result.meta)
    except Exception as e:  # noqa
        import traceback
        _set(job_id, state="error", error=str(e), trace=traceback.format_exc()[-2000:])


@app.get("/api/health")
def health():
    return {"ok": True, "device": ProcessConfig().device}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}")
    job_id = uuid.uuid4().hex[:12]
    jd = _job_dir(job_id)
    jd.mkdir(parents=True, exist_ok=True)
    dest = jd / f"input{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    _set(job_id, state="uploaded", progress=0.0, source=file.filename, input=dest.name)
    return {"job_id": job_id, "state": "uploaded", "filename": file.filename}


@app.post("/api/process/{job_id}")
def process(job_id: str, req: ProcessRequest):
    st = _get(job_id)
    if not st:
        raise HTTPException(404, "job not found")
    jd = _job_dir(job_id)
    inputs = list(jd.glob("input.*"))
    if not inputs:
        raise HTTPException(400, "no input video for this job")
    cfg = ProcessConfig(frame_stride=max(1, req.stride), imgsz=req.imgsz,
                        conf=req.conf, max_seconds=req.max_seconds,
                        render_video=req.render, enable_ocr=req.ocr)
    _set(job_id, state="queued", progress=0.0)
    _executor.submit(_run_job, job_id, inputs[0], "real", cfg)
    return {"job_id": job_id, "state": "queued"}


@app.post("/api/demo")
def demo():
    job_id = "demo_" + uuid.uuid4().hex[:8]
    _set(job_id, state="queued", progress=0.0, source="synthetic demo")
    _executor.submit(_run_job, job_id, Path("unused"), "demo", ProcessConfig())
    return {"job_id": job_id, "state": "queued"}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    st = _get(job_id)
    if not st:
        raise HTTPException(404, "job not found")
    return {k: v for k, v in st.items() if k != "trace"}


@app.get("/api/analysis/{job_id}")
def analysis(job_id: str):
    p = _job_dir(job_id) / "analysis.json"
    if not p.exists():
        raise HTTPException(404, "analysis not ready")
    return JSONResponse(json.loads(p.read_text()))


@app.get("/api/jobs")
def jobs():
    out = []
    for d in sorted(OUTPUT_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        st = _get(d.name) or {}
        out.append({"job_id": d.name, "state": st.get("state", "unknown"),
                    "source": st.get("source"), "shots": st.get("shots"),
                    "has_analysis": (d / "analysis.json").exists()})
    return out


@app.get("/api/media/{job_id}/{name:path}")
def media(job_id: str, name: str):
    jd = _job_dir(job_id).resolve()
    target = (jd / name).resolve()
    if not str(target).startswith(str(jd)) or not target.exists():
        raise HTTPException(404, "not found")
    return FileResponse(str(target))


@app.get("/api/export/{job_id}/{kind}")
def export(job_id: str, kind: str):
    jd = _job_dir(job_id)
    mapping = {"json": "analysis.json", "shots": "shots.csv",
               "players": "players.csv", "video": "annotated_video.mp4"}
    fn = mapping.get(kind)
    if not fn:
        raise HTTPException(400, "unknown export kind")
    p = jd / fn
    if not p.exists():
        raise HTTPException(404, "not available")
    return FileResponse(str(p), filename=f"{job_id}_{fn}")


@app.post("/api/correct/{job_id}")
def correct(job_id: str, c: Correction):
    """Apply a manual team/jersey correction to a track and persist it."""
    p = _job_dir(job_id) / "analysis.json"
    if not p.exists():
        raise HTTPException(404, "analysis not ready")
    data = json.loads(p.read_text())
    changed = False
    for pl in data.get("players", []):
        if pl["track_id"] == c.track_id:
            if c.team is not None:
                pl["team"] = c.team
                pl["team_confidence"] = 1.0
            if c.jersey_number is not None:
                pl["jersey_number"] = c.jersey_number
                pl["jersey_confidence"] = 1.0
            changed = True
    # Propagate to per-frame persons and shots.
    for fr in data.get("frames", []):
        for pn in fr.get("persons", []):
            if pn["track_id"] == c.track_id:
                if c.team is not None:
                    pn["team"] = c.team
                if c.jersey_number is not None:
                    pn["jersey_number"] = c.jersey_number
    for s in data.get("shots", []):
        if s.get("shooter_track_id") == c.track_id:
            if c.team is not None:
                s["team"] = c.team
            if c.jersey_number is not None:
                s["jersey_number"] = c.jersey_number
    if not changed:
        raise HTTPException(404, "track not found")
    p.write_text(json.dumps(data))
    return {"ok": True, "track_id": c.track_id}


# Serve the built frontend if present (single-command deploy).
_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="static")
