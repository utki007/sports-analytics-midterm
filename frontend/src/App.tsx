import { useEffect, useMemo, useRef, useState } from "react";
import * as api from "./lib/api";
import type { Analysis, Shot } from "./lib/types";
import VideoStage, { Toggles } from "./components/VideoStage";
import Timeline from "./components/Timeline";
import PlayerTable from "./components/PlayerTable";
import ShotPanel from "./components/ShotPanel";

const DEFAULT_TOGGLES: Toggles = {
  players: true, labels: true, ball: true, trails: true,
  arcs: true, teamOverlay: true, rim: true,
};

export default function App() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [state, setState] = useState<string>("idle");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [toggles, setToggles] = useState<Toggles>(DEFAULT_TOGGLES);
  const [current, setCurrent] = useState(0);
  const [selectedShot, setSelectedShot] = useState<Shot | null>(null);
  const [view, setView] = useState<"overlay" | "annotated">("overlay");
  const [opts, setOpts] = useState({ stride: 2, imgsz: 960, max_seconds: 30, ocr: false });
  const videoRef = useRef<HTMLVideoElement>(null);

  // Poll status until done/error, then load analysis.
  useEffect(() => {
    if (!jobId || state === "done" || state === "error") return;
    const id = setInterval(async () => {
      try {
        const st = await api.getStatus(jobId);
        setState(st.state);
        setProgress(st.progress || 0);
        if (st.state === "done") {
          const a = await api.getAnalysis(jobId);
          setAnalysis(a);
          clearInterval(id);
        } else if (st.state === "error") {
          setError(st.error || "processing failed");
          clearInterval(id);
        }
      } catch (e: any) { /* keep polling */ }
    }, 1000);
    return () => clearInterval(id);
  }, [jobId, state]);

  const duration = analysis?.meta?.duration || 0;
  const frameH = analysis?.meta?.height || 720;

  const activeShot = useMemo(() => {
    if (!analysis) return null;
    return analysis.shots.find(
      (s) => current >= s.t_start - 0.3 && current <= s.t_end + 0.6
    ) || null;
  }, [analysis, current]);

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setError(null); setAnalysis(null); setState("uploading");
    try {
      const { job_id } = await api.uploadVideo(file);
      await api.startProcess(job_id, { ...opts, render: true });
      setJobId(job_id); setState("queued");
    } catch (err: any) { setError(err.message); setState("error"); }
  }

  async function handleDemo() {
    setError(null); setAnalysis(null); setState("queued");
    const { job_id } = await api.startDemo();
    setJobId(job_id);
  }

  async function handleCorrect(track_id: number, patch: any) {
    if (!jobId) return;
    await api.correctTrack(jobId, track_id, patch);
    const a = await api.getAnalysis(jobId);
    setAnalysis(a);
  }

  function seek(t: number) {
    if (videoRef.current) { videoRef.current.currentTime = t; setCurrent(t); }
  }

  const origUrl = analysis && jobId ? api.mediaUrl(jobId, analysis.meta.source) : "";
  const annotUrl = analysis && jobId && analysis.meta.annotated_video
    ? api.mediaUrl(jobId, analysis.meta.annotated_video) : "";

  const busy = ["uploading", "queued", "processing"].includes(state);

  return (
    <div className="min-h-screen max-w-7xl mx-auto px-4 py-5">
      <header className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div>
          <h1 className="text-xl font-bold text-slate-100">🏀 AI Basketball Analytics</h1>
          <p className="text-xs text-slate-400">
            Player tracking · team ID · ball &amp; shot detection · trajectory physics
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label className="px-3 py-2 bg-sky-600 hover:bg-sky-500 rounded-md text-sm cursor-pointer">
            Upload MP4
            <input type="file" accept="video/*" className="hidden" onChange={handleUpload} />
          </label>
          <button onClick={handleDemo}
            className="px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded-md text-sm">
            Run synthetic demo
          </button>
        </div>
      </header>

      {/* processing options */}
      <div className="bg-panel rounded-lg p-3 mb-4 flex flex-wrap gap-4 items-end border border-slate-800">
        <Opt label="Frame stride" value={opts.stride}
          onChange={(v) => setOpts({ ...opts, stride: v })} min={1} max={5} />
        <Opt label="Inference size" value={opts.imgsz}
          onChange={(v) => setOpts({ ...opts, imgsz: v })} min={640} max={1280} step={320} />
        <Opt label="Max seconds" value={opts.max_seconds}
          onChange={(v) => setOpts({ ...opts, max_seconds: v })} min={0} max={120} step={5} />
        <label className="text-xs text-slate-400 flex items-center gap-1">
          <input type="checkbox" checked={opts.ocr}
            onChange={(e) => setOpts({ ...opts, ocr: e.target.checked })} />
          jersey OCR (if installed)
        </label>
        <span className="text-xs text-slate-500">Options apply to the next upload.</span>
      </div>

      {error && (
        <div className="bg-red-500/15 text-red-300 rounded-md px-3 py-2 mb-4 text-sm">
          Error: {error}
        </div>
      )}

      {busy && (
        <div className="bg-panel rounded-lg p-4 mb-4 border border-slate-800">
          <div className="text-sm text-slate-300 mb-2 capitalize">{state}…</div>
          <div className="h-2 bg-panel2 rounded">
            <div className="h-2 bg-sky-500 rounded transition-all"
              style={{ width: `${Math.round(progress * 100)}%` }} />
          </div>
        </div>
      )}

      {!analysis && !busy && (
        <div className="bg-panel rounded-lg p-10 text-center text-slate-400 border border-slate-800">
          Upload a basketball clip or run the synthetic demo to begin.
        </div>
      )}

      {analysis && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* left: video + timeline + toggles */}
          <div className="lg:col-span-2 space-y-3">
            {analysis.meta.mode === "demo" && (
              <div className="text-xs bg-amber-500/15 text-amber-300 rounded px-2 py-1 inline-block">
                SYNTHETIC DEMO — not real inference
              </div>
            )}
            <div className="flex gap-2 text-xs">
              <button onClick={() => setView("overlay")}
                className={`px-2 py-1 rounded ${view === "overlay" ? "bg-sky-600" : "bg-slate-700"}`}>
                Original + live overlays
              </button>
              {annotUrl && (
                <button onClick={() => setView("annotated")}
                  className={`px-2 py-1 rounded ${view === "annotated" ? "bg-sky-600" : "bg-slate-700"}`}>
                  Rendered annotated MP4
                </button>
              )}
            </div>

            {view === "overlay" ? (
              <VideoStage src={origUrl} analysis={analysis} toggles={toggles}
                activeShot={activeShot} videoRef={videoRef} onTime={setCurrent} />
            ) : (
              <video src={annotUrl} controls className="w-full rounded-lg bg-black"
                ref={videoRef} onTimeUpdate={(e) => setCurrent((e.target as HTMLVideoElement).currentTime)} />
            )}

            <Timeline duration={duration} current={current} shots={analysis.shots} onSeek={seek} />

            {view === "overlay" && (
              <div className="flex flex-wrap gap-3 bg-panel rounded-lg p-3 border border-slate-800">
                {Object.keys(toggles).map((k) => (
                  <label key={k} className="text-xs text-slate-300 flex items-center gap-1 capitalize">
                    <input type="checkbox" checked={(toggles as any)[k]}
                      onChange={(e) => setToggles({ ...toggles, [k]: e.target.checked })} />
                    {k === "teamOverlay" ? "team colors" : k}
                  </label>
                ))}
              </div>
            )}

            {/* exports */}
            <div className="flex flex-wrap gap-2 text-xs">
              {jobId && (["video", "json", "shots", "players"] as const).map((k) => (
                <a key={k} href={api.exportUrl(jobId, k)}
                  className="px-2 py-1 bg-slate-700 hover:bg-slate-600 rounded">
                  export {k === "video" ? "annotated video" : k + " CSV/JSON"}
                </a>
              ))}
            </div>

            <MetaBar analysis={analysis} />
          </div>

          {/* right: roster + shots */}
          <div className="space-y-4">
            <div className="bg-panel rounded-lg p-3 border border-slate-800">
              <PlayerTable players={analysis.players} onCorrect={handleCorrect} />
            </div>
            <div className="bg-panel rounded-lg p-3 border border-slate-800">
              <ShotPanel shots={analysis.shots} selected={selectedShot}
                onSelect={(s) => { setSelectedShot(s); seek(Math.max(0, s.t_release - 0.3)); }}
                frameH={frameH} jobId={jobId!} mediaUrl={api.mediaUrl} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function Opt({ label, value, onChange, min, max, step = 1 }:
  { label: string; value: number; onChange: (v: number) => void; min: number; max: number; step?: number }) {
  return (
    <label className="text-xs text-slate-400 flex flex-col gap-1">
      {label}: <span className="text-slate-200">{value || "all"}</span>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))} className="w-32" />
    </label>
  );
}

function MetaBar({ analysis }: { analysis: Analysis }) {
  const m = analysis.meta;
  const nSig = analysis.players.filter((p) => p.significant).length;
  return (
    <div className="text-[11px] text-slate-500 grid grid-cols-2 sm:grid-cols-4 gap-2 bg-panel2 rounded-md p-2">
      <span>mode: <b className="text-slate-300">{m.mode}</b></span>
      <span>device: {m.device}</span>
      <span>model: {m.model}</span>
      <span>frames: {m.processed_frames}</span>
      <span>fps: {Number(m.fps).toFixed(1)}</span>
      <span>rim: {m.rim_detected ? `yes (${(m.rim_confidence * 100).toFixed(0)}%)` : "no"}</span>
      <span>players: {nSig} on-court</span>
      <span>OCR: {m.ocr_backend}</span>
    </div>
  );
}
