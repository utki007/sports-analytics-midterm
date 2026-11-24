import type { Analysis } from "./types";

const BASE = "/api";

export async function health() {
  const r = await fetch(`${BASE}/health`);
  return r.json();
}

export async function uploadVideo(file: File): Promise<{ job_id: string }> {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${BASE}/upload`, { method: "POST", body: fd });
  if (!r.ok) throw new Error((await r.json()).detail || "upload failed");
  return r.json();
}

export async function startProcess(jobId: string, opts: any) {
  const r = await fetch(`${BASE}/process/${jobId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(opts),
  });
  if (!r.ok) throw new Error((await r.json()).detail || "process failed");
  return r.json();
}

export async function startDemo(): Promise<{ job_id: string }> {
  const r = await fetch(`${BASE}/demo`, { method: "POST" });
  return r.json();
}

export async function getStatus(jobId: string) {
  const r = await fetch(`${BASE}/status/${jobId}`);
  if (!r.ok) throw new Error("status failed");
  return r.json();
}

export async function getAnalysis(jobId: string): Promise<Analysis> {
  const r = await fetch(`${BASE}/analysis/${jobId}`);
  if (!r.ok) throw new Error("analysis not ready");
  return r.json();
}

export async function correctTrack(
  jobId: string,
  track_id: number,
  patch: { team?: string; jersey_number?: string }
) {
  const r = await fetch(`${BASE}/correct/${jobId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ track_id, ...patch }),
  });
  return r.json();
}

export function mediaUrl(jobId: string, name: string) {
  return `${BASE}/media/${jobId}/${name}`;
}

export function exportUrl(jobId: string, kind: string) {
  return `${BASE}/export/${jobId}/${kind}`;
}
