import { useEffect, useRef } from "react";
import type { Analysis, FrameData, Shot } from "../lib/types";
import { TEAM_HEX } from "../lib/types";

export interface Toggles {
  players: boolean;
  labels: boolean;
  ball: boolean;
  trails: boolean;
  arcs: boolean;
  teamOverlay: boolean;
  rim: boolean;
}

interface Props {
  src: string;
  analysis: Analysis;
  toggles: Toggles;
  activeShot: Shot | null;
  videoRef: React.RefObject<HTMLVideoElement>;
  onTime: (t: number) => void;
}

// Find the frame whose timestamp is closest to t (binary search).
function nearestFrame(frames: FrameData[], t: number): FrameData | null {
  if (!frames.length) return null;
  let lo = 0, hi = frames.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].t < t) lo = mid + 1;
    else hi = mid;
  }
  const a = frames[Math.max(0, lo - 1)], b = frames[lo];
  return Math.abs(a.t - t) <= Math.abs(b.t - t) ? a : b;
}

export default function VideoStage({ src, analysis, toggles, activeShot, videoRef, onTime }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const trailRef = useRef<Map<number, [number, number][]>>(new Map());
  const ballTrailRef = useRef<[number, number][]>([]);
  const rafRef = useRef<number>(0);

  const W = analysis.meta.width;
  const H = analysis.meta.height;

  useEffect(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const draw = () => {
      const t = video.currentTime;
      onTime(t);
      const rect = video.getBoundingClientRect();
      // Keep canvas pixel size aligned to display size for crisp 1:1 mapping.
      if (canvas.width !== rect.width || canvas.height !== rect.height) {
        canvas.width = rect.width;
        canvas.height = rect.height;
      }
      const sx = rect.width / W;
      const sy = rect.height / H;
      const ctx = canvas.getContext("2d")!;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const fr = nearestFrame(analysis.frames, t);
      if (fr) {
        // Rim
        if (toggles.rim && fr.rim) {
          ctx.strokeStyle = "#ffa500";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.ellipse(fr.rim.cx * sx, fr.rim.cy * sy,
            Math.max(6, (fr.rim.w / 2) * sx), Math.max(3, (fr.rim.h / 2) * sy),
            0, 0, Math.PI * 2);
          ctx.stroke();
        }
        // Players
        if (toggles.players) {
          for (const p of fr.persons) {
            const color = toggles.teamOverlay ? (TEAM_HEX[p.team] || TEAM_HEX.unknown) : "#9fb0c3";
            const [x1, y1, x2, y2] = p.bbox;
            const strong = p.role === "player" && (p.team === "Team A" || p.team === "Team B");
            ctx.strokeStyle = color;
            ctx.lineWidth = strong ? 2 : 1;
            ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
            if (toggles.trails) {
              const cx = ((x1 + x2) / 2) * sx, cy = y2 * sy;
              const arr = trailRef.current.get(p.track_id) || [];
              arr.push([cx, cy]);
              if (arr.length > 24) arr.shift();
              trailRef.current.set(p.track_id, arr);
              ctx.strokeStyle = color + "88";
              ctx.lineWidth = 1.5;
              ctx.beginPath();
              arr.forEach(([px, py], i) => (i ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
              ctx.stroke();
            }
            if (toggles.labels) {
              const label = p.jersey_number ? `${p.jersey_number} (#${p.track_id})` : `#${p.track_id}`;
              ctx.font = "11px ui-sans-serif";
              const tw = ctx.measureText(label).width;
              ctx.fillStyle = color;
              ctx.fillRect(x1 * sx, y1 * sy - 14, tw + 6, 14);
              ctx.fillStyle = "#111";
              ctx.fillText(label, x1 * sx + 3, y1 * sy - 3);
            }
          }
        }
        // Ball + trail
        if (toggles.ball && fr.ball) {
          const bx = fr.ball.cx * sx, by = fr.ball.cy * sy;
          if (toggles.trails) {
            ballTrailRef.current.push([bx, by]);
            if (ballTrailRef.current.length > 20) ballTrailRef.current.shift();
            ctx.strokeStyle = "#ffcf4d";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ballTrailRef.current.forEach(([px, py], i) => (i ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
            ctx.stroke();
          }
          ctx.beginPath();
          ctx.arc(bx, by, Math.max(5, fr.ball.r * sx), 0, Math.PI * 2);
          ctx.strokeStyle = fr.ball.observed ? "#ff8c00" : "#888";
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }

      // Shot trajectory arcs
      if (toggles.arcs && activeShot && activeShot.physics) {
        const ph = activeShot.physics;
        const obs = ph.fitted_path.length ? ph.fitted_path : ph.observed_path;
        if (obs.length > 1) {
          ctx.strokeStyle = "#ffd24d";
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          obs.forEach(([x, y], i) => (i ? ctx.lineTo(x * sx, y * sy) : ctx.moveTo(x * sx, y * sy)));
          ctx.stroke();
        }
        if (ph.calibrated && ph.ideal_path.length > 1) {
          ctx.strokeStyle = "#78ff78";
          ctx.lineWidth = 2;
          ctx.setLineDash([6, 5]);
          ctx.beginPath();
          ph.ideal_path.forEach(([x, y], i) => (i ? ctx.lineTo(x * sx, y * sy) : ctx.moveTo(x * sx, y * sy)));
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }

      rafRef.current = requestAnimationFrame(draw);
    };
    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [analysis, toggles, activeShot]);

  return (
    <div className="relative w-full bg-black rounded-lg overflow-hidden">
      <video ref={videoRef} src={src} controls className="w-full block"
        style={{ aspectRatio: `${W} / ${H}` }} />
      <canvas ref={canvasRef}
        className="absolute inset-0 pointer-events-none w-full h-full" />
    </div>
  );
}
