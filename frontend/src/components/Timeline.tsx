import type { Shot } from "../lib/types";
import { TEAM_HEX } from "../lib/types";

interface Props {
  duration: number;
  current: number;
  shots: Shot[];
  onSeek: (t: number) => void;
}

const OUTCOME_GLYPH: Record<string, string> = { made: "●", missed: "✕", unknown: "?" };

export default function Timeline({ duration, current, shots, onSeek }: Props) {
  const dur = duration || 1;
  return (
    <div className="w-full">
      <div className="relative h-10 bg-panel2 rounded-md cursor-pointer"
        onClick={(e) => {
          const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
          onSeek(((e.clientX - r.left) / r.width) * dur);
        }}>
        {/* progress */}
        <div className="absolute top-0 bottom-0 bg-sky-500/20"
          style={{ width: `${(current / dur) * 100}%` }} />
        {/* playhead */}
        <div className="absolute top-0 bottom-0 w-0.5 bg-sky-400"
          style={{ left: `${(current / dur) * 100}%` }} />
        {/* shot markers */}
        {shots.map((s) => (
          <button key={s.id} title={`Shot ${s.id} — ${s.outcome} (${s.team})`}
            onClick={(e) => { e.stopPropagation(); onSeek(Math.max(0, s.t_release - 0.3)); }}
            className="absolute -top-1 flex flex-col items-center -translate-x-1/2"
            style={{ left: `${(s.t_release / dur) * 100}%` }}>
            <span className="text-xs leading-none"
              style={{ color: TEAM_HEX[s.team] || "#aaa" }}>
              {OUTCOME_GLYPH[s.outcome] || "?"}
            </span>
          </button>
        ))}
      </div>
      <div className="flex justify-between text-[11px] text-slate-500 mt-1">
        <span>0:00</span>
        <span>{current.toFixed(1)}s</span>
        <span>{dur.toFixed(1)}s</span>
      </div>
    </div>
  );
}
