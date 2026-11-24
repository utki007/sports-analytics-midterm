import type { Shot } from "../lib/types";
import { TEAM_HEX } from "../lib/types";
import TrajectoryChart from "./TrajectoryChart";

interface Props {
  shots: Shot[];
  selected: Shot | null;
  onSelect: (s: Shot) => void;
  frameH: number;
  jobId: string;
  mediaUrl: (job: string, name: string) => string;
}

const OUTCOME_STYLE: Record<string, string> = {
  made: "bg-emerald-500/20 text-emerald-300",
  missed: "bg-red-500/20 text-red-300",
  unknown: "bg-slate-500/20 text-slate-300",
};

function Stat({ label, value, unit, sd }: { label: string; value: any; unit?: string; sd?: number }) {
  return (
    <div className="bg-panel2 rounded-md px-3 py-2">
      <div className="text-[11px] text-slate-400">{label}</div>
      <div className="text-slate-100 font-semibold">
        {value == null ? "—" : value}{value != null && unit ? ` ${unit}` : ""}
        {sd != null && value != null ? <span className="text-[11px] text-slate-500"> ±{sd}</span> : null}
      </div>
    </div>
  );
}

export default function ShotPanel({ shots, selected, onSelect, frameH, jobId, mediaUrl }: Props) {
  return (
    <div className="space-y-3">
      <h3 className="font-semibold text-slate-200">Shots ({shots.length})</h3>
      <div className="rounded-md border border-slate-700 overflow-auto max-h-48">
        <table className="w-full text-sm">
          <thead className="bg-panel2 text-slate-400 text-xs sticky top-0">
            <tr>
              <th className="text-left px-2 py-1">#</th>
              <th className="text-left px-2 py-1">Time</th>
              <th className="text-left px-2 py-1">Team</th>
              <th className="text-left px-2 py-1">Shooter</th>
              <th className="text-left px-2 py-1">Outcome</th>
              <th className="text-left px-2 py-1">Conf</th>
            </tr>
          </thead>
          <tbody>
            {shots.map((s) => (
              <tr key={s.id}
                className={`border-t border-slate-800 cursor-pointer ${selected?.id === s.id ? "bg-sky-500/10" : "hover:bg-panel2/50"}`}
                onClick={() => onSelect(s)}>
                <td className="px-2 py-1">{s.id}</td>
                <td className="px-2 py-1 font-mono">{s.t_release.toFixed(1)}s</td>
                <td className="px-2 py-1">
                  <span className="w-2.5 h-2.5 rounded-full inline-block mr-1"
                    style={{ background: TEAM_HEX[s.team] || "#aaa" }} />
                  {s.team}
                </td>
                <td className="px-2 py-1">{s.shooter_track_id ? `#${s.shooter_track_id}` : "—"}</td>
                <td className="px-2 py-1">
                  <span className={`px-1.5 py-0.5 rounded text-xs ${OUTCOME_STYLE[s.outcome]}`}>
                    {s.kind === "pass?" ? "ball event" : s.outcome}
                  </span>
                </td>
                <td className="px-2 py-1 text-slate-400">{(s.outcome_confidence * 100).toFixed(0)}%</td>
              </tr>
            ))}
            {!shots.length && <tr><td colSpan={6} className="px-2 py-3 text-slate-500 text-center">No shots detected.</td></tr>}
          </tbody>
        </table>
      </div>

      {selected && (
        <div className="bg-panel rounded-lg p-3 space-y-3 border border-slate-700">
          <div className="flex items-center justify-between">
            <div className="font-semibold text-slate-100">Shot {selected.id} detail</div>
            {selected.clip && (
              <a href={mediaUrl(jobId, selected.clip)} target="_blank" rel="noreferrer"
                className="text-xs text-sky-400 underline">open clip</a>
            )}
          </div>

          {selected.physics ? (
            <>
              <TrajectoryChart shot={selected} frameH={frameH} />
              {selected.physics.calibrated ? (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                  <Stat label="Launch angle" value={selected.physics.launch_angle_deg} unit="°"
                    sd={selected.physics.uncertainty?.launch_angle_deg} />
                  <Stat label="Release speed" value={selected.physics.release_speed_mps} unit="m/s"
                    sd={selected.physics.uncertainty?.release_speed_mps} />
                  <Stat label="Apex (rel.)" value={selected.physics.apex_height_m} unit="m"
                    sd={selected.physics.uncertainty?.apex_height_m} />
                  <Stat label="Entry angle" value={selected.physics.entry_angle_deg} unit="°" />
                  <Stat label="Ideal launch" value={selected.physics.ideal_launch_angle_deg} unit="°" />
                  <Stat label="Ideal speed" value={selected.physics.ideal_release_speed_mps} unit="m/s" />
                  <Stat label="Lateral miss" value={selected.physics.miss_distance_m} unit="m" />
                  <Stat label="Confidence" value={(selected.physics.confidence * 100).toFixed(0)} unit="%" />
                </div>
              ) : (
                <div className="text-sm text-amber-300/90 bg-amber-500/10 rounded-md px-3 py-2">
                  Uncalibrated: 3D physics not identifiable from this clip. Showing the
                  observed 2D arc only.
                </div>
              )}
              <div className="text-sm">
                <span className="text-slate-400">Feedback: </span>
                <span className="text-slate-100">{selected.physics.feedback}</span>
              </div>
              {selected.physics.assumptions?.length > 0 && (
                <details className="text-xs text-slate-400">
                  <summary className="cursor-pointer">modeling assumptions &amp; limitations</summary>
                  <ul className="list-disc pl-5 mt-1 space-y-0.5">
                    {selected.physics.assumptions.map((a, i) => <li key={i}>{a}</li>)}
                  </ul>
                </details>
              )}
            </>
          ) : (
            <div className="text-sm text-slate-400">No physics available for this shot.</div>
          )}
        </div>
      )}
    </div>
  );
}
