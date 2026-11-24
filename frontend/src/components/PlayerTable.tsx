import { useState } from "react";
import type { PlayerTrack } from "../lib/types";
import { TEAM_HEX } from "../lib/types";

interface Props {
  players: PlayerTrack[];
  onCorrect: (track_id: number, patch: { team?: string; jersey_number?: string }) => void;
}

const TEAMS = ["Team A", "Team B", "referee", "unknown"];

export default function PlayerTable({ players, onCorrect }: Props) {
  const [showAll, setShowAll] = useState(false);
  const [edit, setEdit] = useState<number | null>(null);
  const [jersey, setJersey] = useState("");

  const rows = showAll ? players : players.filter((p) => p.significant);

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <h3 className="font-semibold text-slate-200">Players</h3>
        <label className="text-xs text-slate-400 flex items-center gap-1">
          <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} />
          show all tracks ({players.length})
        </label>
      </div>
      <div className="max-h-72 overflow-auto rounded-md border border-slate-700">
        <table className="w-full text-sm">
          <thead className="bg-panel2 text-slate-400 text-xs sticky top-0">
            <tr>
              <th className="text-left px-2 py-1">ID</th>
              <th className="text-left px-2 py-1">Team</th>
              <th className="text-left px-2 py-1">#</th>
              <th className="text-left px-2 py-1">Conf</th>
              <th className="px-2 py-1"></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((p) => (
              <tr key={p.track_id} className="border-t border-slate-800 hover:bg-panel2/50">
                <td className="px-2 py-1 font-mono">#{p.track_id}</td>
                <td className="px-2 py-1">
                  <span className="inline-flex items-center gap-1">
                    <span className="w-2.5 h-2.5 rounded-full"
                      style={{ background: TEAM_HEX[p.team] || "#aaa" }} />
                    {p.team}
                  </span>
                </td>
                <td className="px-2 py-1">{p.jersey_number || "?"}</td>
                <td className="px-2 py-1 text-slate-400">{(p.team_confidence * 100).toFixed(0)}%</td>
                <td className="px-2 py-1 text-right">
                  {edit === p.track_id ? (
                    <div className="flex items-center gap-1 justify-end">
                      <input value={jersey} onChange={(e) => setJersey(e.target.value)}
                        placeholder="#" className="w-10 bg-slate-900 border border-slate-700 rounded px-1" />
                      <select defaultValue={p.team} id={`team-${p.track_id}`}
                        className="bg-slate-900 border border-slate-700 rounded px-1 text-xs">
                        {TEAMS.map((t) => <option key={t} value={t}>{t}</option>)}
                      </select>
                      <button className="text-emerald-400"
                        onClick={() => {
                          const sel = (document.getElementById(`team-${p.track_id}`) as HTMLSelectElement).value;
                          onCorrect(p.track_id, { team: sel, jersey_number: jersey || undefined });
                          setEdit(null); setJersey("");
                        }}>✓</button>
                      <button className="text-slate-400" onClick={() => setEdit(null)}>✕</button>
                    </div>
                  ) : (
                    <button className="text-sky-400 text-xs"
                      onClick={() => { setEdit(p.track_id); setJersey(p.jersey_number || ""); }}>
                      edit
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
