import {
  CartesianGrid, Legend, Line, LineChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import type { Shot } from "../lib/types";

// Merge observed / fitted / ideal paths (image px) into one dataset keyed by x.
// y is flipped to "height" (frameH - y) so the arc reads naturally.
function buildData(shot: Shot, frameH: number) {
  const rows: Record<number, any> = {};
  const add = (path: [number, number][], key: string) => {
    for (const [x, y] of path) {
      const xr = Math.round(x);
      rows[xr] = rows[xr] || { x: xr };
      rows[xr][key] = Math.round(frameH - y);
    }
  };
  const ph = shot.physics;
  if (!ph) return [];
  add(ph.observed_path, "observed");
  add(ph.fitted_path, "fitted");
  add(ph.ideal_path, "ideal");
  return Object.values(rows).sort((a: any, b: any) => a.x - b.x);
}

export default function TrajectoryChart({ shot, frameH }: { shot: Shot; frameH: number }) {
  const data = buildData(shot, frameH);
  if (!data.length) return <div className="text-sm text-slate-400">No trajectory data.</div>;
  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={data} margin={{ top: 10, right: 20, bottom: 4, left: -10 }}>
        <CartesianGrid stroke="#263042" strokeDasharray="3 3" />
        <XAxis dataKey="x" stroke="#8aa" tick={{ fontSize: 11 }}
          label={{ value: "x (px)", position: "insideBottom", fill: "#8aa", fontSize: 11, dy: 10 }} />
        <YAxis stroke="#8aa" tick={{ fontSize: 11 }}
          label={{ value: "height (px)", angle: -90, position: "insideLeft", fill: "#8aa", fontSize: 11 }} />
        <Tooltip contentStyle={{ background: "#141922", border: "1px solid #263042", fontSize: 12 }} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Line type="monotone" dataKey="observed" stroke="#ff8c00" dot={{ r: 2 }}
          strokeWidth={2} connectNulls name="Observed" isAnimationActive={false} />
        <Line type="monotone" dataKey="fitted" stroke="#ffd24d" strokeWidth={2}
          dot={false} connectNulls name="2D fit" isAnimationActive={false} />
        <Line type="monotone" dataKey="ideal" stroke="#78ff78" strokeDasharray="6 4"
          strokeWidth={2} dot={false} connectNulls name="Ideal arc" isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}
