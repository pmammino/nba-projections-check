"use client";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export interface Series { key: string; label: string }
const COLORS = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)", "var(--c5)", "var(--c6)", "var(--c7)"];

/** data: one object per x value, with the x label in `x` and one numeric field per series. */
export default function EquityChart({ data, series, yLabel = "units" }: { data: Record<string, number | string | null>[]; series: Series[]; yLabel?: string }) {
  if (!data.length) return <div className="empty">Nothing settled yet.</div>;
  return (
    <div className="chart">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
          <XAxis dataKey="x" tick={{ fill: "var(--muted)", fontSize: 12 }} minTickGap={24} />
          <YAxis tick={{ fill: "var(--muted)", fontSize: 12 }} width={48} unit="" label={{ value: yLabel, angle: -90, position: "insideLeft", fill: "var(--muted)", fontSize: 12 }} />
          <Tooltip contentStyle={{ background: "var(--panel)", border: "1px solid var(--border)", borderRadius: 6 }} />
          {series.length > 1 && <Legend />}
          {series.map((s, i) => (
            <Line key={s.key} type="monotone" dataKey={s.key} name={s.label} stroke={COLORS[i % COLORS.length]} dot={false} strokeWidth={2} connectNulls isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
