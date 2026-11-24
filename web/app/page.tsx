import BetsTable from "@/components/BetsTable";
import EquityChart from "@/components/EquityChart";
import Verdict from "@/components/Verdict";
import { getSummary } from "@/lib/data";
import { ago, num, pct, pctPlain, sign, units } from "@/lib/format";

export const revalidate = 300;

export default async function Overview() {
  const s = await getSummary();
  if (!s) {
    return (
      <>
        <h1>Overview</h1>
        <div className="empty">
          No data yet. The workflow writes <code>site/summary.json</code> to the <code>paper-data</code> branch after its
          first run with games on the board. Check the Actions tab, then refresh.
        </div>
      </>
    );
  }
  const realistic = s.presets.find((p) => p.id === "realistic");
  const sharp = s.presets.find((p) => p.id === "sharp");

  // merge the personas' equity curves onto one date axis
  const dates = [...new Set(s.personas.flatMap((p) => p.equity.map((e) => e.date)))].sort();
  const chart = dates.map((d) => {
    const row: Record<string, number | string | null> = { x: d.slice(5) };
    for (const p of s.personas) {
      const e = [...p.equity].filter((q) => q.date <= d).pop();
      row[p.id] = e ? e.pnl : null;
    }
    return row;
  });

  return (
    <>
      <h1>Overview</h1>
      <p className="sub">
        Updated {ago(s.generatedAt)} · {s.dates.length} slate day{s.dates.length === 1 ? "" : "s"} captured,{" "}
        {s.completeDates.length} graded · {s.candidates.toLocaleString()} candidate bets
      </p>

      <div className="cards">
        {[realistic, sharp].map((p) =>
          p ? (
            <div className="card" key={p.id}>
              <div className="k">{p.label} preset</div>
              <div className={`v ${sign(p.summary.roi)}`}>{pct(p.summary.roi)}</div>
              <div className="n">
                ± {pctPlain(p.summary.roi_band, 1)} (2σ) over {p.summary.settled} settled bets · <Verdict text={p.verdict} />
              </div>
            </div>
          ) : null,
        )}
        <div className="card">
          <div className="k">Closing line value (Realistic)</div>
          <div className="v">{realistic?.summary.clv ? pctPlain(realistic.summary.clv.beat_rate) : "—"}</div>
          <div className="n">
            {realistic?.summary.clv
              ? `beat the close on ${realistic.summary.clv.n} bets (±${pctPlain(realistic.summary.clv.beat_band)} is noise)`
              : "no bets with a comparable close yet"}
          </div>
        </div>
        <div className="card">
          <div className="k">Bets to resolve</div>
          <div className="v">{realistic?.betsToResolve ? realistic.betsToResolve.toLocaleString() : "—"}</div>
          <div className="n">settled bets needed to see an ROI this size at 2σ</div>
        </div>
      </div>

      <div className="note">
        Persona results show what a strategy would have <em>felt</em> like. A persona taking ~5 bets a day makes ~900 a
        season, where the 2σ band on ROI is roughly ±7%, so only the filtered view (far more bets) can say whether the
        projections are real. Read the verdict, not the rank.
      </div>

      <h2>Personas</h2>
      <div className="tablewrap">
        <table>
          <thead>
            <tr><th>Persona</th><th>Bets</th><th>Settled</th><th>Staked</th><th>P/L</th><th>ROI</th><th>Bankroll</th></tr>
          </thead>
          <tbody>
            {s.personas.map((p) => (
              <tr key={p.id} title={p.description}>
                <td>{p.label}</td>
                <td>{p.bets}</td><td>{p.settled}</td><td>{num(p.staked, 1)}u</td>
                <td className={sign(p.pnl)}>{units(p.pnl)}</td>
                <td className={sign(p.roi)}>{pct(p.roi)}</td>
                <td>{p.compounding ? `${num(p.bankroll, 1)}u` : <span className="muted">flat</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>Cumulative P/L by persona</h2>
      <EquityChart data={chart} series={s.personas.map((p) => ({ key: p.id, label: p.label.replace(/^The /, "") }))} />

      <h2>Latest day{ s.latestDate ? ` (${s.latestDate})` : "" }: best Realistic bets</h2>
      <BetsTable rows={s.today} />
    </>
  );
}
