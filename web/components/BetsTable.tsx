import { american, label, num, pct, STAT_LABEL, BOOK_LABEL, sign, units } from "@/lib/format";

type R = { [k: string]: string | number | boolean | null | undefined };

export default function BetsTable({ rows, showStake = false }: { rows: R[]; showStake?: boolean }) {
  if (!rows.length) return <div className="empty">No bets yet.</div>;
  return (
    <div className="tablewrap">
      <table>
        <thead>
          <tr>
            <th>Date</th><th className="l">Player</th><th className="l">Market</th><th>Side</th><th>Line</th><th>Odds</th>
            <th className="l">Book</th><th>Edge</th>{showStake && <th>Stake</th>}<th>Result</th><th>P/L</th><th>CLV</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const pnl = (r.pnl_units ?? r.pnl) as number | null;
            return (
              <tr key={i}>
                <td>{String(r.date).slice(5)}</td>
                <td className="l">{r.name}{r.team ? <span className="muted"> {r.team}</span> : null}</td>
                <td className="l">{label(STAT_LABEL, String(r.stat))}</td>
                <td>{r.side}</td><td>{r.line}</td><td>{american(r.odds)}</td>
                <td className="l">{label(BOOK_LABEL, String(r.book))}</td>
                <td>{pct(r.edge as number)}</td>
                {showStake && <td>{num(r.stake_units as number, 2)}u</td>}
                <td>{r.status === "pending" ? <span className="muted">pending</span> : `${r.status}${r.actual !== null && r.actual !== undefined ? ` (${r.actual})` : ""}`}</td>
                <td className={sign(pnl)}>{pnl === null || pnl === undefined ? "—" : units(pnl)}</td>
                <td className={sign(r.clv as number)}>{r.clv === null || r.clv === undefined ? "—" : pct(r.clv as number)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
