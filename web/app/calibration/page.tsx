import { getSummary } from "@/lib/data";
import { label, pctPlain, STAT_LABEL } from "@/lib/format";

export const revalidate = 300;

export default async function Calibration() {
  const s = await getSummary();
  const rows = s ? Object.entries(s.calibration).sort() : [];
  return (
    <>
      <h1>Is the feed calibrated?</h1>
      <p className="sub">
        RotoWire publishes a floor (25th percentile) and ceiling (75th) for every stat. Each price here is a distribution
        fitted to those, so the fit is only as good as the band. For each stat: how often the actual result fell below the
        floor, above the ceiling, and inside the band, <strong>realised</strong> against what the fitted distribution{" "}
        <strong>implies</strong>. Counts are integers, so the band holds 50% or more, not exactly 50%.
      </p>
      {!rows.length ? (
        <div className="empty">No graded days with a feed snapshot yet.</div>
      ) : (
        <>
          <div className="tablewrap">
            <table>
              <thead>
                <tr>
                  <th>Stat</th><th>Player-games</th><th>Below floor</th><th>(implied)</th>
                  <th>Above ceiling</th><th>(implied)</th><th>Inside band</th><th>(implied)</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(([stat, c]) => {
                  const off = Math.abs(c.above - c.exp_above) + Math.abs(c.below - c.exp_below);
                  return (
                    <tr key={stat}>
                      <td>{label(STAT_LABEL, stat)}</td><td>{c.n}</td>
                      <td>{pctPlain(c.below)}</td><td className="muted">{pctPlain(c.exp_below)}</td>
                      <td>{pctPlain(c.above)}</td><td className="muted">{pctPlain(c.exp_above)}</td>
                      <td className={off > 0.2 ? "warn" : ""}>{pctPlain(c.inside)}</td><td className="muted">{pctPlain(c.exp_inside)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="note">
            Read this before trusting any price. If the realised share outside the band is well above the implied share,
            the feed is over-confident on that stat and every long-shot over looks like value it is not. Small samples
            move a lot: treat a stat with fewer than a few hundred player-games as a hint.
          </div>
        </>
      )}
    </>
  );
}
