"use client";
import { useEffect, useMemo, useState } from "react";
import BetsTable from "./BetsTable";
import EquityChart from "./EquityChart";
import Verdict from "./Verdict";
import { DATA_BASE } from "@/lib/data";
import {
  applyFilters, betsToResolve, DEFAULT_FILTERS, edgeBucket, groupSummaries, oddsBucket, summarize, unpack, verdict,
} from "@/lib/filter";
import { BOOK_LABEL, label, pct, pctPlain, sign, STAT_LABEL, units } from "@/lib/format";
import type { Bet, Filters, PackedPool, PoolIndex, PresetSummary, Row } from "@/lib/types";

const slotRank = (s: string) => ["morning", "pregame", "closing"].indexOf(s);

function toggle(list: string[] | null, all: string[], v: string): string[] | null {
  const cur = new Set(list ?? all);
  if (cur.has(v)) cur.delete(v); else cur.add(v);
  return cur.size === all.length ? null : [...cur];
}

const WINDOWS = [["7", "Last 7 days"], ["14", "Last 14 days"], ["30", "Last 30 days"], ["all", "Whole season"]] as const;

async function getJson<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status} for ${url.split("/").slice(-2).join("/")}`);
  return (await r.json()) as T;
}

export default function PaperExplorer({ presets }: { presets: PresetSummary[] }) {
  const [index, setIndex] = useState<PoolIndex | null>(null);
  const [shards, setShards] = useState<Record<string, Row[]>>({});
  const [win, setWin] = useState<string>("30");
  const [err, setErr] = useState<string | null>(null);
  const [f, setF] = useState<Filters>(DEFAULT_FILTERS);
  const [preset, setPreset] = useState<string | null>("realistic");

  // The pool is one small file per day: load the index, then only the days in the chosen window.
  useEffect(() => {
    let dead = false;
    getJson<PoolIndex>(`${DATA_BASE}/pool/index.json`)
      .then((i) => !dead && setIndex(i))
      .catch((e) => !dead && setErr(String(e.message ?? e)));
    return () => { dead = true; };
  }, []);

  const wanted = useMemo(() => {
    const dates = (index?.shards ?? []).map((x) => x.date).sort();
    return win === "all" ? dates : dates.slice(-Number(win));
  }, [index, win]);

  useEffect(() => {
    const missing = wanted.filter((d) => !(d in shards));
    if (!missing.length) return;
    let dead = false;
    Promise.all(missing.map((d) => getJson<PackedPool>(`${DATA_BASE}/pool/${d}.json`).then((p) => [d, unpack(p)] as const)))
      .then((got) => !dead && setShards((cur) => ({ ...cur, ...Object.fromEntries(got) })))
      .catch((e) => !dead && setErr(String(e.message ?? e)));
    return () => { dead = true; };
  }, [wanted, shards]);

  const loaded = wanted.filter((d) => d in shards);
  const rows: Row[] | null = useMemo(
    () => (index && loaded.length === wanted.length ? loaded.flatMap((d) => shards[d]) : null),
    [index, loaded.length, wanted.length, shards], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const complete = index?.completeDates ?? [];

  const set = (patch: Partial<Filters>) => { setF((x) => ({ ...x, ...patch })); setPreset(null); };
  const usePreset = (p: PresetSummary) => { setF({ ...DEFAULT_FILTERS, ...p.filters }); setPreset(p.id); };

  const stats = useMemo(() => [...new Set((rows ?? []).map((r) => r.stat))].sort(), [rows]);
  const books = useMemo(() => [...new Set((rows ?? []).filter((r) => r.bettable || !f.bettable_only).map((r) => r.book))].sort(), [rows, f.bettable_only]);
  const bets: Bet[] = useMemo(() => (rows ? applyFilters(rows, f) : []), [rows, f]);
  const s = useMemo(() => summarize(bets, complete), [bets, complete]);
  const v = verdict(s);

  const curve = useMemo(() => {
    const settled = bets.filter((b) => b.pnl !== null && (b.status === "won" || b.status === "lost"))
      .sort((a, b) => a.date.localeCompare(b.date) || slotRank(a.slot) - slotRank(b.slot));
    let cum = 0;
    return settled.map((b, i) => { cum += b.pnl as number; return { x: String(i + 1), pnl: Math.round(cum * 100) / 100 }; });
  }, [bets]);

  if (err) return <div className="empty">Could not load the bet pool ({err}). It appears after the first run with games on the board.</div>;
  if (index && !index.shards.length) return <div className="empty">The pool is empty: no candidate bets have been captured yet.</div>;
  if (!rows) return <div className="empty">Loading the bet pool{index ? ` (${loaded.length} of ${wanted.length} days)` : ""}…</div>;

  const dec = (b: Bet) => b.dec;
  const groups: [string, ReturnType<typeof groupSummaries>][] = [
    ["Market", groupSummaries(bets, (b) => b.stat, complete)],
    ["Book", groupSummaries(bets, (b) => b.book, complete)],
    ["Side", groupSummaries(bets, (b) => b.side, complete)],
    ["Odds", groupSummaries(bets, (b) => oddsBucket(dec(b)), complete)],
    ["Edge", groupSummaries(bets, (b) => edgeBucket(b.edge ?? 0), complete)],
    ["Capture", groupSummaries(bets, (b) => b.slot, complete)],
  ];
  const labelFor = (title: string, k: string) => (title === "Market" ? label(STAT_LABEL, k) : title === "Book" ? label(BOOK_LABEL, k) : k);
  const resolve = betsToResolve(s.roi, s.per_bet_sd);

  return (
    <>
      <div className="presets">
        {presets.map((p) => (
          <button key={p.id} className={preset === p.id ? "on" : ""} onClick={() => usePreset(p)} title={p.description}>{p.label}</button>
        ))}
      </div>
      {preset && <p className="sub">{presets.find((p) => p.id === preset)?.description}</p>}

      <div className="controls">
        <label>Days of history
          <select value={win} onChange={(e) => setWin(e.target.value)}>
            {WINDOWS.map(([v, name]) => <option key={v} value={v}>{name}</option>)}
          </select>
        </label>
        <label>Price basis
          <select value={f.basis} onChange={(e) => set({ basis: e.target.value as Filters["basis"] })}>
            <option value="proj">Projection</option><option value="pool">Books + projection share</option><option value="sharp">Sharp books only</option>
          </select>
        </label>
        <label>Min edge (%)
          <input type="number" step="0.5" value={+(f.min_edge * 100).toFixed(2)} onChange={(e) => set({ min_edge: Number(e.target.value) / 100 })} />
        </label>
        <label>Max edge (%) — blank: none
          <input type="number" step="1" value={f.max_edge === null ? "" : +(f.max_edge * 100).toFixed(2)} onChange={(e) => set({ max_edge: e.target.value === "" ? null : Number(e.target.value) / 100 })} />
        </label>
        <label>Odds from
          <input type="number" step="10" value={f.odds_min} onChange={(e) => set({ odds_min: Number(e.target.value) })} />
        </label>
        <label>Odds to
          <input type="number" step="10" value={f.odds_max} onChange={(e) => set({ odds_max: Number(e.target.value) })} />
        </label>
        <label>Sides
          <select value={f.sides} onChange={(e) => set({ sides: e.target.value as Filters["sides"] })}>
            <option value="both">Over and under</option><option value="over">Over</option><option value="under">Under</option>
          </select>
        </label>
        <label>Books quoting the line (min)
          <input type="number" min={1} max={9} value={f.min_books} onChange={(e) => set({ min_books: Number(e.target.value) })} />
        </label>
        <label>Distance from main line
          <select value={f.max_rung === null ? "any" : String(f.max_rung)} onChange={(e) => set({ max_rung: e.target.value === "any" ? null : Number(e.target.value) })}>
            <option value="any">Any rung</option><option value="0">Main line only</option><option value="1">Within 1 rung</option><option value="2">Within 2 rungs</option>
          </select>
        </label>
        <label>Sharp confirmation
          <select value={f.sharp_confirm === null ? "off" : String(f.sharp_confirm)} onChange={(e) => set({ sharp_confirm: e.target.value === "off" ? null : Number(e.target.value) })}>
            <option value="off">Off</option><option value="0">Sharp fair price not worse</option><option value="0.01">Sharp ≥ +1% better</option><option value="0.02">Sharp ≥ +2% better</option>
          </select>
        </label>
        <label>Ladder
          <select value={f.ladder} onChange={(e) => set({ ladder: e.target.value as Filters["ladder"] })}>
            <option value="best">Best rung per player/stat/side</option><option value="all">Every qualifying rung</option>
          </select>
        </label>
        <label>Timing
          <select value={f.timing} onChange={(e) => set({ timing: e.target.value as Filters["timing"] })}>
            <option value="first">First capture that qualifies</option><option value="every">Every capture (counts a market twice)</option>
          </select>
        </label>
        <label className="chk"><input type="checkbox" checked={f.shop} onChange={(e) => set({ shop: e.target.checked })} />Shop for best price</label>
        <label className="chk"><input type="checkbox" checked={!f.bettable_only} onChange={(e) => set({ bettable_only: !e.target.checked })} />Include reference books (Pinnacle, Circa)</label>
      </div>

      <h2>Markets</h2>
      <div className="chips">
        {stats.map((st) => (
          <button key={st} className={(f.stats ?? stats).includes(st) ? "on" : ""} onClick={() => set({ stats: toggle(f.stats, stats, st) })}>{label(STAT_LABEL, st)}</button>
        ))}
      </div>
      <h2>Books to bet at</h2>
      <div className="chips">
        {books.map((b) => (
          <button key={b} className={(f.books ?? books).includes(b) ? "on" : ""} onClick={() => set({ books: toggle(f.books, books, b) })}>{label(BOOK_LABEL, b)}</button>
        ))}
      </div>

      <h2>Result</h2>
      <div className="cards">
        <div className="card"><div className="k">ROI</div><div className={`v ${sign(s.roi)}`}>{pct(s.roi)}</div>
          <div className="n">± {pctPlain(s.roi_band, 1)} (2σ) · <Verdict text={v} /></div></div>
        <div className="card"><div className="k">Bets</div><div className="v">{s.bets.toLocaleString()}</div>
          <div className="n">{s.settled} settled · {s.won}W {s.lost}L {s.push}P · {s.void} void · {s.unplayed} unplayed</div></div>
        <div className="card"><div className="k">P/L (1 unit flat)</div><div className={`v ${sign(s.pnl)}`}>{units(s.pnl)}</div>
          <div className="n">win rate {pctPlain(s.win_rate)} · avg edge {pct(s.avg_edge)}</div></div>
        <div className="card"><div className="k">CLV</div><div className="v">{s.clv ? pctPlain(s.clv.beat_rate) : "—"}</div>
          <div className="n">{s.clv ? `beat the close on ${s.clv.n} bets · avg ${pct(s.clv.avg, 2)}` : "no comparable close yet"}</div></div>
      </div>
      {v === "too few bets to call" && (
        <div className="note">
          Under 200 settled bets the ROI band swallows any real edge, so this number is noise.
          {resolve && resolve > 200 ? ` At this ROI and variance it would take about ${resolve.toLocaleString()} settled bets to resolve.` : ""}
        </div>
      )}
      {s.no_line > 0 && (
        <div className="note">{s.no_line} bets are on graded days but have no box score to settle against, and are left out of ROI.</div>
      )}

      <h2>Cumulative P/L, bet by bet</h2>
      <EquityChart data={curve} series={[{ key: "pnl", label: "Cumulative units" }]} />

      <h2>Where it comes from</h2>
      <div className="grid2">
        {groups.map(([title, gs]) => (
          <div key={title}>
            <div className="muted" style={{ marginBottom: 4 }}>{title}</div>
            <div className="tablewrap"><table>
              <thead><tr><th>{title}</th><th>Bets</th><th>Settled</th><th>ROI</th><th>Band</th></tr></thead>
              <tbody>{gs.map((g) => (
                <tr key={g.key}><td>{labelFor(title, g.key)}</td><td>{g.summary.bets}</td><td>{g.summary.settled}</td>
                  <td className={sign(g.summary.roi)}>{pct(g.summary.roi)}</td><td className="muted">{pctPlain(g.summary.roi_band, 1)}</td></tr>
              ))}</tbody>
            </table></div>
          </div>
        ))}
      </div>

      <h2>Most recent bets</h2>
      <BetsTable rows={[...bets].sort((a, b) => b.date.localeCompare(a.date) || slotRank(b.slot) - slotRank(a.slot) || (b.edge ?? 0) - (a.edge ?? 0)).slice(0, 150)
        .map((b) => ({ ...b }))} />
      {bets.length > 150 && <p className="muted">Showing 150 of {bets.length.toLocaleString()}.</p>}
    </>
  );
}
