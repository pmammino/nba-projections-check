// Filtered paper trading, in the browser. A line-for-line port of paper/paper_filter.py: the
// parity test (filter.test.ts) runs both on the same data and fails if they ever disagree.
//
// Order is fixed and matters: shop within a capture, pick the rung within a capture, THEN take
// the first capture that has one. Picking the capture first would let the 2-hour-out price
// choose a bet the morning board never offered.
import type { Basis, Bet, Filters, PackedPool, Row, Status, Summary } from "./types";

export const SLOT_ORDER = ["morning", "pregame", "closing"];
export const BET_SLOTS = ["morning", "pregame"];
export const MIN_BETS_FOR_A_VERDICT = 200;
const STATUSES: Status[] = ["pending", "won", "lost", "push", "void"];

export const DEFAULT_FILTERS: Filters = {
  slots: BET_SLOTS, dates: null, books: null, bettable_only: true, stats: null, sides: "both",
  odds_min: -400, odds_max: 200, basis: "proj", min_edge: 0.03, max_edge: null, sharp_confirm: null,
  min_books: 3, max_rung: null, ladder: "best", shop: true, timing: "first",
};

export function unpack(p: PackedPool): Row[] {
  const c = Object.fromEntries(p.columns.map((n, i) => [n, i])) as Record<string, number>;
  const prob = (v: number) => (v < 0 ? null : v / 10000);
  const t = p.tables;
  return p.rows.map((a) => {
    const pl = t.players[a[c.player]];
    return {
      date: t.dates[a[c.date]], slot: t.slots[a[c.slot]], playerId: pl.id, name: pl.name, team: pl.team,
      opp: t.opps[a[c.opp]], stat: t.stats[a[c.stat]], book: t.books[a[c.book]],
      side: a[c.side] === 1 ? "under" : "over", line: a[c.line], odds: a[c.odds],
      implied: a[c.implied] / 10000, pProj: prob(a[c.pProj]), pPool: prob(a[c.pPool]), pSharp: prob(a[c.pSharp]),
      nBooks: a[c.nBooks], nSharp: a[c.nSharp], mainLine: a[c.mainLine], rung: a[c.rung],
      status: STATUSES[a[c.status]] ?? "pending", actual: a[c.actual] < 0 ? null : a[c.actual],
      closeFair: prob(a[c.closeFair]), bettable: a[c.bettable] === 1,
    };
  });
}

export const isAmerican = (x: number) => Number.isFinite(x) && Math.abs(x) >= 100;
export function decimalOf(american: number): number | null {
  if (!isAmerican(american)) return null;
  return american > 0 ? 1 + american / 100 : 1 + 100 / -american;
}

export function edgeOf(r: Row, basis: Basis): number | null {
  const p = basis === "pool" ? r.pPool : basis === "sharp" ? r.pSharp : r.pProj;
  return p === null || p === undefined ? null : p - r.implied;
}

/** Compared as DECIMAL odds (continuous across even money); bounds may come in either order. */
export function oddsInRange(american: number, lo: number, hi: number): boolean {
  const d = decimalOf(american), a = decimalOf(lo), b = decimalOf(hi);
  if (d === null || a === null || b === null) return false;
  return d >= Math.min(a, b) - 1e-9 && d <= Math.max(a, b) + 1e-9;
}

const slotRank = (s: string) => { const i = SLOT_ORDER.indexOf(s); return i === -1 ? SLOT_ORDER.length : i; };

export function passesRow(r: Row, f: Filters): boolean {
  if (f.slots && !f.slots.includes(r.slot)) return false;
  if (f.dates && !f.dates.includes(r.date)) return false;
  if (f.bettable_only && !r.bettable) return false;
  if (f.books && !f.books.includes(r.book)) return false;
  if (f.stats && !f.stats.includes(r.stat)) return false;
  if (f.sides !== "both" && r.side !== f.sides) return false;
  if (!oddsInRange(r.odds, f.odds_min, f.odds_max)) return false;
  if (r.nBooks < f.min_books) return false;
  if (f.max_rung !== null && Math.abs(r.rung) > f.max_rung) return false;
  const e = edgeOf(r, f.basis);
  if (e === null || e < f.min_edge - 1e-12) return false;
  if (f.max_edge !== null && e > f.max_edge + 1e-12) return false;
  if (f.sharp_confirm !== null) {
    // A market no sharp book quoted cannot be confirmed: dropped, not waved through.
    if (r.pSharp === null || r.pSharp - r.implied < f.sharp_confirm - 1e-12) return false;
  }
  return true;
}

function bestPer<T>(rows: T[], keyOf: (r: T) => string, better: (a: T, b: T) => boolean): T[] {
  const best = new Map<string, T>();
  for (const r of rows) {
    const k = keyOf(r);
    const cur = best.get(k);
    if (cur === undefined || better(r, cur)) best.set(k, r);
  }
  return [...best.values()];
}

function toBet(r: Row, basis: Basis): Bet {
  const d = decimalOf(r.odds) as number;
  const pnl = r.status === "won" ? d - 1 : r.status === "lost" ? -1 : r.status === "push" || r.status === "void" ? 0 : null;
  return { ...r, edge: edgeOf(r, basis), dec: d, pnl, clv: r.closeFair === null ? null : r.closeFair - r.implied };
}

export function applyFilters(rows: Row[], filters: Partial<Filters> = {}): Bet[] {
  const f: Filters = { ...DEFAULT_FILTERS, ...filters };
  let kept = rows.filter((r) => passesRow(r, f));
  const edge = (r: Row) => edgeOf(r, f.basis) as number;
  const dec = (r: Row) => decimalOf(r.odds) as number;
  if (f.shop) {
    kept = bestPer(kept, (r) => [r.slot, r.date, r.playerId, r.stat, r.line, r.side].join("|"),
      (a, b) => a.odds !== b.odds && dec(a) > dec(b));
  }
  if (f.ladder === "best") {
    kept = bestPer(kept, (r) => [r.slot, r.date, r.playerId, r.stat, r.side, f.shop ? "" : r.book].join("|"),
      (a, b) => edge(a) > edge(b) + 1e-12 || (Math.abs(edge(a) - edge(b)) <= 1e-12 && Math.abs(a.rung) < Math.abs(b.rung)));
  }
  if (f.timing === "first") {
    const keyOf = (r: Row) => (f.ladder === "best"
      ? [r.date, r.playerId, r.stat, r.side, f.shop ? "" : r.book]
      : [r.date, r.playerId, r.stat, r.line, r.side, f.shop ? "" : r.book]).join("|");
    const first = new Map<string, number>();
    for (const r of kept) {
      const k = keyOf(r);
      first.set(k, Math.min(first.get(k) ?? 99, slotRank(r.slot)));
    }
    kept = kept.filter((r) => slotRank(r.slot) === first.get(keyOf(r)));
  }
  return kept.map((r) => toBet(r, f.basis))
    .sort((a, b) => a.date.localeCompare(b.date) || slotRank(a.slot) - slotRank(b.slot) || (b.edge ?? 0) - (a.edge ?? 0));
}

/** ROI, win rate, the band around the ROI, and CLV. A `void` (DNP, stake refunded) is not a loss;
 *  a bet still pending on a COMPLETE date has no box score and is counted as no_line, never a win. */
export function summarize(bets: Bet[], completeDates: string[] = []): Summary {
  const done = new Set(completeDates);
  let won = 0, lost = 0, push = 0, voided = 0, unplayed = 0, noLine = 0, pnl = 0, decSum = 0, edgeSum = 0;
  const returns: number[] = [], clv: number[] = [];
  for (const b of bets) {
    decSum += b.dec; edgeSum += b.edge ?? 0;
    if (b.clv !== null) clv.push(b.clv);
    if (b.status === "won" || b.status === "lost") {
      if (b.status === "won") won++; else lost++;
      pnl += b.pnl as number; returns.push(b.pnl as number);
    } else if (b.status === "push") push++;
    else if (b.status === "void") voided++;
    else if (done.has(b.date)) noLine++;
    else unplayed++;
  }
  const n = returns.length;
  const mean = n ? pnl / n : null;
  const sd = n > 1 ? Math.sqrt(returns.reduce((a, r) => a + (r - (mean as number)) ** 2, 0) / (n - 1)) : null;
  const cn = clv.length;
  const cmean = cn ? clv.reduce((a, x) => a + x, 0) / cn : null;
  const csd = cn > 1 ? Math.sqrt(clv.reduce((a, x) => a + (x - (cmean as number)) ** 2, 0) / (cn - 1)) : null;
  return {
    bets: bets.length, settled: n, won, lost, push, void: voided, unplayed, no_line: noLine, pnl,
    roi: n ? pnl / n : null, roi_band: sd !== null ? (2 * sd) / Math.sqrt(n) : null, per_bet_sd: sd,
    win_rate: n ? won / n : null, avg_decimal: bets.length ? decSum / bets.length : null,
    avg_edge: bets.length ? edgeSum / bets.length : null,
    clv: cn ? { n: cn, avg: cmean as number, avg_band: csd !== null ? (2 * csd) / Math.sqrt(cn) : null,
                beat_rate: clv.filter((x) => x > 0).length / cn, beat_band: 2 * Math.sqrt(0.25 / cn) } : null,
  };
}

export const betsToResolve = (roi: number | null, sd: number | null) =>
  !roi || !sd ? null : Math.ceil((2 * sd / Math.abs(roi)) ** 2);

export function verdict(s: Summary, floor = MIN_BETS_FOR_A_VERDICT): string {
  if (s.settled < floor) return "too few bets to call";
  if (s.roi === null || s.roi_band === null) return "no ROI";
  if (s.roi - s.roi_band > 0) return "positive";
  if (s.roi + s.roi_band < 0) return "negative";
  return "indistinguishable from zero";
}

export function groupSummaries(bets: Bet[], keyOf: (b: Bet) => string, completeDates: string[] = []) {
  const groups = new Map<string, Bet[]>();
  for (const b of bets) { const k = keyOf(b); (groups.get(k) ?? groups.set(k, []).get(k)!).push(b); }
  return [...groups.entries()].map(([key, rs]) => ({ key, summary: summarize(rs, completeDates) }))
    .sort((a, b) => b.summary.bets - a.summary.bets || a.key.localeCompare(b.key));
}

export const ODDS_BUCKETS: [string, (d: number) => boolean][] = [
  ["heavy favourite (-400 or shorter)", (d) => d <= 1.25 + 1e-9],
  ["favourite (-399 to -150)", (d) => d <= 1 + 100 / 150 + 1e-9],
  ["near even (-149 to +120)", (d) => d <= 2.2 + 1e-9],
  ["underdog (+121 to +300)", (d) => d <= 4 + 1e-9],
  ["long shot (+301 and longer)", () => true],
];
export const oddsBucket = (dec: number) => (ODDS_BUCKETS.find(([, t]) => t(dec)) as [string, unknown])[0];
export const EDGE_BUCKETS: [string, (e: number) => boolean][] = [
  ["3–5%", (e) => e < 0.05], ["5–10%", (e) => e < 0.1], ["10–15%", (e) => e < 0.15], ["15%+", () => true],
];
export const edgeBucket = (e: number) => (EDGE_BUCKETS.find(([, t]) => t(e)) as [string, unknown])[0];
