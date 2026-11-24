// Parity: the TypeScript filter engine must reproduce paper/paper_filter.py exactly on the same pool.
import { describe, expect, it } from "vitest";
import fixture from "./__fixtures__/parity.json";
import { applyFilters, decimalOf, oddsInRange, summarize, unpack, verdict } from "./filter";
import type { Filters, PackedPool } from "./types";

const rows = unpack(fixture.pool as unknown as PackedPool);
const key = (b: { date: string; slot: string; playerId: number; stat: string; book: string; line: number; side: string; odds: number }) =>
  [b.date, b.slot, b.playerId, b.stat, b.book, b.line, b.side, b.odds];

describe("filter parity with python", () => {
  it("unpacks every row", () => {
    expect(rows.length).toBe(fixture.pool.rows.length);
    expect(rows.length).toBeGreaterThan(1000);
  });

  for (const [name, c] of Object.entries(fixture.cases)) {
    it(`${name}: same bets, same order, same summary`, () => {
      const bets = applyFilters(rows, c.filters as unknown as Partial<Filters>);
      expect(bets.map(key)).toEqual(c.bets);
      const s = summarize(bets, fixture.completeDates);
      const want = c.summary as Record<string, unknown>;
      for (const k of ["bets", "settled", "won", "lost", "push", "void", "unplayed", "no_line"]) {
        expect(s[k as keyof typeof s], `${name}.${k}`).toBe(want[k]);
      }
      for (const k of ["pnl", "roi", "roi_band", "per_bet_sd", "win_rate", "avg_edge"]) {
        const w = want[k] as number | null;
        const g = s[k as keyof typeof s] as number | null;
        if (w === null) expect(g, `${name}.${k}`).toBeNull();
        else expect(g as number, `${name}.${k}`).toBeCloseTo(w, 9);
      }
      const wc = want.clv as { n: number; avg: number; beat_rate: number } | null;
      if (wc === null) expect(s.clv).toBeNull();
      else {
        expect(s.clv?.n).toBe(wc.n);
        expect(s.clv?.avg as number).toBeCloseTo(wc.avg, 9);
        expect(s.clv?.beat_rate as number).toBeCloseTo(wc.beat_rate, 9);
      }
      expect(verdict(s)).toBe(c.verdict);
    });
  }

  it("the default filter finds bets, so the parity checks are not vacuous", () => {
    expect(fixture.cases.default.bets.length).toBeGreaterThan(20);
    expect(fixture.cases.everything.bets.length).toBeGreaterThan(fixture.cases.default.bets.length);
  });
});

describe("odds", () => {
  it("compares in decimal space, bounds in either order", () => {
    expect(oddsInRange(-400, -400, 200)).toBe(true);
    expect(oddsInRange(200, 200, -400)).toBe(true);
    expect(oddsInRange(-500, -400, 200)).toBe(false);
    expect(decimalOf(50)).toBeNull();
  });
});
