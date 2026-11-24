export const pct = (x: number | null | undefined, d = 1) =>
  x === null || x === undefined || !Number.isFinite(x) ? "—" : `${x >= 0 ? "+" : ""}${(x * 100).toFixed(d)}%`;
export const pctPlain = (x: number | null | undefined, d = 0) =>
  x === null || x === undefined || !Number.isFinite(x) ? "—" : `${(x * 100).toFixed(d)}%`;
export const units = (x: number | null | undefined, d = 2) =>
  x === null || x === undefined || !Number.isFinite(x) ? "—" : `${x >= 0 ? "+" : ""}${x.toFixed(d)}u`;
export const num = (x: number | null | undefined, d = 0) =>
  x === null || x === undefined || !Number.isFinite(x) ? "—" : x.toFixed(d);
export const american = (o: unknown) => {
  const n = Number(o);
  return !Number.isFinite(n) ? "—" : n > 0 ? `+${n}` : `${n}`;
};
export const sign = (x: number | null | undefined) => (x === null || x === undefined ? "" : x > 0 ? "pos" : x < 0 ? "neg" : "");
export const ago = (iso: string | undefined) => {
  if (!iso) return "never";
  const m = Math.round((Date.now() - Date.parse(iso)) / 60000);
  return m < 1 ? "just now" : m < 60 ? `${m} min ago` : m < 2880 ? `${Math.round(m / 60)} h ago` : `${Math.round(m / 1440)} d ago`;
};
export const STAT_LABEL: Record<string, string> = {
  pts: "Points", reb: "Rebounds", ast: "Assists", threes: "Made threes", stl: "Steals", blk: "Blocks", tov: "Turnovers",
  threesa: "Threes attempted", pr: "Pts+Reb", pa: "Pts+Ast", ra: "Reb+Ast", pra: "Pts+Reb+Ast", stlblk: "Stl+Blk",
  fgm: "FG made", fga: "FG attempted", ftm: "FT made", fta: "FT attempted", oreb: "Off. rebounds", dreb: "Def. rebounds",
  dd: "Double-double", td: "Triple-double",
};
export const BOOK_LABEL: Record<string, string> = {
  draftkings: "DraftKings", fanduel: "FanDuel", betmgm: "BetMGM", caesars: "Caesars", betrivers: "BetRivers",
  hardrock: "Hard Rock", thescore: "theScore", pinnacle: "Pinnacle", circasports: "Circa",
};
export const label = (m: Record<string, string>, k: string) => m[k] ?? k;
