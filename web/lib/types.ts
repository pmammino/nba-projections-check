export type Side = "over" | "under";
export type Status = "pending" | "won" | "lost" | "push" | "void";
export type Basis = "proj" | "pool" | "sharp";

/** One candidate bet (a book's price on one side of one line), unpacked from pool.json. */
export interface Row {
  date: string; slot: string; playerId: number; name: string; team: string; opp: string;
  stat: string; book: string; side: Side; line: number; odds: number; implied: number;
  pProj: number | null; pPool: number | null; pSharp: number | null;
  nBooks: number; nSharp: number; mainLine: number; rung: number;
  status: Status; actual: number | null; closeFair: number | null; bettable: boolean;
}

export interface Bet extends Row { edge: number | null; dec: number; pnl: number | null; clv: number | null }

export interface Filters {
  slots: string[] | null; dates: string[] | null; books: string[] | null; bettable_only: boolean;
  stats: string[] | null; sides: "both" | Side; odds_min: number; odds_max: number; basis: Basis;
  min_edge: number; max_edge: number | null; sharp_confirm: number | null; min_books: number;
  max_rung: number | null; ladder: "all" | "best"; shop: boolean; timing: "first" | "every";
}

export interface Summary {
  bets: number; settled: number; won: number; lost: number; push: number; void: number;
  unplayed: number; no_line: number; pnl: number; roi: number | null; roi_band: number | null;
  per_bet_sd: number | null; win_rate: number | null; avg_decimal: number | null; avg_edge: number | null;
  clv: { n: number; avg: number; avg_band: number | null; beat_rate: number; beat_band: number } | null;
}

export interface PackedPool {
  generatedAt: string; completeDates: string[]; columns: string[];
  tables: { dates: string[]; slots: string[]; opps: string[]; stats: string[]; books: string[];
            players: { id: number; name: string; team: string }[] };
  rows: number[][];
}

export interface PersonaSummary {
  id: string; label: string; description: string; compounding: boolean;
  bets: number; settled: number; staked: number; pnl: number; roi: number | null; bankroll: number;
  equity: { date: string; pnl: number; bankroll: number }[];
  recent: Record<string, string | number | null>[];
}

export interface PresetSummary {
  id: string; label: string; description: string; filters: Filters; summary: Summary;
  verdict: string; betsToResolve: number | null;
}

export interface SiteSummary {
  generatedAt: string; dates: string[]; completeDates: string[]; candidates: number;
  personas: PersonaSummary[]; presets: PresetSummary[]; latestDate: string | null;
  today: Record<string, string | number | null>[];
  calibration: Record<string, { n: number; below: number; exp_below: number; above: number; exp_above: number;
                                inside: number; exp_inside: number }>;
  defaults: Filters;
}

export interface PoolIndex {
  generatedAt: string; completeDates: string[];
  shards: { date: string; rows: number; graded: boolean }[];
}
