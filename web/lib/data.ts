import type { SiteSummary } from "./types";

/** Public repo: the Action commits site/*.json to the `paper-data` branch and raw.githubusercontent
 *  serves it with open CORS, so no token, no proxy. Override to point at a fork or another host. */
export const DATA_BASE =
  process.env.NEXT_PUBLIC_DATA_BASE_URL ||
  "https://raw.githubusercontent.com/pmammino/nba-projections-check/paper-data/site";

/** Server-side, cached 5 minutes (ISR). Returns null until the first export exists. */
export async function getSummary(): Promise<SiteSummary | null> {
  try {
    const res = await fetch(`${DATA_BASE}/summary.json`, { next: { revalidate: 300 } });
    return res.ok ? ((await res.json()) as SiteSummary) : null;
  } catch {
    return null;
  }
}
