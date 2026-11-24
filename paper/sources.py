"""Pull the preseason projection sheet from RotoWire, in this repo, no manual download.

    https://www.rotowire.com/basketball/tables/projections.php?type=pre&pos=ALL

That URL serves the table's rows as JSON. The model needs these columns (see
rates.py / state.build_priors): newsid, firstname, lastname, team, position,
games, minutes, fga, fgm, pt3a, pt3m, fta, ftm, oreb, dreb, ast, stl, blk, tov,
pf. RotoWire's own spellings (assists/steals/..., fg3a/...) are mapped, and the
fetch FAILS LOUDLY listing any column it cannot find rather than letting the
model run on a sheet with a silent hole in it.

STATUS: written against the endpoint's known shape; NOT run against the live site
from the build session (rotowire.com was blocked there). The first run prints the
columns it saw. If RotoWire changed a field name, add it to ALIASES below.

Be polite: the sheet is static preseason data, so it is fetched once and cached;
`refresh=True` re-pulls it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .teams import canon

URL = "https://www.rotowire.com/basketball/tables/projections.php?type=pre&pos=ALL"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; nba-paper-trading/1.0)", "Accept": "application/json,*/*"}

REQUIRED = ["newsid", "firstname", "lastname", "team", "position", "games", "minutes", "fga", "fgm",
            "pt3a", "pt3m", "fta", "ftm", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf"]
ALIASES = {
    "ast": ["ast", "assists"], "stl": ["stl", "steals"], "blk": ["blk", "blocks"],
    "tov": ["tov", "turnovers", "to"], "pt3a": ["pt3a", "fg3a", "threepa", "3pa", "tpa"],
    "pt3m": ["pt3m", "fg3m", "threepm", "3pm", "tpm"], "minutes": ["minutes", "min"],
    "games": ["games", "gp", "g"], "position": ["position", "pos"],
    "newsid": ["newsid", "playerid", "id"], "firstname": ["firstname", "first_name"],
    "lastname": ["lastname", "last_name"], "team": ["team", "teamabbr", "team_abbr"],
    "pf": ["pf", "fouls"], "oreb": ["oreb", "offreb", "orb"], "dreb": ["dreb", "defreb", "drb"],
}


def parse_preseason(payload) -> pd.DataFrame:
    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("RotoWire returned no rows (changed URL, or blocked).")
    df = pd.DataFrame(rows)
    df.columns = [str(c).strip().lower() for c in df.columns]
    out, missing = pd.DataFrame(index=df.index), []
    for col in REQUIRED:
        src = next((a for a in ALIASES.get(col, [col]) if a in df.columns), None)
        if src is None:
            missing.append(col)
        else:
            out[col] = df[src]
    if missing:
        raise ValueError(f"RotoWire sheet is missing {missing}. Columns seen: {sorted(df.columns)}")
    num = [c for c in REQUIRED if c not in ("firstname", "lastname", "team", "position")]
    for c in num:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["newsid"]).copy()
    out["newsid"] = out["newsid"].astype("int64")
    for c in num[1:]:
        out[c] = out[c].fillna(0.0)
    out["team"] = out["team"].map(lambda t: canon(t) if isinstance(t, str) and t.strip() else None)
    return out.drop_duplicates("newsid").reset_index(drop=True)


def fetch_preseason(session=None, url: str = URL) -> pd.DataFrame:
    if session is None:
        import requests
        session = requests.Session()
    res = session.get(url, headers=HEADERS, timeout=60)
    if not res.ok:
        raise RuntimeError(f"RotoWire HTTP {res.status_code} for the preseason sheet")
    try:
        payload = res.json()
    except ValueError:
        payload = json.loads(res.text)
    return parse_preseason(payload)


def load_preseason(cache: Path, refresh=False, session=None) -> pd.DataFrame:
    """Cached fetch, then the model's own integrity repair (validate.repair) so one
    bad row (pt3a > fga) cannot stop a slate."""
    cache = Path(cache)
    if cache.exists() and not refresh:
        df = pd.read_csv(cache)
    else:
        df = fetch_preseason(session)
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
    from . import bridge
    return bridge.model().validate.repair(df)
