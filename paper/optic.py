"""HTTP client for the OpticOdds v3 API: auth, rate limiting, batching, retries.

The only module that touches the network for odds. Limits (same API as the NFL
repo's client): /fixtures/odds takes at most 5 sportsbooks and 5 fixture ids per
request, with sportsbooks sent as REPEATED params; /fixtures/odds/historical
takes exactly ONE fixture. Auth is an X-Api-Key header, never the query string,
so the key cannot land in a logged URL.

NOTE: written against the NFL repo's live-verified behaviour, NOT run against
NBA from the build session (api.opticodds.com was blocked there, and no key was
set). `python -m paper.cli discover` is the first thing to run live.
"""

from __future__ import annotations

import os
import random
import time
from urllib.parse import urlencode

DEFAULT_BASE_URL = "https://api.opticodds.com/api/v3"
MAX_BOOKS = 5
MAX_FIXTURES = 5
MAX_FIXTURES_HISTORICAL = 1
DEFAULT_RATE = (10, 15.0)  # 10 requests / 15 s (the tighter, historical limit)


class OpticOddsError(RuntimeError):
    def __init__(self, msg, status=None, endpoint=None):
        super().__init__(msg)
        self.status, self.endpoint = status, endpoint


def chunk(seq, size):
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def build_query(params) -> str:
    """Arrays become repeated params (what OpticOdds requires)."""
    pairs = []
    for k, v in (params or {}).items():
        if v is None or v == "":
            continue
        if isinstance(v, (list, tuple)):
            pairs.extend((k, str(x)) for x in v if x is not None and x != "")
        else:
            pairs.append((k, str(v)))
    return urlencode(pairs)


class _Limiter:
    def __init__(self, requests, per_s):
        self.requests, self.per_s, self.times = requests, per_s, []

    def take(self, now=time.monotonic, sleep=time.sleep):
        while True:
            t = now()
            self.times = [x for x in self.times if t - x < self.per_s]
            if len(self.times) < self.requests:
                self.times.append(t)
                return
            sleep(self.per_s - (t - self.times[0]) + 0.005)


class OpticOddsClient:
    def __init__(self, api_key=None, base_url=None, rate=DEFAULT_RATE, max_retries=4,
                 session=None, logger=print):
        self.api_key = api_key or os.environ.get("OPTICODDS_API_KEY")
        if not self.api_key:
            raise OpticOddsError("No OpticOdds API key. Set OPTICODDS_API_KEY.")
        self.base_url = (base_url or os.environ.get("OPTICODDS_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.limiter = _Limiter(*rate)
        self.max_retries = max_retries
        self.logger = logger
        self.request_count = 0
        if session is None:
            import requests
            session = requests.Session()
        self.session = session

    def get(self, endpoint, params=None):
        path = endpoint if endpoint.startswith("/") else "/" + endpoint
        qs = build_query(params)
        url = f"{self.base_url}{path}" + (f"?{qs}" if qs else "")
        last = None
        for attempt in range(self.max_retries + 1):
            self.limiter.take()
            self.request_count += 1
            try:
                res = self.session.get(url, headers={"X-Api-Key": self.api_key,
                                                     "Accept": "application/json"}, timeout=45)
            except Exception as err:  # network-level: worth retrying
                last = OpticOddsError(f"Network error: {err}", endpoint=path)
                if attempt < self.max_retries:
                    time.sleep(self._backoff(attempt))
                    continue
                raise last
            if res.ok:
                try:
                    return res.json()
                except ValueError:
                    raise OpticOddsError(f"Non-JSON response from {path}: {res.text[:200]}",
                                         status=res.status_code, endpoint=path)
            if res.status_code == 429 or res.status_code >= 500:
                last = OpticOddsError(f"HTTP {res.status_code} from {path}",
                                      status=res.status_code, endpoint=path)
                if attempt < self.max_retries:
                    ra = res.headers.get("retry-after")
                    wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else self._backoff(attempt)
                    self.logger(f"  OpticOdds {res.status_code} on {path}; retry in {wait:.0f}s")
                    time.sleep(wait)
                    continue
                raise last
            # other 4xx: a bad key / bad param fails identically every time.
            raise OpticOddsError(f"HTTP {res.status_code} from {path}: {res.text[:200]}",
                                 status=res.status_code, endpoint=path)
        raise last

    @staticmethod
    def _backoff(attempt):
        return 1.0 * 2 ** attempt * (0.5 + random.random() * 0.5)

    def get_all(self, endpoint, params=None, max_pages=50):
        rows = []
        for page in range(1, max_pages + 1):
            payload = self.get(endpoint, {**(params or {}), "page": page})
            batch = unwrap(payload)
            rows.extend(batch)
            if isinstance(payload, dict) and isinstance(payload.get("has_more"), bool):
                if not payload["has_more"]:
                    return rows
                continue
            tp = payload.get("total_pages") if isinstance(payload, dict) else None
            if isinstance(tp, (int, float)):
                if page >= tp:
                    return rows
                continue
            return rows
        self.logger(f"  {endpoint}: hit the {max_pages}-page cap; results may be truncated")
        return rows

    # ---- discovery -------------------------------------------------------
    def get_sportsbooks(self):
        return self.get_all("/sportsbooks")

    def get_markets(self, sport="basketball", league="nba", markets_only=True):
        return self.get_all("/markets", {"sport": sport, "league": league,
                                         "markets_only": str(markets_only).lower()})

    # ---- fixtures --------------------------------------------------------
    def get_fixtures(self, start_date=None, end_date=None, status=None,
                     sport="basketball", league="nba"):
        return self.get_all("/fixtures", {"sport": sport, "league": league,
                                          "start_date": start_date, "end_date": end_date,
                                          "status": status})

    # ---- odds ------------------------------------------------------------
    def _fan_out(self, endpoint, fixture_ids, sportsbooks, markets, per_fixture, on_progress):
        if not fixture_ids:
            return []
        if not sportsbooks:
            raise OpticOddsError("At least one sportsbook is required.")
        fb, bb = chunk(list(fixture_ids), per_fixture), chunk(list(sportsbooks), MAX_BOOKS)
        total, done, out = len(fb) * len(bb), 0, []
        for books in bb:
            for fx in fb:
                try:
                    out.append(self.get(endpoint, {"sportsbook": books, "fixture_id": fx,
                                                   "market": markets, "odds_format": "AMERICAN"}))
                except OpticOddsError as err:
                    # one book/fixture combo failing must not abort the capture
                    self.logger(f"  odds fetch failed books={books} fixtures={len(fx)}: {err}")
                done += 1
                if on_progress:
                    on_progress(done, total)
        return out

    def get_odds_for_fixtures(self, fixture_ids, sportsbooks, markets, on_progress=None):
        return self._fan_out("/fixtures/odds", fixture_ids, sportsbooks, markets,
                             MAX_FIXTURES, on_progress)

    def get_historical_odds(self, fixture_ids, sportsbooks, markets, on_progress=None):
        return self._fan_out("/fixtures/odds/historical", fixture_ids, sportsbooks, markets,
                             MAX_FIXTURES_HISTORICAL, on_progress)

    # ---- results (settlement alternative to nba_api) ----------------------
    def get_fixture_results(self, fixture_ids):
        out = []
        for fx in chunk(list(fixture_ids), MAX_FIXTURES):
            out.append(self.get("/fixtures/results", {"fixture_id": fx}))
        return out


def unwrap(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        d = payload.get("data")
        if isinstance(d, list):
            return d
        if isinstance(d, dict):
            return [d]
    return []
