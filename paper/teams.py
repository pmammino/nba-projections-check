"""One canonical spelling per team (NBA tri-codes), so a join can never silently
drop a team.

RotoWire, ESPN, OpticOdds and nba.com each abbreviate a handful of teams
differently (BRO/BKN, PHO/PHX, NY/NYK, SA/SAS, GS/GSW, NO/NOP/NOR, WSH/WAS,
UTAH/UTA, CHR/CHA...). ingest.py already warns that an inner join on raw
abbreviations drops "roughly a quarter of the league, invisibly". Every team
label that enters this package goes through canon() first, on BOTH sides of a
join; an unknown label comes back unchanged (and unmatched) rather than guessed.
"""

from __future__ import annotations

import re

_ALIASES = {
    "ATL": [], "BOS": [], "BKN": ["BRO", "BK", "BRK"], "CHA": ["CHR", "CHO"], "CHI": [],
    "CLE": [], "DAL": [], "DEN": [], "DET": [], "GSW": ["GS", "GOL", "GSTATE"], "HOU": [],
    "IND": [], "LAC": [], "LAL": [], "MEM": [], "MIA": [], "MIL": [], "MIN": [],
    "NOP": ["NO", "NOR", "NOH", "NOLA"], "NYK": ["NY", "NYN"], "OKC": [], "ORL": [],
    "PHI": [], "PHX": ["PHO"], "POR": [], "SAC": [], "SAS": ["SA", "SAN"], "TOR": [],
    "UTA": ["UTAH", "UTH"], "WAS": ["WSH", "WSW", "WIZ"],
}
_FULL = {
    "ATLANTAHAWKS": "ATL", "BOSTONCELTICS": "BOS", "BROOKLYNNETS": "BKN", "CHARLOTTEHORNETS": "CHA",
    "CHICAGOBULLS": "CHI", "CLEVELANDCAVALIERS": "CLE", "DALLASMAVERICKS": "DAL", "DENVERNUGGETS": "DEN",
    "DETROITPISTONS": "DET", "GOLDENSTATEWARRIORS": "GSW", "HOUSTONROCKETS": "HOU", "INDIANAPACERS": "IND",
    "LACLIPPERS": "LAC", "LOSANGELESCLIPPERS": "LAC", "LOSANGELESLAKERS": "LAL", "MEMPHISGRIZZLIES": "MEM",
    "MIAMIHEAT": "MIA", "MILWAUKEEBUCKS": "MIL", "MINNESOTATIMBERWOLVES": "MIN",
    "NEWORLEANSPELICANS": "NOP", "NEWYORKKNICKS": "NYK", "OKLAHOMACITYTHUNDER": "OKC",
    "ORLANDOMAGIC": "ORL", "PHILADELPHIA76ERS": "PHI", "PHOENIXSUNS": "PHX",
    "PORTLANDTRAILBLAZERS": "POR", "SACRAMENTOKINGS": "SAC", "SANANTONIOSPURS": "SAS",
    "TORONTORAPTORS": "TOR", "UTAHJAZZ": "UTA", "WASHINGTONWIZARDS": "WAS",
}
_LOOKUP = {}
for code, alts in _ALIASES.items():
    for a in [code, *alts]:
        _LOOKUP[a] = code
TEAMS = sorted(_ALIASES)


def canon(team):
    """Canonical tri-code, or the cleaned input unchanged if unknown."""
    if team is None:
        return ""
    raw = re.sub(r"[^A-Za-z0-9]", "", str(team)).upper()
    return _LOOKUP.get(raw) or _FULL.get(raw) or raw


def is_known(team) -> bool:
    return canon(team) in _ALIASES
