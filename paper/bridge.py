"""Load the sibling model modules (config, state, game_sim, ...).

The repo root is itself a package (it has an __init__.py with relative imports)
but its directory name contains hyphens, so `import nba-projections-check` is
impossible. Load it under a stable alias instead; nothing else in `paper/`
depends on the model package, so only slate.py needs this.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ALIAS = "nbaproj"


def model():
    if ALIAS in sys.modules:
        return sys.modules[ALIAS]
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(ALIAS, root / "__init__.py",
                                                  submodule_search_locations=[str(root)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[ALIAS] = mod
    spec.loader.exec_module(mod)
    return mod
