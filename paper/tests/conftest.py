import sys
from pathlib import Path

# Make `import paper` work from the repo root regardless of the hyphenated dir name.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def small_league(monkeypatch):
    """The synthetic leagues in the tests have 20 players; production refuses a preseason sheet under 250.
    Tests of the guard itself put the real threshold back."""
    from paper import sources
    monkeypatch.setattr(sources, "MIN_ROWS", 5)
