import sys
from pathlib import Path

# Make `import paper` work from the repo root regardless of the hyphenated dir name.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
