import sys
from pathlib import Path

# Ensure project root is on sys.path so signal_pipeline resolves correctly.
# Without this, pytest treats tests/ as a package (tests/__init__.py exists),
# which causes tests.signal_pipeline to shadow the top-level signal_pipeline module.
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
