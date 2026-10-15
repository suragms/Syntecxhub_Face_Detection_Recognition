"""Convenience entry point for consent-based person registration."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import main


if __name__ == "__main__":
    # The main CLI validates mode-specific arguments and uses the same services.
    raise SystemExit(main(["--register", "webcam", *sys.argv[1:]]))
