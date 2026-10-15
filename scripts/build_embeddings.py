"""Validate and rewrite a fresh local embedding database generation."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import main


if __name__ == "__main__":
    raise SystemExit(main(["--rebuild", *sys.argv[1:]]))
