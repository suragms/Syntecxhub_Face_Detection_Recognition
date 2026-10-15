"""Optional GUI placeholder intentionally deferred until the CLI is manually validated.

The internship requirements are completely accessible through ``main.py``.
A future Tkinter interface must call the existing core services instead of
copying model, database, recognition, or camera lifecycle logic here.
"""
from __future__ import annotations

import sys


def main() -> int:
    print(
        "The optional desktop GUI is not included in this CLI-first release. "
        "Use 'python main.py --help' for the complete supported application."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
