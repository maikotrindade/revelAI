"""Entry point for ``python -m revelai``."""

from __future__ import annotations

import sys

from revelai.cli import main

if __name__ == "__main__":
    sys.exit(main())
