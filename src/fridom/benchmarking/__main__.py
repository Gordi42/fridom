"""Run the benchmarking CLI via `python -m fridom.benchmarking`."""
from __future__ import annotations

import sys

from fridom.benchmarking.cli import main

if __name__ == "__main__":
    sys.exit(main())
