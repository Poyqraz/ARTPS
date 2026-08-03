#!/usr/bin/env python3
"""Thin CLI shim — ledger path scripts/benchmark_cv_core_speed.py stays stable."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "iac2026"))

from benchmark_cv_core_speed import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
