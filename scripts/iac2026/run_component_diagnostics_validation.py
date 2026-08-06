"""Optional instrumented component-diagnostics runner (validation only).

Without ARTPS_DATASET_ROOT: refuse (fail closed). Default audit path writes
unavailable reasons from committed JSONL via audit_independent_eval_validation_sanity.py
and does not require this runner.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="validation")
    args = p.parse_args(argv)
    if str(args.split).strip().lower() != "validation":
        raise SystemExit("refusing non-validation split (final test closed)")
    root = (os.environ.get("ARTPS_DATASET_ROOT") or "").strip()
    if not root:
        raise SystemExit(
            "ARTPS_DATASET_ROOT unset: instrumented component diagnostics refused "
            "(fail closed). Use audit script JSONL unavailable CSV instead."
        )
    # ponytail: full GPU instrumented path deferred; score-parity rerun is a separate operator step.
    raise SystemExit(
        "instrumented GPU diagnostics not executed in this PR build; "
        "committed component_diagnostics_v1.csv uses "
        "unavailable_requires_instrumented_validation_rerun"
    )


if __name__ == "__main__":
    raise SystemExit(main())
