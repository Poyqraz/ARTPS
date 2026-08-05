"""Deterministic group-aware split builder for independent_eval_v1 (contract only)."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import read_csv_dicts, sha256_file, write_json
from independent_eval_contract import load_protocol_lock


class SplitContractError(RuntimeError):
    """Fail-closed split builder errors."""


def _lock_value(node: Any) -> Any:
    if isinstance(node, Mapping) and "value" in node:
        return node["value"]
    return node


def assert_ratios_selected(lock: Mapping[str, Any]) -> None:
    ratios = lock.get("split_ratios") or {}
    status = ratios.get("status") if isinstance(ratios, Mapping) else None
    if status == "PENDING_RATIO_SELECTION" or _lock_value(ratios) is None:
        raise SplitContractError(
            "split_ratios still PENDING_RATIO_SELECTION; refuse inventing ratios "
            "(including from historical 2847/1247/892/708). Update protocol lock after "
            "labeled volume is known and bump protocol_version."
        )


def assert_no_aggregate_quota(notes: str) -> None:
    banned = ("2847", "1247", "892", "708")
    low = notes.lower()
    if any(b in notes for b in banned) and ("quota" in low or "aggregate" in low or "target" in low):
        raise SplitContractError(
            "historical accepted abstract counts must not be used as split quotas"
        )


def check_group_leakage(rows: Sequence[Mapping[str, str]], group_fields: Sequence[str]) -> List[str]:
    errs: List[str] = []
    for key in group_fields:
        by_val: Dict[str, Set[str]] = defaultdict(set)
        for r in rows:
            v = str(r.get(key) or "")
            if not v:
                continue
            by_val[v].add(str(r.get("split")))
        for v, splits in by_val.items():
            if len(splits) > 1:
                errs.append(f"leakage: {key}={v!r} spans splits {sorted(splits)}")
    return errs


def class_balance_report(rows: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
    by_split: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        try:
            by_split[str(r.get("split"))][int(r["binary_label"])] += 1
        except (KeyError, TypeError, ValueError):
            continue
    return {s: dict(c) for s, c in sorted(by_split.items())}


def distribution_report(
    rows: Sequence[Mapping[str, str]], fields: Sequence[str]
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for field in fields:
        by_split: Dict[str, Counter] = defaultdict(Counter)
        for r in rows:
            by_split[str(r.get("split"))][str(r.get(field, ""))] += 1
        out[field] = {s: dict(c) for s, c in sorted(by_split.items())}
    return out


def refuse_frozen_test_mutation(
    *,
    freeze_marker: Path,
    output_manifest: Path,
) -> None:
    if freeze_marker.is_file() and output_manifest.is_file():
        raise SplitContractError(
            f"test split frozen at {freeze_marker}; refusing mutation of {output_manifest}. "
            "Create a new protocol/run version instead."
        )


def build_split_report(
    *,
    input_manifest: Path,
    output_manifest: Path,
    lock: Mapping[str, Any],
    seed: Optional[int],
) -> Dict[str, Any]:
    rows = read_csv_dicts(input_manifest) if input_manifest.is_file() else []
    group_fields = list(_lock_value(lock.get("group_fields")) or [])
    leakage = check_group_leakage(rows, group_fields)
    return {
        "protocol_id": lock.get("protocol_id"),
        "split_method": _lock_value(lock.get("split_method")),
        "split_seed": seed,
        "input_manifest_sha256": sha256_file(input_manifest) if input_manifest.is_file() else None,
        "output_manifest_sha256": sha256_file(output_manifest) if output_manifest.is_file() else None,
        "group_fields": group_fields,
        "leakage_errors": leakage,
        "class_balance": class_balance_report(rows),
        "mission_instrument_distribution": distribution_report(rows, ["mission", "instrument"]),
        "test_freeze_policy": _lock_value(lock.get("test_freeze_policy")),
        "note": "No split assignment performed while split_ratios are PENDING_RATIO_SELECTION.",
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-manifest", type=Path, required=True)
    ap.add_argument("--output-manifest", type=Path, required=True)
    ap.add_argument("--report-json", type=Path, required=True)
    ap.add_argument("--freeze-marker", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--allow-pending-ratios",
        action="store_true",
        help="Only emit a contract refusal report (still exit 2 if ratios pending).",
    )
    args = ap.parse_args(argv)

    try:
        lock, lock_sha, _ = load_protocol_lock()
        assert_no_aggregate_quota(str(args.input_manifest))
        freeze = args.freeze_marker or (args.output_manifest.parent / "TEST_SPLIT_FROZEN")
        refuse_frozen_test_mutation(freeze_marker=freeze, output_manifest=args.output_manifest)
        assert_ratios_selected(lock)
    except SplitContractError as exc:
        # Still write a diagnostic report when possible
        try:
            lock, lock_sha, _ = load_protocol_lock()
        except Exception:
            lock, lock_sha = {}, None
        report = {
            "passed": False,
            "protocol_lock_sha256": lock_sha,
            "error": str(exc),
        }
        if lock:
            report.update(
                build_split_report(
                    input_manifest=args.input_manifest,
                    output_manifest=args.output_manifest,
                    lock=lock,
                    seed=args.seed,
                )
            )
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.report_json, report)
        print(f"SPLIT CONTRACT FAILED: {exc}", file=sys.stderr)
        return 2

    # Ratios selected path reserved for future data work; not implemented here.
    print("split ratios selected but assignment not implemented in this contract PR", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
