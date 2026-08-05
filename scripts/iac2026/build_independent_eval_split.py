"""Deterministic group-aware split builder for independent_eval_v1."""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import REPO_ROOT, read_csv_dicts, sha256_file, write_json, write_text
from independent_eval_contract import load_protocol_lock


class SplitContractError(RuntimeError):
    """Fail-closed split builder errors."""


def _lock_value(node: Any) -> Any:
    if isinstance(node, Mapping) and "value" in node:
        return node["value"]
    return node


def select_split_ratios(included_n: int) -> Optional[Tuple[float, float, float]]:
    """Return (train, validation, test) from included N only — never from model scores."""
    if included_n >= 360:
        return (0.70, 0.15, 0.15)
    if included_n >= 240:
        return (0.60, 0.20, 0.20)
    return None


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
            v = str(r.get(key) or "").strip()
            if not v or v.upper() == "UNKNOWN":
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


def _group_key(row: Mapping[str, str]) -> str:
    # Prefer duplicate_group_id; fall back to sha256 so exact dups stay together.
    return str(row.get("duplicate_group_id") or row.get("sha256") or row.get("sample_id"))


def assign_splits(
    rows: List[Dict[str, str]],
    ratios: Tuple[float, float, float],
    seed: int,
) -> List[Dict[str, str]]:
    """Group-aware, label-stratified deterministic assignment."""
    rng = random.Random(seed)
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        groups[_group_key(r)].append(r)

    by_label_groups: Dict[int, List[List[Dict[str, str]]]] = {0: [], 1: []}
    for members in groups.values():
        labels = [int(m["binary_label"]) for m in members]
        maj = 1 if sum(labels) > len(labels) / 2.0 else 0
        by_label_groups[maj].append(members)

    for lab in (0, 1):
        rng.shuffle(by_label_groups[lab])

    def split_list(items: List[List[Dict[str, str]]]) -> Dict[str, List[List[Dict[str, str]]]]:
        # Assign whole groups by cumulative size toward ratio targets.
        sizes = [sum(len(g) for g in items)]
        n_lab = sizes[0] if sizes else 0
        t_train = int(round(ratios[0] * n_lab))
        t_val = int(round(ratios[1] * n_lab))
        out: Dict[str, List[List[Dict[str, str]]]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        c_train = c_val = 0
        for g in items:
            gsz = len(g)
            if c_train + gsz <= t_train or (c_train < t_train and c_val >= t_val):
                out["train"].append(g)
                c_train += gsz
            elif c_val + gsz <= t_val or c_val < t_val:
                out["validation"].append(g)
                c_val += gsz
            else:
                out["test"].append(g)
        # If any bucket empty but items exist, steal one group from train.
        for sp in ("validation", "test"):
            if not out[sp] and out["train"]:
                out[sp].append(out["train"].pop())
        return out

    buckets = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for lab in (0, 1):
        parts = split_list(by_label_groups[lab])
        for sp in buckets:
            buckets[sp].extend(parts[sp])

    assigned: List[Dict[str, str]] = []
    label_counts = {s: Counter() for s in buckets}
    for sp, glist in buckets.items():
        for members in glist:
            for m in members:
                row = dict(m)
                row["split"] = sp
                assigned.append(row)
                label_counts[sp][int(row["binary_label"])] += 1

    for sp in ("validation", "test"):
        if label_counts[sp][0] == 0 or label_counts[sp][1] == 0:
            raise SplitContractError(
                f"{sp} missing a class after stratified group assignment: {dict(label_counts[sp])}"
            )
    return assigned


def build_split_report(
    *,
    input_manifest: Path,
    output_manifest: Path,
    lock: Mapping[str, Any],
    seed: Optional[int],
    rows: Optional[Sequence[Mapping[str, str]]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    rows = list(rows) if rows is not None else (
        read_csv_dicts(output_manifest) if output_manifest.is_file() else []
    )
    group_fields = list(_lock_value(lock.get("group_fields")) or [])
    leakage = check_group_leakage(rows, group_fields)
    report: Dict[str, Any] = {
        "passed": len(leakage) == 0,
        "protocol_id": lock.get("protocol_id"),
        "protocol_version": lock.get("protocol_version"),
        "split_method": _lock_value(lock.get("split_method")),
        "split_seed": seed,
        "split_ratios": _lock_value(lock.get("split_ratios")),
        "input_manifest_sha256": sha256_file(input_manifest) if input_manifest.is_file() else None,
        "output_manifest_sha256": sha256_file(output_manifest) if output_manifest.is_file() else None,
        "group_fields": group_fields,
        "leakage_errors": leakage,
        "class_balance": class_balance_report(rows),
        "mission_instrument_distribution": distribution_report(rows, ["mission", "instrument"]),
        "test_freeze_policy": _lock_value(lock.get("test_freeze_policy")),
        "included_n": len(rows),
        "test_sample_ids": [r["sample_id"] for r in rows if r.get("split") == "test"],
    }
    if extra:
        report.update(extra)
    return report


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "UNKNOWN"


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
    ap.add_argument(
        "--write-build-reports",
        action="store_true",
        help="Also write dataset_build split_report.md / class_distribution / leakage.",
    )
    args = ap.parse_args(argv)

    try:
        lock, lock_sha, _ = load_protocol_lock()
        assert_no_aggregate_quota(str(args.input_manifest))
        freeze = args.freeze_marker or (args.output_manifest.parent / "TEST_SPLIT_FROZEN")
        refuse_frozen_test_mutation(freeze_marker=freeze, output_manifest=args.output_manifest)
        assert_ratios_selected(lock)
    except SplitContractError as exc:
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

    rows = read_csv_dicts(args.input_manifest)
    included = [
        r
        for r in rows
        if r.get("inclusion_status") == "included"
        and r.get("adjudication_status") == "resolved"
        and r.get("binary_label") in ("0", "1")
    ]
    n = len(included)
    policy = select_split_ratios(n)
    if policy is None:
        report = {
            "passed": False,
            "dataset_readiness": "insufficient",
            "included_n": n,
            "error": "included N < 240; refuse split freeze",
        }
        write_json(args.report_json, report)
        print("SPLIT REFUSED: included N < 240", file=sys.stderr)
        return 2

    lock_ratios = _lock_value(lock.get("split_ratios"))
    if isinstance(lock_ratios, Mapping):
        expected = (
            float(lock_ratios["train"]),
            float(lock_ratios["validation"]),
            float(lock_ratios["test"]),
        )
    elif isinstance(lock_ratios, (list, tuple)) and len(lock_ratios) == 3:
        expected = (float(lock_ratios[0]), float(lock_ratios[1]), float(lock_ratios[2]))
    else:
        raise SplitContractError(f"unusable split_ratios value: {lock_ratios!r}")

    if tuple(round(x, 4) for x in expected) != tuple(round(x, 4) for x in policy):
        raise SplitContractError(
            f"lock split_ratios {expected} != policy for N={n} {policy}"
        )

    seed = args.seed
    if seed is None:
        seed = int(_lock_value(lock.get("split_seed")) or 0)
    assigned = assign_splits(included, expected, seed)
    leakage = check_group_leakage(assigned, list(_lock_value(lock.get("group_fields")) or []))
    if leakage:
        report = {"passed": False, "leakage_errors": leakage}
        write_json(args.report_json, report)
        print("SPLIT LEAKAGE:", leakage, file=sys.stderr)
        return 2

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(assigned[0].keys())
    with args.output_manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in assigned:
            w.writerow(row)

    freeze = args.freeze_marker or (args.output_manifest.parent / "TEST_SPLIT_FROZEN")
    test_ids = [r["sample_id"] for r in assigned if r["split"] == "test"]
    freeze_payload = {
        "protocol_id": lock.get("protocol_id"),
        "protocol_version": lock.get("protocol_version"),
        "split_seed": seed,
        "split_ratios": expected,
        "test_sample_ids": test_ids,
        "test_manifest_sha256": sha256_file(args.output_manifest),
        "builder_commit_sha": _git_head(),
        "included_n": n,
    }
    write_json(freeze, freeze_payload)

    report = build_split_report(
        input_manifest=args.input_manifest,
        output_manifest=args.output_manifest,
        lock=lock,
        seed=seed,
        rows=assigned,
        extra={
            "passed": True,
            "protocol_lock_sha256": lock_sha,
            "builder_commit_sha": freeze_payload["builder_commit_sha"],
            "policy_ratios": policy,
        },
    )
    write_json(args.report_json, report)

    if args.write_build_reports:
        build_dir = REPO_ROOT / "results" / "iac2026" / "dataset_build"
        write_json(build_dir / "split_report.json", report)
        write_json(build_dir / "group_leakage_report.json", {"leakage_errors": leakage})
        # class distribution CSV
        bal = class_balance_report(assigned)
        cd_path = build_dir / "class_distribution.csv"
        with cd_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["split", "label_0", "label_1", "total"])
            for sp in ("train", "validation", "test"):
                c = bal.get(sp, {})
                w.writerow([sp, c.get(0, 0), c.get(1, 0), sum(c.values())])
        md = [
            "# Split report (independent_eval_v1)",
            "",
            f"- included_n: **{n}**",
            f"- ratios: **{expected}**",
            f"- seed: **{seed}**",
            f"- test_manifest_sha256: `{freeze_payload['test_manifest_sha256']}`",
            f"- builder_commit_sha: `{freeze_payload['builder_commit_sha']}`",
            f"- leakage_errors: {leakage or 'none'}",
            "",
            "## Class balance",
            "",
            "```json",
            json.dumps(bal, indent=2),
            "```",
            "",
        ]
        write_text(build_dir / "split_report.md", "\n".join(md))

    print(f"Wrote split manifest ({n} rows) -> {args.output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
