#!/usr/bin/env python3
"""Check whether C05/C06 definitions are evidence-ready for a real run.

Reads reproduction/iac2026/C05_C06_DEFINITIONS.yaml only. Does not guess.
Author attestation metadata is reported but never sufficient alone.
Expected outcome on HEAD without closed artifacts: readiness=blocked.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

REPO = Path(__file__).resolve().parents[2]
DEFAULT_DEFS = REPO / "reproduction" / "iac2026" / "C05_C06_DEFINITIONS.yaml"
TEMPLATE_NAME = "c05_c06_manifest.template.csv"

# P0 fields that must be evidence-closed (not UNKNOWN / not claim-only) before real run.
P0_FIELDS = (
    "task_level",
    "positive_class",
    "anomaly_score_definition",
    "score_orientation",
    "sample_unit",
    "train_split",
    "validation_split",
    "test_split",
    "label_semantics",
    "threshold_policy",
    "threshold_selection_metric",
    "pr_metric_method",
    "c06_baseline_identity",
)
# raw_predictions XOR re_inference handled separately (either evidence path suffices).

CLOSED_STATUSES = frozenset({"LOCATED", "PARTIALLY_LOCATED"})
# PARTIALLY_LOCATED alone is not enough for several P0 keys — see per-field rules.

def _load_defs(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"definitions must be a mapping: {path}")
    return data


def _status(node: Any) -> str:
    if not isinstance(node, Mapping):
        return "UNKNOWN_EVIDENCE_NOT_LOCATED"
    return str(node.get("status") or "UNKNOWN_EVIDENCE_NOT_LOCATED")


def _value(node: Any) -> Any:
    if not isinstance(node, Mapping):
        return None
    return node.get("value")


def _missing(msg: str, missing: List[str]) -> None:
    missing.append(msg)


def _attestation_meta(defs: Mapping[str, Any]) -> Dict[str, Any]:
    """Read author_attestation block; never treat as artifact closure."""
    block = defs.get("author_attestation")
    if not isinstance(block, Mapping):
        return {
            "author_attestation_status": "pending",
            "author_verified": False,
        }
    status = block.get("status", "pending")
    verified = bool(block.get("author_verified", False))
    return {
        "author_attestation_status": str(status),
        "author_verified": verified,
    }


def evaluate(defs: Mapping[str, Any], *, repo: Path = REPO) -> Dict[str, Any]:
    missing: List[str] = []
    attest = _attestation_meta(defs)

    # Audit observation / response_status is never a readiness unlock.
    rs = defs.get("response_status")
    if isinstance(rs, Mapping) and rs.get("type") == "audit_observation":
        if rs.get("author_verified") is True:
            _missing(
                "response_status is audit_observation and must not set author_verified=true",
                missing,
            )

    support = defs.get("support_level_locked")
    if support != "accepted_abstract_reproduction_pending":
        _missing(
            f"support_level_locked must remain accepted_abstract_reproduction_pending (got {support!r})",
            missing,
        )

    # --- P0 field closure ---
    for key in P0_FIELDS:
        node = defs.get(key)
        st = _status(node)
        if st == "UNKNOWN_EVIDENCE_NOT_LOCATED":
            _missing(f"P0 field {key} is UNKNOWN_EVIDENCE_NOT_LOCATED", missing)
        elif st == "MANUSCRIPT_CLAIM_ONLY":
            _missing(f"P0 field {key} is MANUSCRIPT_CLAIM_ONLY (not executable/raw)", missing)
        elif st == "PATH_REFERENCE_ONLY":
            _missing(f"P0 field {key} is PATH_REFERENCE_ONLY (not verified)", missing)

    # task_level must be exactly image_binary when closed (harness constraint)
    tl = _value(defs.get("task_level"))
    tl_st = _status(defs.get("task_level"))
    if tl_st in CLOSED_STATUSES or tl is not None:
        if tl != "image_binary":
            _missing(
                f"task_level must be image_binary for current harness (got {tl!r}, status={tl_st})",
                missing,
            )

    # positive class must be non-null when not UNKNOWN
    if _status(defs.get("positive_class")) != "UNKNOWN_EVIDENCE_NOT_LOCATED":
        if _value(defs.get("positive_class")) is None:
            _missing("positive_class status is set but value is null", missing)

    # threshold_policy: PARTIALLY_LOCATED validation_selected still needs selection metric
    tp_st = _status(defs.get("threshold_policy"))
    if tp_st == "PARTIALLY_LOCATED":
        if _status(defs.get("threshold_selection_metric")) == "UNKNOWN_EVIDENCE_NOT_LOCATED":
            _missing(
                "threshold_policy is only PARTIALLY_LOCATED; threshold_selection_metric still UNKNOWN",
                missing,
            )

    # PR method must be one of the harness enums when closed
    pr = _value(defs.get("pr_metric_method"))
    pr_st = _status(defs.get("pr_metric_method"))
    if pr_st != "UNKNOWN_EVIDENCE_NOT_LOCATED":
        if pr not in ("average_precision", "trapezoidal_pr_auc"):
            _missing(f"pr_metric_method must be average_precision or trapezoidal_pr_auc (got {pr!r})", missing)

    # C06 identity must be an explicit non-null string when closed
    if _status(defs.get("c06_baseline_identity")) != "UNKNOWN_EVIDENCE_NOT_LOCATED":
        if not _value(defs.get("c06_baseline_identity")):
            _missing("c06_baseline_identity status set but value empty", missing)

    # Manifest: must point to a real non-template CSV with ≥1 data row
    manifest_rel = defs.get("real_manifest_path")
    if not manifest_rel:
        _missing("real_manifest_path is null (template-only; no pinned dataset manifest)", missing)
    else:
        mpath = repo / str(manifest_rel) if not Path(str(manifest_rel)).is_absolute() else Path(str(manifest_rel))
        if TEMPLATE_NAME in mpath.name:
            _missing(f"real_manifest_path must not be the template ({mpath.name})", missing)
        elif not mpath.is_file():
            _missing(f"real_manifest_path missing on disk: {mpath}", missing)
        else:
            lines = [ln for ln in mpath.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if len(lines) < 2:
                _missing("real manifest has header only (no data rows)", missing)

    # Splits must be pinned (non-null) when status closed
    for split_key in ("train_split", "validation_split", "test_split"):
        if _status(defs.get(split_key)) != "UNKNOWN_EVIDENCE_NOT_LOCATED" and _value(defs.get(split_key)) is None:
            _missing(f"{split_key} closed status without value", missing)

    # Raw predictions OR documented re-inference path
    raw_ok = (
        _status(defs.get("raw_predictions")) != "UNKNOWN_EVIDENCE_NOT_LOCATED"
        and _value(defs.get("raw_predictions")) is not None
    )
    reinfer_ok = (
        _status(defs.get("re_inference_path_documented")) != "UNKNOWN_EVIDENCE_NOT_LOCATED"
        and _value(defs.get("re_inference_path_documented")) is not None
    )
    if not raw_ok and not reinfer_ok:
        _missing(
            "neither raw_predictions nor re_inference_path_documented is evidence-closed",
            missing,
        )

    # author_verified alone never unlocks; if false while somehow missing empty, still blocked via P0
    if not attest["author_verified"]:
        # Informational gate: keep blocked messaging explicit when attestation pending
        # (does not replace artifact requirements).
        pass

    blocked = len(missing) > 0
    # Attestation true without artifacts: still blocked because missing list non-empty.
    # Attestation never clears missing.
    return {
        "readiness": "blocked" if blocked else "ready",
        "real_run_allowed": False if blocked else True,
        "author_attestation_status": attest["author_attestation_status"],
        "author_verified": attest["author_verified"],
        "missing": missing,
        "definitions_path": str(DEFAULT_DEFS.relative_to(repo)) if DEFAULT_DEFS.exists() else None,
        "p0_fields_checked": list(P0_FIELDS),
        "notes": (
            "Manuscript-claimed metrics (0.894/0.847/0.823/0.856) never authorize real_run_allowed. "
            "Audit observations are not author attestations. "
            "author_verified alone never unlocks a real run; artifacts remain required. "
            "This checker refuses to invent splits, labels, thresholds, or baseline identity."
        ),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--definitions", type=Path, default=DEFAULT_DEFS)
    ap.add_argument("--json", action="store_true", help="Print JSON only")
    args = ap.parse_args(argv)

    defs_path = args.definitions
    if not defs_path.is_file():
        print(f"definitions not found: {defs_path}", file=sys.stderr)
        return 2

    result = evaluate(_load_defs(defs_path), repo=REPO)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"readiness: {result['readiness']}")
        print(f"real_run_allowed: {result['real_run_allowed']}")
        print(f"author_attestation_status: {result['author_attestation_status']}")
        print(f"author_verified: {result['author_verified']}")
        print("missing:")
        for m in result["missing"]:
            print(f"  - {m}")
    return 0 if result["real_run_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
