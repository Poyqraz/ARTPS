#!/usr/bin/env python3
"""Fail-closed C05/C06 definition + artifact + input-audit readiness.

YAML status labels alone never unlock real_run_allowed. Requires:
  definition_readiness == ready
  artifact_reference_readiness == ready
  input_audit_readiness == passed  (via --input-audit-json)

Author attestation is informational only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import jsonschema
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    REPO_ROOT,
    load_json_schema,
    read_csv_dicts,
    sha256_file,
    validate_rows,
)

REPO = REPO_ROOT
DEFAULT_DEFS = REPO / "reproduction" / "iac2026" / "C05_C06_DEFINITIONS.yaml"
TEMPLATE_NAME = "c05_c06_manifest.template.csv"
HEX64 = re.compile(r"^[a-fA-F0-9]{64}$")
SYNTHETIC_SHA = re.compile(r"^SYNTHETIC_", re.I)

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

REINFER_KEYS = (
    "inference_script",
    "config_path",
    "checkpoint_path",
    "checkpoint_sha256",
    "dataset_manifest",
    "environment_lock",
    "output_prediction_path",
)

FORBIDDEN_PRED_MARKERS = (
    "synthetic_predictions",
    "c05_c06_manifest.template",
    "fixtures/synthetic_",
)


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


def _miss(bag: List[str], msg: str) -> None:
    bag.append(msg)


def _attestation_meta(defs: Mapping[str, Any]) -> Dict[str, Any]:
    block = defs.get("author_attestation")
    if not isinstance(block, Mapping):
        return {"author_attestation_status": "pending", "author_verified": False}
    return {
        "author_attestation_status": str(block.get("status", "pending")),
        "author_verified": bool(block.get("author_verified", False)),
    }


def _resolve(path_str: str, *, repo: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (repo / p).resolve()


def _is_forbidden_evidence_path(path: Path) -> bool:
    s = str(path).replace("\\", "/").lower()
    return any(m in s for m in FORBIDDEN_PRED_MARKERS) or TEMPLATE_NAME in path.name


def check_definition_readiness(defs: Mapping[str, Any]) -> Tuple[str, List[str]]:
    missing: List[str] = []

    rs = defs.get("response_status")
    if isinstance(rs, Mapping) and rs.get("type") == "audit_observation":
        if rs.get("author_verified") is True:
            _miss(missing, "response_status is audit_observation and must not set author_verified=true")

    support = defs.get("support_level_locked")
    if support != "accepted_abstract_reproduction_pending":
        _miss(
            missing,
            f"support_level_locked must remain accepted_abstract_reproduction_pending (got {support!r})",
        )

    for key in P0_FIELDS:
        node = defs.get(key)
        st = _status(node)
        if st != "LOCATED":
            _miss(missing, f"P0 field {key} status must be LOCATED (got {st})")
            continue
        if _value(node) is None or _value(node) == "":
            _miss(missing, f"P0 field {key} is LOCATED but value is empty")

    if _status(defs.get("task_level")) == "LOCATED":
        if _value(defs.get("task_level")) != "image_binary":
            _miss(
                missing,
                f"task_level must be image_binary (got {_value(defs.get('task_level'))!r})",
            )

    if _status(defs.get("threshold_policy")) == "LOCATED":
        policy = _value(defs.get("threshold_policy"))
        if policy == "validation_selected":
            if _status(defs.get("threshold_selection_metric")) != "LOCATED":
                _miss(missing, "validation_selected requires LOCATED threshold_selection_metric")
            elif _value(defs.get("threshold_selection_metric")) != "f1":
                _miss(
                    missing,
                    f"threshold_selection_metric must be f1 for validation_selected "
                    f"(got {_value(defs.get('threshold_selection_metric'))!r})",
                )
        if policy == "fixed_historical":
            # fixed threshold may live on fixed_threshold node; not always in P0 list
            pass

    if _status(defs.get("pr_metric_method")) == "LOCATED":
        pr = _value(defs.get("pr_metric_method"))
        if pr not in ("average_precision", "trapezoidal_pr_auc"):
            _miss(missing, f"pr_metric_method must be average_precision or trapezoidal_pr_auc (got {pr!r})")

    return ("ready" if not missing else "blocked", missing)


def _validate_manifest_structure(
    path: Path, *, repo: Path
) -> Tuple[bool, List[str], Optional[str]]:
    errs: List[str] = []
    if not path.is_file():
        return False, [f"manifest missing: {path}"], None
    if TEMPLATE_NAME in path.name or "template" in path.name.lower():
        return False, [f"manifest must not be a template: {path.name}"], None

    try:
        rows = read_csv_dicts(path)
    except OSError as exc:
        return False, [f"manifest unreadable: {exc}"], None
    if not rows:
        return False, ["manifest has no data rows"], None

    schema_errs = validate_rows(
        rows, load_json_schema("dataset_manifest.schema.json"), coerce_ints=["binary_label"]
    )
    errs.extend(schema_errs)

    ids = [r.get("sample_id", "") for r in rows]
    if len(ids) != len(set(ids)):
        errs.append("manifest sample_id not unique")

    splits = {r.get("split") for r in rows}
    for need in ("train", "validation", "test"):
        if need not in splits:
            errs.append(f"manifest missing split={need}")

    for split in ("validation", "test"):
        labels = {int(r["binary_label"]) for r in rows if r.get("split") == split and r.get("binary_label") not in ("", None)}
        if labels != {0, 1}:
            errs.append(f"manifest {split} must contain both classes 0 and 1 (got {sorted(labels)})")

    for i, r in enumerate(rows):
        digest = str(r.get("sha256") or "")
        if SYNTHETIC_SHA.match(digest) or not HEX64.match(digest):
            errs.append(f"row {i}: sha256 must be real 64-hex (SYNTHETIC_* forbidden for real-run)")
        if not str(r.get("relative_path") or "").strip():
            errs.append(f"row {i}: relative_path empty")
        if not str(r.get("scene_group_id") or "").strip():
            errs.append(f"row {i}: scene_group_id empty")
        if not str(r.get("duplicate_group_id") or "").strip():
            errs.append(f"row {i}: duplicate_group_id empty")

    # Leakage: same sha256 / scene / duplicate across different splits
    def _cross_split(key: str) -> None:
        by_val: Dict[str, set] = {}
        for r in rows:
            v = str(r.get(key) or "")
            if not v:
                continue
            by_val.setdefault(v, set()).add(str(r.get("split")))
        for v, sp in by_val.items():
            if len(sp) > 1:
                errs.append(f"leakage: {key}={v!r} appears in splits {sorted(sp)}")

    _cross_split("sha256")
    _cross_split("scene_group_id")
    _cross_split("duplicate_group_id")

    digest = sha256_file(path)
    return (len(errs) == 0, errs, digest)


def _validate_predictions(
    node: Any, *, repo: Path
) -> Tuple[bool, List[str], Optional[str], Optional[Path]]:
    errs: List[str] = []
    if not isinstance(node, Mapping):
        return False, ["raw_predictions must be a mapping node"], None, None
    st = _status(node)
    if st != "LOCATED":
        return False, [f"raw_predictions status must be LOCATED (got {st})"], None, None
    val = _value(node)
    if isinstance(val, str):
        return (
            False,
            [
                "raw_predictions.value must be {path, sha256}; bare string is rejected "
                "(migrate to mapping form)"
            ],
            None,
            None,
        )
    if not isinstance(val, Mapping):
        return False, ["raw_predictions.value must be a mapping with path and sha256"], None, None
    path_str = val.get("path")
    declared = str(val.get("sha256") or "")
    if not path_str:
        return False, ["raw_predictions.value.path is empty"], None, None
    if not HEX64.match(declared):
        return False, ["raw_predictions.value.sha256 must be 64 hex"], None, None

    path = _resolve(str(path_str), repo=repo)
    if _is_forbidden_evidence_path(path):
        return False, [f"raw_predictions path forbidden as real evidence: {path}"], None, None
    if not path.is_file():
        return False, [f"raw_predictions file missing: {path}"], None, None

    try:
        rows = read_csv_dicts(path)
    except OSError as exc:
        return False, [f"raw_predictions unreadable: {exc}"], None, None
    if not rows:
        return False, ["raw_predictions CSV is empty"], None, None

    schema_errs = validate_rows(
        rows,
        load_json_schema("prediction_table.schema.json"),
        coerce_ints=["y_true"],
        coerce_floats=["anomaly_score"],
    )
    errs.extend(schema_errs)

    splits = {r.get("split") for r in rows}
    if "validation" not in splits or "test" not in splits:
        errs.append("raw_predictions must include validation and test rows")
    test_labels = {
        int(r["y_true"])
        for r in rows
        if r.get("split") == "test" and r.get("y_true") not in ("", None)
    }
    if test_labels != {0, 1}:
        errs.append(f"raw_predictions test split must contain both classes (got {sorted(test_labels)})")

    actual = sha256_file(path)
    if actual.lower() != declared.lower():
        errs.append(f"raw_predictions sha256 mismatch: declared={declared} actual={actual}")

    ok = len(errs) == 0
    return ok, errs, (actual if ok else None), path


def _validate_reinference(node: Any, *, repo: Path) -> Tuple[bool, List[str], Optional[str]]:
    errs: List[str] = []
    if not isinstance(node, Mapping):
        return False, ["re_inference_path_documented must be a mapping node"], None
    st = _status(node)
    if st != "LOCATED":
        return False, [f"re_inference status must be LOCATED (got {st})"], None
    val = _value(node)
    if not isinstance(val, Mapping):
        return False, ["re_inference value must be a mapping contract"], None
    for key in REINFER_KEYS:
        if not val.get(key):
            errs.append(f"re_inference missing required key: {key}")
    if errs:
        return False, errs, None

    must_exist = (
        "inference_script",
        "config_path",
        "checkpoint_path",
        "dataset_manifest",
        "environment_lock",
    )
    paths: Dict[str, Path] = {}
    for key in must_exist:
        p = _resolve(str(val[key]), repo=repo)
        paths[key] = p
        if not p.is_file():
            errs.append(f"re_inference {key} missing on disk: {p}")
        if _is_forbidden_evidence_path(p) and key != "dataset_manifest":
            # dataset_manifest may be under reproduction/; still forbid synthetic fixtures for claim
            pass
        if "synthetic" in str(p).replace("\\", "/").lower() and key in (
            "inference_script",
            "checkpoint_path",
        ):
            errs.append(f"re_inference {key} looks synthetic/forbidden: {p}")

    # output path: existence not required
    _ = _resolve(str(val["output_prediction_path"]), repo=repo)

    declared = str(val.get("checkpoint_sha256") or "")
    if not HEX64.match(declared):
        errs.append("re_inference checkpoint_sha256 must be 64 hex")
    elif paths.get("checkpoint_path") and paths["checkpoint_path"].is_file():
        actual = sha256_file(paths["checkpoint_path"])
        if actual.lower() != declared.lower():
            errs.append(f"checkpoint sha256 mismatch: declared={declared} actual={actual}")
        else:
            return (len(errs) == 0), errs, actual

    return (len(errs) == 0), errs, None


def check_artifact_readiness(
    defs: Mapping[str, Any], *, repo: Path
) -> Tuple[str, List[str], Dict[str, Any]]:
    missing: List[str] = []
    meta: Dict[str, Any] = {
        "manifest_structure_ready": False,
        "dataset_files_audited": False,
        "manifest_sha256": None,
        "predictions_sha256": None,
        "checkpoint_sha256": None,
        "raw_predictions_ok": False,
        "re_inference_ok": False,
    }

    manifest_rel = defs.get("real_manifest_path")
    if not manifest_rel:
        _miss(missing, "real_manifest_path is null")
    else:
        mpath = _resolve(str(manifest_rel), repo=repo)
        ok, m_errs, m_sha = _validate_manifest_structure(mpath, repo=repo)
        meta["manifest_structure_ready"] = ok
        meta["manifest_sha256"] = m_sha if ok else None
        missing.extend(m_errs)

    pred_ok, pred_errs, pred_sha, _pred_path = _validate_predictions(
        defs.get("raw_predictions"), repo=repo
    )
    if pred_ok:
        meta["raw_predictions_ok"] = True
        meta["predictions_sha256"] = pred_sha

    rein_ok, rein_errs, ckpt_sha = _validate_reinference(
        defs.get("re_inference_path_documented"), repo=repo
    )
    if rein_ok:
        meta["re_inference_ok"] = True
        meta["checkpoint_sha256"] = ckpt_sha

    if not (meta["raw_predictions_ok"] or meta["re_inference_ok"]):
        _miss(
            missing,
            "neither raw_predictions nor re_inference_path_documented passed "
            "strict LOCATED artifact checks",
        )
        missing.extend(f"raw_predictions: {e}" for e in pred_errs[:12])
        missing.extend(f"re_inference: {e}" for e in rein_errs[:12])

    artifact_ready = meta["manifest_structure_ready"] and (
        meta["raw_predictions_ok"] or meta["re_inference_ok"]
    )
    return ("ready" if artifact_ready else "blocked"), missing, meta


def check_input_audit(
    audit_path: Optional[Path],
    *,
    repo: Path,
    artifact_meta: Mapping[str, Any],
) -> Tuple[str, List[str]]:
    if audit_path is None:
        return "not_run", ["--input-audit-json not provided (fresh real_evidence audit required)"]

    errs: List[str] = []
    if not audit_path.is_file():
        return "failed", [f"input audit JSON missing: {audit_path}"]

    try:
        prior = json.loads(audit_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return "failed", [f"input audit JSON invalid: {exc}"]

    schema = load_json_schema("input_audit.schema.json")
    v_errs = sorted(jsonschema.Draft202012Validator(schema).iter_errors(prior), key=lambda e: list(e.path))
    if v_errs:
        return "failed", ["input audit schema: " + "; ".join(e.message for e in v_errs[:5])]

    if prior.get("passed") is not True:
        errs.append("input audit passed != true")
    if prior.get("evidence_mode") != "real_evidence":
        errs.append(f"input audit evidence_mode must be real_evidence (got {prior.get('evidence_mode')!r})")
    if prior.get("git_dirty") is not False:
        errs.append("input audit git_dirty must be false")

    for key in ("config_sha256", "manifest_sha256", "predictions_sha256", "checkpoint_sha256", "git_head"):
        val = prior.get(key)
        if key == "checkpoint_sha256" and not artifact_meta.get("checkpoint_sha256") and not artifact_meta.get(
            "re_inference_ok"
        ):
            # checkpoint may be null if using predictions-only path
            continue
        if val is None or val == "":
            if key in ("config_sha256", "manifest_sha256", "predictions_sha256", "git_head"):
                errs.append(f"input audit {key} must be non-null for real_evidence")

    # Hash match against artifacts we know
    m_sha = artifact_meta.get("manifest_sha256")
    if m_sha and prior.get("manifest_sha256"):
        if str(prior["manifest_sha256"]).lower() != str(m_sha).lower():
            errs.append("input audit manifest_sha256 does not match current manifest file")

    p_sha = artifact_meta.get("predictions_sha256")
    if p_sha and prior.get("predictions_sha256"):
        if str(prior["predictions_sha256"]).lower() != str(p_sha).lower():
            errs.append("input audit predictions_sha256 does not match current predictions file")

    c_sha = artifact_meta.get("checkpoint_sha256")
    if c_sha and prior.get("checkpoint_sha256"):
        if str(prior["checkpoint_sha256"]).lower() != str(c_sha).lower():
            errs.append("input audit checkpoint_sha256 does not match current checkpoint")

    if errs:
        return "failed", errs
    return "passed", []


def evaluate(
    defs: Mapping[str, Any],
    *,
    repo: Path = REPO,
    input_audit_json: Optional[Path] = None,
) -> Dict[str, Any]:
    attest = _attestation_meta(defs)
    def_status, def_missing = check_definition_readiness(defs)
    art_status, art_missing, art_meta = check_artifact_readiness(defs, repo=repo)
    audit_status, audit_missing = check_input_audit(
        input_audit_json, repo=repo, artifact_meta=art_meta
    )

    real_ok = (
        def_status == "ready"
        and art_status == "ready"
        and audit_status == "passed"
    )

    all_missing = list(def_missing) + list(art_missing) + list(audit_missing)

    return {
        "definition_readiness": def_status,
        "artifact_reference_readiness": art_status,
        "input_audit_readiness": audit_status,
        "real_run_allowed": real_ok,
        "manifest_structure_ready": bool(art_meta.get("manifest_structure_ready")),
        "dataset_files_audited": False,
        "manifest_sha256": art_meta.get("manifest_sha256"),
        "predictions_sha256": art_meta.get("predictions_sha256"),
        "checkpoint_sha256": art_meta.get("checkpoint_sha256"),
        "author_attestation_status": attest["author_attestation_status"],
        "author_verified": attest["author_verified"],
        "missing": all_missing,
        "definitions_path": str(DEFAULT_DEFS.relative_to(repo)) if DEFAULT_DEFS.exists() else None,
        "p0_fields_checked": list(P0_FIELDS),
        "notes": (
            "Layered readiness: definition + on-disk artifact references + passing "
            "real_evidence input audit. YAML LOCATED labels are insufficient alone. "
            "dataset_files_audited remains false here; use audit_reproduction_inputs.py "
            "for per-file image SHA audit. Author attestation is informational only. "
            "Prior audit JSON must hash-match current artifacts; metrics runner still "
            "requires a fresh audit_inputs() call."
        ),
        # Back-compat alias for older tests/docs glancing at readiness
        "readiness": "ready" if real_ok else "blocked",
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--definitions", type=Path, default=DEFAULT_DEFS)
    ap.add_argument(
        "--input-audit-json",
        type=Path,
        default=None,
        help="Optional prior/fresh input_audit.json (must pass real_evidence schema + hash match).",
    )
    ap.add_argument("--json", action="store_true", help="Print JSON only")
    args = ap.parse_args(argv)

    defs_path = args.definitions
    if not defs_path.is_file():
        print(f"definitions not found: {defs_path}", file=sys.stderr)
        return 2

    result = evaluate(
        _load_defs(defs_path),
        repo=REPO,
        input_audit_json=args.input_audit_json,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"definition_readiness: {result['definition_readiness']}")
        print(f"artifact_reference_readiness: {result['artifact_reference_readiness']}")
        print(f"input_audit_readiness: {result['input_audit_readiness']}")
        print(f"real_run_allowed: {result['real_run_allowed']}")
        print(f"author_attestation_status: {result['author_attestation_status']}")
        print(f"author_verified: {result['author_verified']}")
        print("missing:")
        for m in result["missing"]:
            print(f"  - {m}")
    return 0 if result["real_run_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
