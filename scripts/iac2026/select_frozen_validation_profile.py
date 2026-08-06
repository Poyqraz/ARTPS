"""Select frozen ARTPS validation profile from predeclared profile prediction CSVs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parents[1]
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(_REPO))

from _common import REPO_ROOT, sha256_file, write_json  # noqa: E402
from detection_metrics_lib import (  # noqa: E402
    average_precision,
    binary_auroc,
    confusion,
    f1_precision_recall,
    orient_scores,
    select_threshold_on_validation,
)
from src.utils.image_enhancement import ENHANCE_PROFILES  # noqa: E402

# Leaderboard-only profiles (AMP / exploratory configs must not appear here).
DEFAULT_PROFILES: tuple[tuple[str, str, str], ...] = (
    (
        "artps_full_frozen_raw_clf_on_v1",
        "results/iac2026/independent_eval_v1/validation/artps_full_frozen_raw_clf_on_v1/predictions.csv",
        "reproduction/iac2026/configs/independent_eval_artps_full_frozen.yaml",
    ),
    (
        "artps_full_frozen_raw_clf_off_v1",
        "results/iac2026/independent_eval_v1/validation/artps_full_frozen_raw_clf_off_v1/predictions.csv",
        "reproduction/iac2026/configs/independent_eval_artps_full_frozen_no_classifier.yaml",
    ),
    (
        "artps_full_frozen_mars_clf_on_v1",
        "results/iac2026/independent_eval_v1/validation/artps_full_frozen_mars_clf_on_v1/predictions.csv",
        "reproduction/iac2026/configs/independent_eval_artps_full_frozen_mars.yaml",
    ),
    (
        "artps_full_frozen_mars_clf_off_v1",
        "results/iac2026/independent_eval_v1/validation/artps_full_frozen_mars_clf_off_v1/predictions.csv",
        "reproduction/iac2026/configs/independent_eval_artps_full_frozen_mars_no_classifier.yaml",
    ),
)

ALLOWED_CONFIG_IDS = frozenset(cid for cid, _, _ in DEFAULT_PROFILES)

PROFILE_META = {
    "artps_full_frozen_raw_clf_on_v1": {"classifier_off": False, "raw_rgb": True},
    "artps_full_frozen_raw_clf_off_v1": {"classifier_off": True, "raw_rgb": True},
    "artps_full_frozen_mars_clf_on_v1": {"classifier_off": False, "raw_rgb": False},
    "artps_full_frozen_mars_clf_off_v1": {"classifier_off": True, "raw_rgb": False},
}

SELECTION_RULE = [
    "highest_validation_average_precision",
    "tie_highest_validation_auroc",
    "tie_simpler_classifier_off",
    "tie_raw_rgb_v1",
]


def _read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return data


def _validation_arrays(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    y: list[int] = []
    s: list[float] = []
    for row in rows:
        if row.get("split") != "validation":
            continue
        y.append(int(row["y_true"]))
        s.append(float(row["anomaly_score"]))
    return np.asarray(y, dtype=np.int32), np.asarray(s, dtype=np.float64)


def _profile_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    y, s_raw = _validation_arrays(rows)
    if y.size == 0:
        raise ValueError("no validation rows in predictions")
    if any(r.get("split") == "test" for r in rows):
        raise ValueError("test-split rows forbidden in validation selection predictions")
    s = orient_scores(s_raw, higher_means_anomalous=True)
    ap = average_precision(y, s)
    auroc = binary_auroc(y, s)
    threshold, f1_at_threshold = select_threshold_on_validation(
        y, s, metric="f1", tie_break="highest_threshold"
    )
    if threshold is None:
        raise ValueError("threshold selection failed on validation")
    y_pred = (s >= float(threshold)).astype(np.int32)
    cm = confusion(y, y_pred)
    f1, precision, recall = f1_precision_recall(cm)
    return {
        "validation_sample_count": int(y.size),
        "validation_n": int(y.size),
        "average_precision": ap,
        "auroc": auroc,
        "selected_threshold": float(threshold),
        "validation_f1_at_threshold": f1_at_threshold,
        "F1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
    }


def _checkpoint_hashes(cfg: dict[str, Any]) -> dict[str, str | None]:
    return {
        "autoencoder_sha256": (cfg.get("autoencoder") or {}).get("checkpoint_sha256"),
        "depth_sha256": (cfg.get("depth") or {}).get("checkpoint_sha256"),
        "classifier_sha256": (cfg.get("depth_classifier") or {}).get("checkpoint_sha256"),
    }


def _assert_preprocessing_contracts(profiles: list[dict[str, Any]]) -> None:
    mars_cfgs = [
        p["config_snapshot"]
        for p in profiles
        if p.get("preprocessing_profile") == "mars_enhancement_v1"
    ]
    if len(mars_cfgs) >= 2:
        keys = (
            "preprocessing_profile",
            "autoencoder",
            "depth",
            "operational_masks",
            "priority_buffer",
        )
        ref = {k: mars_cfgs[0].get(k) for k in keys}
        for other in mars_cfgs[1:]:
            cur = {k: other.get(k) for k in keys}
            if cur != ref:
                raise ValueError("mars enhancement profiles disagree on pinned preprocessing/model blocks")
    mars_profile = ENHANCE_PROFILES["mars"]
    if mars_profile.get("enable_realesrgan"):
        raise ValueError("Real-ESRGAN must remain disabled in mars ENHANCE_PROFILES")
    for p in profiles:
        if p.get("preprocessing_profile") == "raw_rgb_v1":
            # raw: only model input resize/normalize — no mars enhancement block
            if p.get("preprocessing_profile") != "raw_rgb_v1":
                raise ValueError("raw profile preprocessing mismatch")
        if p.get("preprocessing_profile") == "mars_enhancement_v1":
            if mars_profile.get("enable_realesrgan"):
                raise ValueError("Real-ESRGAN must stay off for mars profiles")


def _better(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Return True if profile a beats b under selection policy."""
    ap_a = a["metrics"].get("average_precision")
    ap_b = b["metrics"].get("average_precision")
    if ap_a is None and ap_b is None:
        pass
    elif ap_a is None:
        return False
    elif ap_b is None:
        return True
    elif ap_a > ap_b:
        return True
    elif ap_a < ap_b:
        return False

    auc_a = a["metrics"].get("auroc")
    auc_b = b["metrics"].get("auroc")
    if auc_a is not None and auc_b is not None:
        if auc_a > auc_b:
            return True
        if auc_a < auc_b:
            return False

    meta_a = PROFILE_META.get(a["config_id"], {})
    meta_b = PROFILE_META.get(b["config_id"], {})
    if meta_a.get("classifier_off") and not meta_b.get("classifier_off"):
        return True
    if meta_b.get("classifier_off") and not meta_a.get("classifier_off"):
        return False

    if meta_a.get("raw_rgb") and not meta_b.get("raw_rgb"):
        return True
    return False


def build_selection_artifact(
    entries: list[tuple[str, Path, Path]],
) -> dict[str, Any]:
    if len(entries) != 4:
        raise ValueError(f"exactly four validation profiles required, got {len(entries)}")
    ids = [cid for cid, _, _ in entries]
    if set(ids) != ALLOWED_CONFIG_IDS:
        raise ValueError(f"profile IDs must be exactly {sorted(ALLOWED_CONFIG_IDS)}, got {sorted(ids)}")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate config_id in selection input")

    profiles: list[dict[str, Any]] = []
    for config_id, pred_path, cfg_path in entries:
        if not pred_path.is_file():
            raise FileNotFoundError(f"missing predictions for {config_id}: {pred_path}")
        if not cfg_path.is_file():
            raise FileNotFoundError(f"missing config for {config_id}: {cfg_path}")
        cfg = _load_yaml(cfg_path)
        if str(cfg.get("config_id")) != config_id:
            raise ValueError(f"config_id mismatch: file={cfg.get('config_id')} expected={config_id}")
        rows = _read_predictions(pred_path)
        metrics = _profile_metrics(rows)
        clf_enabled = bool((cfg.get("depth_classifier") or {}).get("enabled", False))
        candidate = {
            "config_id": config_id,
            "predictions_csv": str(pred_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "prediction_csv_sha256": sha256_file(pred_path),
            "config_path": str(cfg_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "config_sha256": sha256_file(cfg_path),
            "preprocessing_profile": str(cfg.get("preprocessing_profile")),
            "classifier_enabled": clf_enabled,
            "precision_mode": str(cfg.get("precision", "fp32")),
            "checkpoint_hashes": _checkpoint_hashes(cfg),
            "metrics": metrics,
            "config_snapshot": {
                "preprocessing_profile": cfg.get("preprocessing_profile"),
                "autoencoder": cfg.get("autoencoder"),
                "depth": cfg.get("depth"),
                "operational_masks": cfg.get("operational_masks"),
                "priority_buffer": cfg.get("priority_buffer"),
            },
        }
        profiles.append(candidate)

    _assert_preprocessing_contracts(profiles)

    best = profiles[0]
    for candidate in profiles[1:]:
        if _better(candidate, best):
            best = candidate

    # Drop heavy snapshot from public candidates
    public_candidates = []
    for p in profiles:
        pub = {k: v for k, v in p.items() if k != "config_snapshot"}
        public_candidates.append(pub)

    body: dict[str, Any] = {
        "protocol_id": "independent_eval_v1",
        "selection_split": "validation",
        "test_opened": False,
        "not_final_test_result": True,
        "eligible_for_manuscript_primary_results": False,
        "evaluation_designation": "frozen-checkpoint current evaluation with unverified training provenance",
        "primary_precision": "fp32",
        "amp_parity_status": "rejected",
        "selection_rule": SELECTION_RULE,
        "selection_policy": {
            "primary_metric": "average_precision",
            "tie_breakers": ["auroc", "classifier_off", "raw_rgb_v1"],
            "threshold_metric": "f1",
            "threshold_tie_break": "highest_threshold",
        },
        "mars_enhancement_snapshot": dict(ENHANCE_PROFILES["mars"]),
        "selected_config_id": best["config_id"],
        "selected_predictions_csv": best["predictions_csv"],
        "selected_threshold": best["metrics"]["selected_threshold"],
        "selected_metrics": best["metrics"],
        "candidates": public_candidates,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    body["artifact_sha256"] = hashlib.sha256(canonical).hexdigest()
    return body


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Select frozen ARTPS validation profile")
    ap.add_argument(
        "--predictions",
        action="append",
        default=[],
        help="config_id=path/to/predictions.csv (repeatable; must be exactly the four allowed IDs)",
    )
    ap.add_argument(
        "--output",
        default="results/iac2026/independent_eval_v1/validation/profile_selection.json",
    )
    args = ap.parse_args(argv)

    default_cfg = {cid: cfg for cid, _, cfg in DEFAULT_PROFILES}
    default_pred = {cid: pred for cid, pred, _ in DEFAULT_PROFILES}

    entries: list[tuple[str, Path, Path]] = []
    if args.predictions:
        for item in args.predictions:
            if "=" not in item:
                print(f"ERROR: expected config_id=path, got {item!r}", file=sys.stderr)
                return 2
            cid, raw = item.split("=", 1)
            cid = cid.strip()
            if cid not in ALLOWED_CONFIG_IDS:
                print(f"ERROR: config_id not allowed on validation leaderboard: {cid}", file=sys.stderr)
                return 2
            if cid not in default_cfg:
                print(f"ERROR: unknown config_id {cid}", file=sys.stderr)
                return 2
            entries.append(
                (
                    cid,
                    (REPO_ROOT / raw).resolve(),
                    (REPO_ROOT / default_cfg[cid]).resolve(),
                )
            )
    else:
        for cid, pred, cfg in DEFAULT_PROFILES:
            entries.append((cid, (REPO_ROOT / pred).resolve(), (REPO_ROOT / cfg).resolve()))

    try:
        artifact = build_selection_artifact(entries)
    except Exception as exc:  # noqa: BLE001 — fail-loud selection gate
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    out_path = (REPO_ROOT / args.output).resolve()
    write_json(out_path, artifact)
    print(f"OK: selected {artifact['selected_config_id']} -> {out_path}")
    print(f"artifact_sha256={artifact['artifact_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
