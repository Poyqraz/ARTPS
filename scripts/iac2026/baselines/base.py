"""Fail-loud baseline stubs.

Historical C06: protocol UNKNOWN — refuse invented scores / fake PaDiM+PatchCore averages.
independent_eval_v1: contract keys are locked in INDEPENDENT_EVAL_V1.yaml / protocol doc,
but weights_sha256 + train_bank_recipe must still be real before scores are produced.
Train bank must contain only binary_label=0 samples when train_bank_sample_ids is supplied.
No anomalib. Never invent a combined 0.856 cell.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence


class BaselineContractError(RuntimeError):
    """Raised when a baseline cannot be run with an evidence-backed contract."""


def _require_keys(name: str, config: Mapping[str, Any], keys: Sequence[str]) -> None:
    missing = [k for k in keys if not config.get(k) or str(config.get(k)).upper().endswith("_TBD")]
    if missing:
        raise BaselineContractError(
            f"{name}: missing/unknown contract fields: {missing}. "
            "Historical C06: see paper/iac2026/reproduction/ARCHAEOLOGY_REPORT.md. "
            "Independent eval: see paper/iac2026/reproduction/INDEPENDENT_EVALUATION_PROTOCOL.md."
        )


def _forbid_positive_train_bank(name: str, config: Mapping[str, Any]) -> None:
    ids = config.get("train_bank_sample_ids")
    labels = config.get("train_bank_binary_labels")
    if not ids:
        return
    if labels is None:
        raise BaselineContractError(
            f"{name}: train_bank_sample_ids set but train_bank_binary_labels missing"
        )
    if len(list(ids)) != len(list(labels)):
        raise BaselineContractError(f"{name}: train_bank_sample_ids/labels length mismatch")
    if any(int(x) != 0 for x in labels):
        raise BaselineContractError(
            f"{name}: train bank must contain only binary_label=0 (normal) samples"
        )


def predict_padim(
    sample_ids: Sequence[str], *, split: str, config: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    _require_keys(
        "padim",
        config,
        [
            "backbone",
            "layers",
            "image_size",
            "weights_path",
            "weights_sha256",
            "score_aggregation",
            "train_bank_recipe",
        ],
    )
    _forbid_positive_train_bank("padim", config)
    raise BaselineContractError(
        "padim: refusing to invent scores. Historical C06 provenance unverified; "
        "independent_eval_v1 still requires real weights_sha256 + train_bank_recipe. "
        "Do not average with PatchCore into a fake 0.856."
    )


def predict_patchcore(
    sample_ids: Sequence[str], *, split: str, config: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    _require_keys(
        "patchcore",
        config,
        [
            "backbone",
            "layers",
            "image_size",
            "weights_path",
            "weights_sha256",
            "score_aggregation",
            "coreset_ratio",
            "train_bank_recipe",
        ],
    )
    _forbid_positive_train_bank("patchcore", config)
    raise BaselineContractError(
        "patchcore: refusing to invent scores. Historical C06 UNKNOWN; "
        "independent_eval_v1 contract keys alone are insufficient without pinned weights. "
        "anomalib is not used."
    )


# Back-compat thin wrappers for tests
class PaDiMAdapter:
    name = "padim"

    def predict_rows(self, sample_ids, *, split, config):
        return predict_padim(sample_ids, split=split, config=config)


class PatchCoreAdapter:
    name = "patchcore"

    def predict_rows(self, sample_ids, *, split, config):
        return predict_patchcore(sample_ids, split=split, config=config)
