"""Fail-loud baseline stubs.

Historical C06: protocol UNKNOWN — refuse invented scores / fake PaDiM+PatchCore averages.
independent_eval_v1: contract keys are locked in INDEPENDENT_EVAL_V1.yaml / protocol doc,
but weights_sha256 + train_bank_recipe must still be real before scores are produced.
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
