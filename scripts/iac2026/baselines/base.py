"""Fail-loud baseline stubs until C06 protocol is recovered."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence


class BaselineContractError(RuntimeError):
    """Raised when a baseline cannot be run with an evidence-backed contract."""


def _require_keys(name: str, config: Mapping[str, Any], keys: Sequence[str]) -> None:
    missing = [k for k in keys if not config.get(k) or str(config.get(k)).upper().endswith("_TBD")]
    if missing:
        raise BaselineContractError(
            f"{name}: missing/unknown contract fields: {missing}. "
            "See paper/iac2026/reproduction/ARCHAEOLOGY_REPORT.md"
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
        "padim: code references expected local weight paths; presence/checksum/provenance "
        "unverified and C06 protocol is UNKNOWN. Refusing to invent scores. "
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
        "patchcore: C06 protocol UNKNOWN. Refusing to invent scores. anomalib is not used."
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
