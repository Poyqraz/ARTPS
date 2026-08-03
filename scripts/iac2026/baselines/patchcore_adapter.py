"""PatchCore baseline adapter — fail-loud until memory-bank contract is known."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from .base import BaselineAdapter, BaselineContractError


class PatchCoreAdapter(BaselineAdapter):
    name = "patchcore"

    def predict_rows(
        self,
        sample_ids: Sequence[str],
        *,
        split: str,
        config: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        self.require_keys(
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
            "patchcore: weights may exist locally (results/patchcore_bank.pth) but the "
            "accepted-abstract C06 protocol is UNKNOWN. Refusing to invent scores. "
            "Do not average with PaDiM into a fake 0.856. anomalib is not used."
        )


def build_adapter() -> PatchCoreAdapter:
    return PatchCoreAdapter()
