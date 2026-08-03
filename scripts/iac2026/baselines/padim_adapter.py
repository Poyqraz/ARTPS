"""PaDiM baseline adapter — fail-loud until extractor/config/weights contract is known."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from .base import BaselineAdapter, BaselineContractError


class PaDiMAdapter(BaselineAdapter):
    name = "padim"

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
                "train_bank_recipe",
            ],
        )
        # Intentionally not implemented: archaeology has not pinned the C06 protocol.
        raise BaselineContractError(
            "padim: weights may exist locally (results/padim_stats.pth) but the "
            "accepted-abstract C06 protocol (extractor version, bank recipe, "
            "image-score aggregation, split) is UNKNOWN. Refusing to invent scores. "
            "Do not average with PatchCore into a fake 0.856."
        )


def build_adapter() -> PaDiMAdapter:
    return PaDiMAdapter()
