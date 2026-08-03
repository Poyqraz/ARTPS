"""Shared prediction-table interface for baseline adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Sequence


class BaselineContractError(RuntimeError):
    """Raised when a baseline cannot be run with an evidence-backed contract."""


class BaselineAdapter(ABC):
    """Produce rows compatible with prediction_table.schema.json."""

    name: str = "baseline"

    @abstractmethod
    def predict_rows(
        self,
        sample_ids: Sequence[str],
        *,
        split: str,
        config: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def require_keys(self, config: Mapping[str, Any], keys: Sequence[str]) -> None:
        missing = [k for k in keys if not config.get(k) or str(config.get(k)).upper().endswith("_TBD")]
        if missing:
            raise BaselineContractError(
                f"{self.name}: missing/unknown contract fields: {missing}. "
                "See paper/iac2026/reproduction/ARCHAEOLOGY_REPORT.md"
            )
