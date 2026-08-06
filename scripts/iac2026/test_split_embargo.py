"""Test-split embargo helpers for independent_eval_v1."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATUS = (
    REPO_ROOT / "reproduction" / "iac2026" / "test_freeze" / "TEST_OPEN_STATUS.yaml"
)


def load_test_open_status(path: Path | str | None = None) -> dict[str, Any]:
    status_path = Path(path) if path else DEFAULT_STATUS
    if not status_path.is_absolute():
        status_path = (REPO_ROOT / status_path).resolve()
    with status_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"TEST_OPEN_STATUS root must be mapping: {status_path}")
    return data


def assert_split_allowed(split: str, status: Mapping[str, Any] | None = None) -> None:
    """Raise ValueError if split=test while test_opened is false."""
    st = dict(status or load_test_open_status())
    normalized = str(split or "").strip().lower()
    if normalized != "test":
        return
    if bool(st.get("test_opened", False)):
        return
    reason = st.get("reason") or "test split embargo active"
    raise ValueError(f"split=test refused: test_opened=false ({reason})")


def is_test_split_open(status: Mapping[str, Any] | None = None) -> bool:
    st = dict(status or load_test_open_status())
    return bool(st.get("test_opened", False))
