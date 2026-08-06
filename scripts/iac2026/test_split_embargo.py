"""Test-split embargo helpers for independent_eval_v1."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATUS = (
    REPO_ROOT / "reproduction" / "iac2026" / "test_freeze" / "TEST_OPEN_STATUS.yaml"
)
DEFAULT_FINAL_SCOPE = (
    REPO_ROOT / "reproduction" / "iac2026" / "test_freeze" / "FINAL_TEST_SCOPE.yaml"
)

BLOCKED_STATUSES = frozenset(
    {
        "blocked_validation_sanity_review",
        "pending_final_test_authorization",
    }
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


def load_final_test_scope(path: Path | str | None = None) -> dict[str, Any]:
    scope_path = Path(path) if path else DEFAULT_FINAL_SCOPE
    if not scope_path.is_absolute():
        scope_path = (REPO_ROOT / scope_path).resolve()
    with scope_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"FINAL_TEST_SCOPE root must be mapping: {scope_path}")
    return data


def is_final_test_authorized(scope: Mapping[str, Any] | None = None) -> bool:
    sc = dict(scope or load_final_test_scope())
    auth = sc.get("authorization_status") or {}
    if isinstance(auth, Mapping) and "final_test_authorized" in auth:
        return bool(auth.get("final_test_authorized"))
    return bool(sc.get("final_test_authorized", False))


def assert_final_test_authorized(scope: Mapping[str, Any] | None = None) -> None:
    """Refuse final-test open/run while scope is blocked or unauthorized."""
    sc = dict(scope or load_final_test_scope())
    status = str(sc.get("status") or "")
    if status == "blocked_validation_sanity_review" or not is_final_test_authorized(sc):
        reason = (sc.get("authorization_status") or {}).get("reason") if isinstance(
            sc.get("authorization_status"), Mapping
        ) else None
        reason = reason or sc.get("reason") or status or "final_test_not_authorized"
        raise ValueError(
            f"final test refused: status={status!r} final_test_authorized=false ({reason})"
        )


def assert_split_allowed(
    split: str,
    status: Mapping[str, Any] | None = None,
    scope: Mapping[str, Any] | None = None,
) -> None:
    """Raise ValueError if split=test while test_opened is false or FINAL_TEST_SCOPE blocked."""
    st = dict(status or load_test_open_status())
    normalized = str(split or "").strip().lower()
    if normalized != "test":
        return
    if bool(st.get("test_opened", False)):
        # Even if TEST_OPEN_STATUS flips, blocked FINAL_TEST_SCOPE still forbids test.
        assert_final_test_authorized(scope)
        return
    reason = st.get("reason") or "test split embargo active"
    raise ValueError(f"split=test refused: test_opened=false ({reason})")


def is_test_split_open(status: Mapping[str, Any] | None = None) -> bool:
    st = dict(status or load_test_open_status())
    return bool(st.get("test_opened", False))
