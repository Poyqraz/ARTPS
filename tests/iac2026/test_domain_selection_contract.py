"""Domain selection document contract."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "paper" / "iac2026" / "reproduction" / "INDEPENDENT_EVAL_V1_DOMAIN_SELECTION.md"


def test_domain_selection_doc_exists_and_locks_primary():
    text = DOC.read_text(encoding="utf-8")
    assert "PRIMARY" in text
    assert "curiosity_mastcam_roboflow_v1" in text
    assert "Model scores" in text or "model scores" in text.lower()
    assert "annotator_count=1" in text
    assert "independent_double_review=false" in text
