"""Manuscript integration guards for independent_eval_v1_1 (no abstract/test/C05 rewrite)."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
FREEZE = REPO / "reproduction/iac2026/test_freeze/FINAL_TEST_SCOPE.yaml"
RESULTS = REPO / "paper/iac2026/sections/results.tex"
MAIN = REPO / "paper/iac2026/main.tex"
DECLARATION = REPO / "paper/iac2026/sections/declaration.tex"
LEDGER = REPO / "paper/iac2026/CLAIM_EVIDENCE_LEDGER.md"
DISCUSSION = REPO / "paper/iac2026/sections/discussion.tex"
LIMITATIONS = REPO / "paper/iac2026/sections/limitations.tex"


def test_historical_abstract_and_table1_unchanged():
    main = MAIN.read_text(encoding="utf-8")
    results = RESULTS.read_text(encoding="utf-8")
    for token in ("0.894", "0.847", "0.823", "0.856", "28.1"):
        assert token in main
        assert token in results
    assert "tab:accepted-abstract" in results
    assert "0.772" not in main  # abstract must not replace 0.894 with v1_1 AUROC


def test_v1_1_supplementary_table_and_auroc_headline():
    results = RESULTS.read_text(encoding="utf-8")
    assert "tab:indep-v11" in results
    assert "Supplementary Human-Reviewed Evaluation" in results
    assert "0.772" in results
    assert "0.956" in results
    assert "0.852" in results
    assert "principal ranking" in results.lower() or "principal ranking-oriented" in results.lower()
    flat = " ".join(results.lower().split())
    assert "all-positive" in flat
    assert "not treated as" in flat and "performance measures" in flat
    assert "0.920" not in results
    assert "0/8/0/46" not in results
    assert "threshold $0.0$" not in results
    assert "$0.0$" not in results


def test_v1_1_audit_artifacts_keep_degenerate_f1():
    report = (REPO / "paper/iac2026/reproduction/INDEPENDENT_EVAL_V1_1_REPORT.md").read_text(
        encoding="utf-8"
    )
    evidence = (REPO / "paper/iac2026/reproduction/INDEPENDENT_EVAL_V1_1_EVIDENCE_MAP.md").read_text(
        encoding="utf-8"
    )
    blob = report + "\n" + evidence
    assert "0.920" in blob
    assert "0.0" in blob
    assert "0/8/0/46" in blob


def test_declaration_and_funding_absent():
    decl = DECLARATION.read_text(encoding="utf-8")
    assert "language verification" in decl.lower()
    blob = "\n".join(
        p.read_text(encoding="utf-8").lower()
        for p in (
            RESULTS,
            MAIN,
            DECLARATION,
            REPO / "paper/iac2026/sections/experiments.tex",
            REPO / "paper/iac2026/sections/discussion.tex",
            REPO / "paper/iac2026/sections/limitations.tex",
            REPO / "paper/iac2026/sections/conclusion.tex",
        )
    )
    for banned in ("grant number", "sponsored by", "funding agency", "this work was funded"):
        assert banned not in blob


def test_ledger_v1_pending_v11_measured_validation_only():
    text = LEDGER.read_text(encoding="utf-8")
    v1 = next(ln for ln in text.splitlines() if ln.startswith("| IND_EVAL_V1 |") and not ln.startswith("| IND_EVAL_V1_1 |"))
    v11 = next(ln for ln in text.splitlines() if ln.startswith("| IND_EVAL_V1_1 |"))
    assert "protocol_defined_pending_data" in v1
    assert "`measured`" in v11 or "| `measured` |" in v11 or " `measured` " in v11
    assert "validation" in v11.lower()
    assert "no test" in v11.lower() or "not a final-test" in v11.lower() or "no test result" in v11.lower()
    for cid in ("C05", "C06", "C07"):
        line = next(ln for ln in text.splitlines() if ln.startswith(f"| {cid} |"))
        assert "accepted_abstract_reproduction_pending" in line


def test_limitations_and_discussion_present():
    lim = LIMITATIONS.read_text(encoding="utf-8").lower()
    disc = DISCUSSION.read_text(encoding="utf-8").lower()
    assert "repeat-author" in lim or "not an independent second annotation" in lim
    assert "20" in lim and "8" in lim and "2" in lim
    assert "operating-threshold calibration unstable" in lim or "calibration unstable" in lim
    assert "no final test" in lim or "test was opened" in lim
    assert "heuristic" in disc and "180/180" in disc.replace(" ", "")
    assert "0.772" in disc
    disc_flat = " ".join(disc.split())
    assert "historical" in disc_flat and "same issue" in disc_flat
    assert "planetary imagery as a whole" in disc_flat
    assert "naturally high" not in disc_flat
    exp = (REPO / "paper/iac2026/sections/experiments.tex").read_text(encoding="utf-8").lower()
    res = RESULTS.read_text(encoding="utf-8").lower()
    assert "compatibility" in exp and "dataset and validation artifacts now exist" in exp
    assert "claim closure remain" in res or "claim closure remains" in res
    assert "data are still pending" not in res and "data still pending" not in res


def test_no_test_split_metrics_in_tex():
    tex = "\n".join(
        p.read_text(encoding="utf-8").lower()
        for p in (
            RESULTS,
            REPO / "paper/iac2026/sections/experiments.tex",
            REPO / "paper/iac2026/sections/conclusion.tex",
        )
    )
    assert "test auroc" not in tex
    assert "test f1" not in tex
    freeze = yaml.safe_load(FREEZE.read_text(encoding="utf-8"))
    assert freeze["test_opened"] is False
    assert freeze["final_test_authorized"] is False
