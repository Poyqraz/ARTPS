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
    methods = (REPO / "paper/iac2026/sections/methods.tex").read_text(encoding="utf-8")
    assert "max_valid_candidate_after_masks" in methods or "max\\_valid\\_candidate\\_after\\_masks" in methods
    assert "curiosity" in flat and "not applied" in flat
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
    assert "curiosity" in lim and "priority buffer" in lim
    assert "no final test" in lim or "test was opened" in lim
    assert "heuristic" in disc and "180/180" in disc.replace(" ", "")
    assert "0.772" in disc
    disc_flat = " ".join(disc.split())
    assert "historical" in disc_flat and "same issue" in disc_flat
    assert "planetary imagery as a whole" in disc_flat
    assert "naturally high" not in disc_flat
    methods = (REPO / "paper/iac2026/sections/methods.tex").read_text(encoding="utf-8").lower()
    assert "entropy" in methods and "fixed-weight" in methods
    assert "c(r)" in methods.replace(" ", "") or "c(r)" in methods
    intro = (REPO / "paper/iac2026/sections/introduction.tex").read_text(encoding="utf-8").lower()
    assert "fixed-weight" in intro or "fixed-weight multi-cue" in intro
    exp = (REPO / "paper/iac2026/sections/experiments.tex").read_text(encoding="utf-8").lower()
    res = RESULTS.read_text(encoding="utf-8").lower()
    assert "held-out test" in exp and "not opened" in exp
    assert "historical" in exp and "supplementary" in exp
    assert "validation-only" in res or "validation only" in res
    assert "held-out test" in res and "unopened" in res
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


def test_camera_ready_figure_and_bibliography():
    methods = (REPO / "paper/iac2026/sections/methods.tex").read_text(encoding="utf-8")
    fig = (REPO / "paper/iac2026/figures/artps_pipeline.tex").read_text(encoding="utf-8")
    bib = (REPO / "paper/iac2026/references.bib").read_text(encoding="utf-8")
    sty = (REPO / "paper/iac2026/iac2026.sty").read_text(encoding="utf-8")
    assert "fig:pipeline" in methods
    assert r"\input{figures/artps_pipeline}" in methods
    assert r"\begin{tikzpicture}" in fig
    assert r"\RequirePackage{tikz}" in sty or r"\usepackage{tikz}" in sty
    cap = methods.lower()
    assert "frozen" in cap
    assert "entropy" in cap
    assert "not included" in cap
    fig_l = fig.lower()
    assert "fixed-weight" in fig_l
    assert "entropy-weighted" in fig_l
    assert "not in frozen" in fig_l
    assert "0.772" not in fig
    assert "0.894" not in fig
    for doi in (
        "10.1145/2168752.2168764",
        "10.1007/978-3-030-68799-1_35",
        "10.1109/CVPR52688.2022.01392",
        "10.1109/ICCV48922.2021.01196",
    ):
        assert doi in bib
    assert "Tara and others" not in bib
    assert "Increased Mars Rover Autonomy" not in bib
    main = MAIN.read_text(encoding="utf-8")
    assert "poyrazbaydemir@gmail.com" in main
    assert "CORRESPONDING_EMAIL_TBD" not in main
    assert r"CORRESPONDING\_EMAIL\_TBD" not in main
