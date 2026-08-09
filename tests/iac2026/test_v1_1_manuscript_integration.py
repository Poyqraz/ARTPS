"""Manuscript integration guards for independent_eval_v1_1 (no abstract/test/C05 rewrite)."""
from __future__ import annotations

import hashlib
import json
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
    assert "entropy" in methods
    assert "fixed-weight" in methods or "fixed-coefficient" in methods or "fixed fusion coefficients" in methods
    assert "c(r)" in methods.replace(" ", "") or "c(r)" in methods
    intro = (REPO / "paper/iac2026/sections/introduction.tex").read_text(encoding="utf-8").lower()
    assert (
        "fixed-weight" in intro
        or "fixed-weight multi-cue" in intro
        or "fixed-coefficient" in intro
    )
    exp = (REPO / "paper/iac2026/sections/experiments.tex").read_text(encoding="utf-8").lower()
    res = RESULTS.read_text(encoding="utf-8").lower()
    assert "tab:eval-tracks" in exp
    assert "not directly interchangeable" in exp
    assert "held-out test" in exp and ("not opened" in exp or "unopened" in exp)
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
    assert "fig:qualitative" in methods
    assert "figures/fig_qualitative_artps.png" in methods
    assert "qualitative" in methods.lower()
    assert "non-metric" in methods.lower() or "relative and non-metric" in methods.lower()
    assert "not used as an additional quantitative benchmark" in methods.lower()
    assert (REPO / "paper/iac2026/figures/fig_qualitative_artps.png").is_file()
    meta = json.loads(
        (REPO / "paper/iac2026/figures/fig_qualitative_artps.meta.json").read_text(encoding="utf-8")
    )
    assert meta["test_used"] is False
    assert meta["score_blind_selection"] is True
    assert meta.get("model_output_used_for_selection") is False
    assert meta.get("score_based_cherry_picking") is False
    assert str(meta["split"]).lower() != "test"
    rel = str(meta["relative_path"]).replace("\\", "/")
    assert "test/" not in rel.lower()
    assert rel == (
        "train/boulder/curiosity_300_MAST_453_jpg.rf.6ecd29659d982741653bbe91b11ef22b.jpg"
    )
    assert "deterministically selected" not in methods.lower()
    assert r"\begin{tikzpicture}" in fig
    assert r"\RequirePackage{tikz}" in sty or r"\usepackage{tikz}" in sty
    cap = methods.lower()
    assert "frozen" in cap
    assert "entropy" in cap
    assert "not included" in cap
    fig_l = fig.lower()
    assert "fixed-weight" in fig_l or "fixed-coefficient" in fig_l or "fixed-coeff" in fig_l
    assert "entropy-weighted" in fig_l
    assert "not in frozen" in fig_l
    assert "0.772" not in fig
    assert "0.894" not in fig
    for doi in (
        "10.1145/2168752.2168764",
        "10.1007/978-3-030-68799-1_35",
        "10.1109/CVPR52688.2022.01392",
        "10.1109/ICCV48922.2021.01196",
        "10.1126/scirobotics.aan4582",
        "10.1016/j.pss.2019.03.007",
        "10.1002/rob.21979",
    ):
        assert doi in bib
    assert "Tara and others" not in bib
    assert "Increased Mars Rover Autonomy" not in bib
    main = MAIN.read_text(encoding="utf-8")
    assert "poyrazbaydemir@gmail.com" in main
    assert "CORRESPONDING_EMAIL_TBD" not in main
    assert r"CORRESPONDING\_EMAIL\_TBD" not in main


def test_cue_decomposition_figure_and_caption():
    methods = (REPO / "paper/iac2026/sections/methods.tex").read_text(encoding="utf-8")
    results = RESULTS.read_text(encoding="utf-8")
    disc = DISCUSSION.read_text(encoding="utf-8")
    png = REPO / "paper/iac2026/figures/fig_cue_decomposition_artps.png"
    meta_path = REPO / "paper/iac2026/figures/fig_cue_decomposition_artps.meta.json"
    fig2_meta = json.loads(
        (REPO / "paper/iac2026/figures/fig_qualitative_artps.meta.json").read_text(encoding="utf-8")
    )
    cue_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert "fig:cue-decomp" in methods
    assert "figures/fig_cue_decomposition_artps.png" in methods
    assert png.is_file()
    cap = methods[methods.find(r"\label{fig:cue-decomp}") - 800 : methods.find(r"\label{fig:cue-decomp}")]
    cap_l = cap.lower()
    assert "qualitative" in methods.lower() or "illustrative" in cap_l
    assert "non-metric" in cap_l
    assert "not used as an additional quantitative benchmark" in cap_l
    assert cue_meta["sample_id"] == fig2_meta["sample_id"]
    assert cue_meta["file_sha256"] == fig2_meta["file_sha256"]
    assert cue_meta["test_used"] is False
    assert cue_meta.get("new_sample_selection") is False
    assert "0.920" not in results
    assert "0/8/0/46" not in results
    assert "$0.0$" not in results
    assert "fig:cue-decomp" in results
    assert "fig:cue-decomp" in disc
    methods_flat = " ".join(methods.lower().split())
    assert "scientifically relevant" in methods_flat
    assert "scientifically irrelevant" not in methods_flat
    assert "independently min-max" in cap_l or "not numerically comparable" in cap_l
    assert "pre-suppression" in cap_l or "pre-suppression fused" in methods_flat
    fig2_cap = methods[methods.find(r"\label{fig:qualitative}") - 1200 : methods.find(r"\label{fig:qualitative}")]
    assert "post-suppression" in fig2_cap.lower()
    assert "candidate-support overlay" in fig2_cap.lower()
    assert "corner brackets" in fig2_cap.lower()
    assert "not pixel-level" in fig2_cap.lower()


def test_scientific_definition_pass_language():
    tex_paths = (
        REPO / "paper/iac2026/sections/methods.tex",
        REPO / "paper/iac2026/sections/experiments.tex",
        REPO / "paper/iac2026/sections/results.tex",
        REPO / "paper/iac2026/sections/discussion.tex",
        REPO / "paper/iac2026/sections/limitations.tex",
        REPO / "paper/iac2026/sections/conclusion.tex",
        REPO / "paper/iac2026/sections/introduction.tex",
        REPO / "paper/iac2026/sections/related_work.tex",
    )
    blob = "\n".join(p.read_text(encoding="utf-8") for p in tex_paths)
    visible = "\n".join(
        ln for ln in blob.splitlines() if not ln.lstrip().startswith("%")
    )
    vis_l = visible.lower()
    for banned in (
        "this draft",
        "planned baselines",
        "planning level",
        "may be discussed qualitatively",
        "must not migrate",
        "deterministically selected",
        "c05",
        "c06",
        "c07",
        "test_opened",
        "accepted_abstract_reproduction_pending",
        "protocol_defined_pending_data",
    ):
        assert banned not in vis_l
    intro = (REPO / "paper/iac2026/sections/introduction.tex").read_text(encoding="utf-8")
    intro_vis = "\n".join(ln for ln in intro.splitlines() if not ln.lstrip().startswith("%"))
    assert "low-confidence" not in intro_vis.lower()
    methods = (REPO / "paper/iac2026/sections/methods.tex").read_text(encoding="utf-8")
    methods_vis = "\n".join(ln for ln in methods.splitlines() if not ln.lstrip().startswith("%"))
    assert "diversity or suppression" not in methods_vis.lower()
    methods_flat = " ".join(methods.lower().split())
    assert "fixed fusion coefficients" in methods_flat
    assert r"s_{\mathrm{image}}" in methods.lower().replace(" ", "")
    assert "v_r" in methods.lower().replace(" ", "") or "v_{r}" in methods.lower()
    exp = (REPO / "paper/iac2026/sections/experiments.tex").read_text(encoding="utf-8").lower()
    assert "held-out test" in exp and ("unopened" in exp or "not opened" in exp)
    assert "curiosity" in exp and "priority buffer" in exp
    res = RESULTS.read_text(encoding="utf-8").lower()
    assert "curiosity ranking" in res and "not applied" in res


ALLOWED_CLOSE_FAR_POOL = {
    "train/hills_or_ridge/curiosity_1100_MAST_938_jpg.rf.7417a3036ec4af81b3b9d4305c05eee3.jpg",
    "train/boulder/percy_sol1450_MCZ_RIGHT_9_jpg.rf.f390f8c84becbe615a34db73d9f2610e.jpg",
    "train/flat_terrain/curiosity_1100_MAST_827_jpg.rf.fd10bd35d413cba7432b79ab8433e9b6.jpg",
    "train/rocky/curiosity_1100_MAST_817_jpg.rf.7d755ad9d3fcbac273a3dfffdc0b3c40.jpg",
    "train/rover/percy_sol150_NAVCAM_LEFT_8_jpg.rf.5d964d0db273d6db4a7054ec8516c688.jpg",
}


def test_close_far_qualitative_figure():
    results = RESULTS.read_text(encoding="utf-8")
    png = REPO / "paper/iac2026/figures/fig_close_far_qualitative_artps.png"
    meta_path = REPO / "paper/iac2026/figures/fig_close_far_qualitative_artps.meta.json"
    assert png.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert "fig:close-far" in results
    assert "figures/fig_close_far_qualitative_artps.png" in results
    cap = results[results.find(r"\label{fig:close-far}") - 1200 : results.find(r"\label{fig:close-far}")]
    cap_l = " ".join(cap.lower().split())
    assert "illustrative" in cap_l or "qualitative" in results.lower()
    assert "author-provided" in cap_l or "author-provided" in results.lower()
    assert "not presented as an additional quantitative benchmark" in cap_l
    assert "post-suppression" in cap_l
    assert "candidate-support overlay" in cap_l
    assert "corner brackets" in cap_l
    for banned in (
        "demonstrates superior",
        "proves robustness",
        "validates distance generalization",
        "superior near/far performance",
    ):
        assert banned not in results.lower()
        assert banned not in cap_l
    assert meta["test_used"] is False
    assert meta["author_provided_pool_only"] is True
    assert meta["agent_selected_outside_pool"] is False
    assert meta["quantitative_experiment"] is False
    assert meta.get("score_maximization_cherrypick") is False
    close_rel = str(meta["close"]["relative_path"]).replace("\\", "/")
    far_rel = str(meta["far"]["relative_path"]).replace("\\", "/")
    assert close_rel in ALLOWED_CLOSE_FAR_POOL
    assert far_rel in ALLOWED_CLOSE_FAR_POOL
    fig2_rel = json.loads(
        (REPO / "paper/iac2026/figures/fig_qualitative_artps.meta.json").read_text(encoding="utf-8")
    )["relative_path"].replace("\\", "/")
    assert close_rel != fig2_rel and far_rel != fig2_rel
    assert str(meta["close"]["split"]).lower() != "test"
    assert str(meta["far"]["split"]).lower() != "test"
    assert "test/" not in close_rel.lower()
    assert "test/" not in far_rel.lower()
    exp = (REPO / "paper/iac2026/sections/experiments.tex").read_text(encoding="utf-8").lower()
    assert "fig:close-far" in exp or "close-range" in exp
    lim = LIMITATIONS.read_text(encoding="utf-8").lower()
    assert "near/far" in lim or "qualitative figures" in lim
    assert meta.get("visualization_only") is True
    assert meta.get("candidate_scores_changed") is False
    assert meta.get("validity_decisions_changed") is False
    assert meta.get("image_scores_changed") is False


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_same_candidates(actual: list[dict], expected: list[dict]) -> None:
    assert len(actual) == len(expected)
    for got, exp in zip(actual, expected):
        assert int(got["x"]) == int(exp["x"])
        assert int(got["y"]) == int(exp["y"])
        assert int(got["w"]) == int(exp["w"])
        assert int(got["h"]) == int(exp["h"])
        assert abs(float(got["score"]) - float(exp["score"])) < 1e-9


def test_candidate_support_overlay_invariance_and_language():
    fixture = json.loads(
        (REPO / "tests/iac2026/fixtures/qualitative_overlay_candidates.json").read_text(
            encoding="utf-8"
        )
    )
    fig2 = json.loads(
        (REPO / "paper/iac2026/figures/fig_qualitative_artps.meta.json").read_text(encoding="utf-8")
    )
    fig4 = json.loads(
        (REPO / "paper/iac2026/figures/fig_close_far_qualitative_artps.meta.json").read_text(
            encoding="utf-8"
        )
    )
    cue_png = REPO / "paper/iac2026/figures/fig_cue_decomposition_artps.png"
    assert _sha256_file(cue_png) == fixture["fig3_png_sha256"]
    assert fig2["relative_path"].replace("\\", "/") == fixture["fig2"]["relative_path"]
    assert fig2["n_raw_detections"] == fixture["fig2"]["n_raw_detections"]
    assert fig2["n_valid_candidates"] == fixture["fig2"]["n_valid_candidates"]
    _assert_same_candidates(fig2["candidates"], fixture["fig2"]["candidates"])
    assert abs(max(c["score"] for c in fig2["candidates"]) - fixture["fig2"]["image_score"]) < 1e-9
    assert fig4["close"]["relative_path"].replace("\\", "/") == fixture["fig4_close"]["relative_path"]
    assert fig4["far"]["relative_path"].replace("\\", "/") == fixture["fig4_far"]["relative_path"]
    assert fig4["close"]["n_raw_detections"] == fixture["fig4_close"]["n_raw_detections"]
    assert fig4["far"]["n_raw_detections"] == fixture["fig4_far"]["n_raw_detections"]
    _assert_same_candidates(fig4["close"]["candidates"], fixture["fig4_close"]["candidates"])
    _assert_same_candidates(fig4["far"]["candidates"], fixture["fig4_far"]["candidates"])
    assert fig2.get("visualization_only") is True
    assert fig2.get("overlay_visualization_version") == "candidate_support_v1"
    methods = (REPO / "paper/iac2026/sections/methods.tex").read_text(encoding="utf-8").lower()
    results = RESULTS.read_text(encoding="utf-8").lower()
    disc = DISCUSSION.read_text(encoding="utf-8").lower()
    blob = methods + "\n" + results + "\n" + disc
    for banned in (
        "segmentation mask",
        "object segmentation",
        "ground-truth boundary",
        "object silhouette",
    ):
        assert banned not in blob
    assert "deterministically selected" not in methods
