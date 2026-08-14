"""Manuscript integration guards for independent_eval_v1_1 (no abstract/test/C05 rewrite)."""
from __future__ import annotations

import hashlib
import json
import re
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


def test_iac2026_template_conformance_formatting():
    sty = (REPO / "paper/iac2026/iac2026.sty").read_text(encoding="utf-8")
    main = MAIN.read_text(encoding="utf-8")
    methods = (REPO / "paper/iac2026/sections/methods.tex").read_text(encoding="utf-8")
    results = RESULTS.read_text(encoding="utf-8")
    disc = DISCUSSION.read_text(encoding="utf-8")
    exp = (REPO / "paper/iac2026/sections/experiments.tex").read_text(encoding="utf-8")
    assert "newtxtext" in sty and "newtxmath" in sty
    assert "mathptmx" not in sty
    assert r"name=Fig." in sty
    assert r"\raggedright" in sty
    assert "justification=raggedright" in sty
    assert r"headrulewidth}{0pt}" in sty.replace(" ", "")
    assert r"footrulewidth}{0pt}" in sty.replace(" ", "")
    assert "0.98in" in sty and "0.95in" in sty
    assert "0.22in" in sty
    assert "2.25cm" not in sty and "3.35cm" not in sty
    assert r"\fontsize{8}{9.5}" in sty
    assert r"\fontsize{10}{12}" in sty
    assert "5.58in" in sty
    assert "5--9 Oct 2026" in sty
    assert r"\singlespacing" in sty
    assert "1.0in" in sty  # bottom (and possibly top-adjacent)
    title_fn = sty[sty.find(r"\newcommand{\IACmaketitle}") : sty.find(r"\setlength{\parskip}")]
    assert r"\normalsize\IACpapercode" in title_fn
    assert title_fn.find(r"\IACpapercode") < title_fn.find("#1")
    assert r"\normalsize\bfseries #1" in title_fn.replace("\n", "") or r"\normalsize\bfseries #1" in title_fn
    assert r"\large\bfseries #1" not in title_fn
    assert r"\Large" not in title_fn
    assert r"\small\itshape" not in title_fn
    assert r"\normalsize\itshape\noindent #3" in title_fn.replace("\n", " ")
    assert r"\normalsize\upshape\noindent\IACcorresponding" in title_fn.replace("\n", "")
    end_center = title_fn.find(r"\end{center}")
    assert end_center > 0
    assert title_fn.find(r"\normalsize\bfseries #2") < end_center
    assert title_fn.find(r"\noindent #3") > end_center
    assert title_fn.find(r"\IACcorresponding") > end_center
    assert "IAC-26,A3,IP,109,x109221" in main
    assert r"Poyraz Baydemir\textsuperscript{a,*}" in main
    assert r"Faculty of Technology" in main
    assert r"\textsuperscript{a}" in main
    assert r"\IACcorresponding" in main and "poyrazbaydemir@gmail.com" in main
    assert r"Abstract" in title_fn
    freeze = yaml.safe_load(FREEZE.read_text(encoding="utf-8"))
    assert freeze["test_opened"] is False
    assert freeze["final_test_authorized"] is False
    decl = DECLARATION.read_text(encoding="utf-8").lower()
    assert "language verification" in decl
    for token in ("0.894", "0.847", "0.823", "0.856", "28.1"):
        assert token in main

    for blob in (methods, results, disc, exp):
        assert r"Figure~\ref" not in blob
        assert r"Fig.~\ref" in blob

    assert r"\RequirePackage{needspace}" in sty
    decl_src = DECLARATION.read_text(encoding="utf-8")
    assert r"\Needspace{8\baselineskip}" in decl_src
    assert decl_src.find(r"\Needspace") < decl_src.find(r"\section*{Declaration")
    freeze_md = (REPO / "paper/iac2026/SUBMISSION_FREEZE.md").read_text(encoding="utf-8")
    assert "balanced_float_flow_complete: true" in freeze_md
    assert "heading_first_indent_complete: true" in freeze_md
    assert "publication_language_complete: true" in freeze_md
    assert "primary_evaluation_completeness_complete: true" in freeze_md
    assert r"\titlespacing*{\section}" not in sty
    assert r"\titlespacing*{\subsection}" not in sty
    assert r"\titlespacing*{\subsubsection}" not in sty
    assert r"\titlespacing{\section}" in sty
    assert r"\titlespacing{\subsection}" in sty
    assert r"\titlespacing{\subsubsection}" in sty
    assert "indentfirst" not in sty
    assert r"\setlength{\parindent}{12pt}" in sty
    # No aggressive post-figure* FloatBarrier chain (PR #49 regression).
    for label in ("fig:pipeline", "fig:qualitative", "fig:cue-decomp"):
        i = methods.find(rf"\label{{{label}}}")
        assert i > 0
        chunk = methods[i : i + 80]
        assert r"\end{figure*}" in chunk
        assert r"\FloatBarrier" not in methods[i : i + 80]
    i4 = results.find(r"\label{fig:close-far}")
    assert i4 > 0
    assert r"\FloatBarrier" not in results[i4 : i4 + 80]
    # Table 1 queued in Experimental Protocol after Dataset, not after Hardware.
    assert exp.find(r"\subsection{Dataset and annotation semantics}") < exp.find(
        r"\label{tab:eval-tracks}"
    )
    assert exp.find(r"\label{tab:eval-tracks}") < exp.find(r"\subsection{Baselines and metrics}")
    assert exp.find(r"\label{tab:eval-tracks}") < exp.find(r"\subsection{Hardware scope}")
    # Optional section-level barriers only (not post-float chains).
    assert results.lstrip().startswith(r"\FloatBarrier") or r"\section{Results}" in results
    disc_src = DISCUSSION.read_text(encoding="utf-8")
    assert disc_src.lstrip().startswith(r"\FloatBarrier") or r"\section{Discussion}" in disc_src


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
    assert r"\subsection{Human-Reviewed Validation}" in results
    v11_cap = results[results.find(r"\label{tab:indep-v11}") - 500 : results.find(r"\label{tab:indep-v11}")]
    assert "Human-reviewed ARTPS validation results" in v11_cap
    assert "independent_eval" not in v11_cap
    exp = (REPO / "paper/iac2026/sections/experiments.tex").read_text(encoding="utf-8")
    assert r"\subsection{Primary performance evaluation}" in exp
    assert r"\subsection{Human-reviewed validation protocol}" in exp
    assert "independent_eval" not in exp
    assert r"\subsection{ARTPS Performance}" in results
    assert "0.772" in results
    assert "0.956" in results
    assert "0.852" in results
    assert "principal ranking" in results.lower() or "principal ranking-oriented" in results.lower()
    flat = " ".join(results.lower().split())
    assert "all-positive" in flat
    assert "non-discriminative" in flat or (
        "principal ranking" in flat and "auroc" in flat
    )
    methods = (REPO / "paper/iac2026/sections/methods.tex").read_text(encoding="utf-8")
    assert "maximum score among candidates" in methods.lower()
    assert "max_valid_candidate_after_masks" not in methods
    assert r"max\_valid\_candidate\_after\_masks" not in methods
    assert "layer~b" in flat and "layer~c" in flat
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
    assert (
        "reserved test" in lim
        or "reserved test partition" in lim
        or "reserved 54-image test partition" in lim
        or "test partition remains" in lim
    )
    assert "heuristic" in disc and "180/180" in disc.replace(" ", "")
    assert "0.772" in disc
    disc_flat = " ".join(disc.split())
    assert "ranking evidence" in disc_flat
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
    assert "summary of the artps evaluation design" in exp
    assert (
        "54-image test partition was reserved" in exp
        or "54-image test partition remains reserved" in exp
        or "test partition remains reserved" in exp
    )
    assert "primary artps evaluation" in exp and "human-reviewed validation" in exp
    assert "human-reviewed validation" in res
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
    assert "quantitative evaluation is reported" in methods.lower()
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
    assert "image-level scoring path" in cap
    assert "entropy" in cap
    assert "layer~b terminates" in cap or "layer~c consumes" in cap
    fig_l = fig.lower()
    assert "fixed-weight" in fig_l or "fixed-coefficient" in fig_l or "fixed-coeff" in fig_l
    assert "entropy-weighted" in fig_l
    assert "image-level scoring path" in fig_l
    assert "operational ranking" in fig_l
    assert "not in frozen" not in fig_l
    assert "max_valid_candidate" not in fig_l
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
        "10.1109/CVPR.2019.00982",
        "10.1109/TPAMI.2020.3019967",
        "10.1002/2016EA000252",
        "10.1007/s11214-020-00755-x",
        "10.1126/scirobotics.adi3099",
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
    assert "quantitative performance is reported separately" in cap_l
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
    assert "science-interest" in methods_flat
    assert "scientifically irrelevant" not in methods_flat
    assert "independently min-max" in cap_l or "independent min-max" in cap_l
    assert "pre-suppression" in cap_l or "pre-suppression fused" in methods_flat
    fig2_cap = methods[methods.find(r"\label{fig:qualitative}") - 1200 : methods.find(r"\label{fig:qualitative}")]
    assert "post-suppression" in fig2_cap.lower()
    assert "candidate-support overlay" in fig2_cap.lower()
    assert "corner brackets" in fig2_cap.lower()
    assert "proposal-support geometry" in fig2_cap.lower() or "image-region level" in fig2_cap.lower()


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
    assert "54-image test partition" in exp and "reserved" in exp
    assert "curiosity" in exp and "priority buffer" in exp
    res = RESULTS.read_text(encoding="utf-8").lower()
    assert "layer~b" in res and "layer~c" in res
    assert "before layer~c operational ranking" in res or "layer~b image-level score" in res


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
    assert "illustrative" in cap_l or "qualitative" in results.lower() or "non-test" in cap_l
    assert "preselected non-test" in results.lower() or "non-test examples" in cap_l
    assert (
        "quantitative performance is reported" in cap_l
        or "apparent-scale context" in cap_l
    )
    assert "post-suppression" in cap_l
    assert "candidate-support overlay" in cap_l
    assert "corner brackets" in cap_l
    assert "proposal-support geometry" in cap_l or "image-region level" in cap_l
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
    assert fig2.get("overlay_visualization_version") == "candidate_support_v3"
    assert fig4.get("overlay_visualization_version") == "candidate_support_v3"
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
    assert "translucent footprint" in methods
    assert "translucent footprint" in results
    assert "support footprint" in disc or "translucent support footprint" in disc
    assert "deterministically selected" not in methods


def _visible_tex(path: Path) -> str:
    return "\n".join(
        ln for ln in path.read_text(encoding="utf-8").splitlines() if not ln.lstrip().startswith("%")
    )


def test_primary_evaluation_completeness_and_fusion_scope():
    intro = REPO / "paper/iac2026/sections/introduction.tex"
    methods = REPO / "paper/iac2026/sections/methods.tex"
    exp_p = REPO / "paper/iac2026/sections/experiments.tex"
    results_p = RESULTS
    disc_p = DISCUSSION
    lim_p = LIMITATIONS
    conc = REPO / "paper/iac2026/sections/conclusion.tex"
    body_paths = (intro, methods, exp_p, results_p, disc_p, lim_p, conc)
    body = "\n".join(_visible_tex(p) for p in body_paths)
    body_l = body.lower()
    exp = _visible_tex(exp_p)
    exp_l = exp.lower()
    methods_l = _visible_tex(methods).lower()
    results = _visible_tex(results_p)
    freeze_md = (REPO / "paper/iac2026/SUBMISSION_FREEZE.md").read_text(encoding="utf-8")
    audit = (REPO / "paper/iac2026/PRIMARY_EVALUATION_DEFINITION_AUDIT.md").read_text(
        encoding="utf-8"
    )

    assert "entropy-weighted" in methods_l
    assert "specified adaptive fusion mode" in methods_l
    assert "fixed fusion coefficients" in methods_l
    assert "the human-reviewed validation uses the fixed-coefficient" in body_l
    for banned in (
        "the reported evaluation uses",
        "reported evaluation applies",
        "reported evaluation uses the fixed",
        "coefficients used to generate the reported image-level",
    ):
        assert banned not in body_l

    flat = " ".join(body.split())
    for sent in re.split(r"(?<=[A-Za-z\}])\.\s+", flat):
        sl = sent.lower()
        if "0.894" in sl:
            assert "fixed-coefficient" not in sl
            assert "entropy-weighted" not in sl
        if "708" in sent:
            assert "comprising" not in sl

    assert r"2{,}847" in exp or "2,847" in exp
    assert r"1{,}247" in exp or "1,247" in exp
    assert "892" in exp
    assert "708" in exp
    assert "sol 100" in exp_l and "sol 1" in exp_l
    assert "diverse field conditions" in exp_l
    assert "wrn-50-2" in exp_l
    assert "selected on validation" in exp_l
    assert "measured on the test set" in exp_l
    assert "reference labels" in exp_l
    assert "primary anomaly-discrimination" in exp_l
    assert "artps evaluation ground truth" not in exp_l
    assert "summarizes detection and lightweight runtime" in exp_l
    assert "restates" not in exp_l
    assert "primary artps detection configuration" in exp_l
    assert "layer~c operational ranking" in exp_l or "layer~b detection metrics" in exp_l

    fig2 = _visible_tex(methods)
    fig2_cap = fig2[fig2.find(r"\label{fig:qualitative}") - 1200 : fig2.find(r"\label{fig:qualitative}")]
    fig2_flat = " ".join(fig2_cap.split())
    assert "from a non-test Mars scene selected before inference" in fig2_flat
    fig4_cap = results[results.find(r"\label{fig:close-far}") - 1200 : results.find(r"\label{fig:close-far}")]
    fig4_flat = " ".join(fig4_cap.split())
    assert "apparent-scale context" in fig4_flat.lower()
    assert "Reviewed / unchanged / relabeled" in results
    assert "None of these figures" not in results

    disc = _visible_tex(disc_p)
    assert "operational target-prioritization architecture" in disc
    lim = _visible_tex(lim_p)
    assert "jetson" in lim.lower() and (
        "future hardware-validation" in lim.lower() or "hardware-validation stage" in lim.lower()
    )

    main = MAIN.read_text(encoding="utf-8")
    assert "0.772" not in main
    for token in ("0.894", "0.847", "0.823", "0.856", "28.1"):
        assert token in main
        assert token in results
    assert "0.772" in results
    assert "UNRESOLVED (C)" in audit
    assert "primary_evaluation_completeness_complete: true" in freeze_md
    assert "fusion-mode C" in freeze_md.lower() or "UNRESOLVED (C)" in freeze_md
    assert "there was no test inference" not in body_l
    assert "no test metric" not in body_l
    assert "combined baseline" not in body_l
    assert "the evaluation uses a fixed configuration" not in body_l
    assert "within this corpus" in exp_l
    assert "test/validation" in exp_l
    assert "padim/patchcore baseline entry" in exp_l
    repro = exp_l[exp_l.find("reproducibility") :]
    assert "for the human-reviewed validation" in repro
    assert "checkpoint hashes" in repro
    assert "image-score definition" in repro or "image-score" in repro
    assert "measured on the test set" in exp_l
    lim_flat = " ".join(lim.lower().split())
    assert "reserved 54-image test partition" in lim_flat or "54-image test partition remains" in lim_flat
    # reserved-test protocol stated in §4.2 (not repeated in Reproducibility)
    assert "54-image test partition" in exp_l and "reserved" in exp_l
