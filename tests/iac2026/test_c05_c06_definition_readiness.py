"""C05/C06 layered readiness: defs + artifacts + input audit; no fake-ready path."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "iac2026"))

from check_c05_c06_definition_readiness import evaluate  # noqa: E402
from _common import sha256_file  # noqa: E402

MANIFEST_FIELDS = [
    "sample_id",
    "mission",
    "instrument",
    "sol",
    "source_id",
    "source_url",
    "relative_path",
    "sha256",
    "split",
    "binary_label",
    "label_semantics",
    "label_source",
    "annotation_version",
    "scene_group_id",
    "duplicate_group_id",
    "notes",
]

PRED_FIELDS = [
    "sample_id",
    "split",
    "y_true",
    "anomaly_score",
    "model_name",
    "model_version",
    "config_id",
]


def _hex(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _base_defs():
    return yaml.safe_load(
        (REPO / "reproduction/iac2026/C05_C06_DEFINITIONS.yaml").read_text(encoding="utf-8")
    )


def _closed_node(value):
    return {
        "value": value,
        "status": "LOCATED",
        "required_for_real_run": True,
        "sources": [{"file": "tests/synthetic"}],
    }


def _close_p0(defs: dict) -> None:
    defs["task_level"] = _closed_node("image_binary")
    defs["positive_class"] = _closed_node(1)
    defs["anomaly_score_definition"] = _closed_node("image_anomaly_score")
    defs["score_orientation"] = _closed_node("higher_is_more_anomalous")
    defs["sample_unit"] = _closed_node("image")
    defs["train_split"] = _closed_node("train")
    defs["validation_split"] = _closed_node("validation")
    defs["test_split"] = _closed_node("test")
    defs["label_semantics"] = _closed_node("anomaly_binary")
    defs["threshold_policy"] = _closed_node("validation_selected")
    defs["threshold_selection_metric"] = _closed_node("f1")
    defs["pr_metric_method"] = _closed_node("average_precision")
    defs["c06_baseline_identity"] = _closed_node("padim_only")


def _write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _valid_manifest_rows():
    rows = []
    specs = [
        ("s_tr0", "train", 0, "scene_tr", "dup_tr0"),
        ("s_tr1", "train", 1, "scene_tr2", "dup_tr1"),
        ("s_va0", "validation", 0, "scene_va", "dup_va0"),
        ("s_va1", "validation", 1, "scene_va2", "dup_va1"),
        ("s_te0", "test", 0, "scene_te", "dup_te0"),
        ("s_te1", "test", 1, "scene_te2", "dup_te1"),
    ]
    for sid, split, label, scene, dup in specs:
        rows.append(
            {
                "sample_id": sid,
                "mission": "curiosity",
                "instrument": "mastcam",
                "sol": "100",
                "source_id": sid,
                "source_url": "https://example.test/" + sid,
                "relative_path": f"images/{sid}.jpg",
                "sha256": _hex(f"file-{sid}"),
                "split": split,
                "binary_label": label,
                "label_semantics": "anomaly_binary",
                "label_source": "author",
                "annotation_version": "v1",
                "scene_group_id": scene,
                "duplicate_group_id": dup,
                "notes": "",
            }
        )
    return rows


def _valid_prediction_rows():
    rows = []
    for sid, split, y in (
        ("s_va0", "validation", 0),
        ("s_va1", "validation", 1),
        ("s_te0", "test", 0),
        ("s_te1", "test", 1),
    ):
        rows.append(
            {
                "sample_id": sid,
                "split": split,
                "y_true": y,
                "anomaly_score": 0.1 if y == 0 else 0.9,
                "model_name": "padim",
                "model_version": "v1",
                "config_id": "test_cfg",
            }
        )
    return rows


def _ready_defs_with_artifacts(tmp_path: Path) -> tuple[dict, Path, Path]:
    defs = copy.deepcopy(_base_defs())
    _close_p0(defs)
    man = tmp_path / "real_manifest.csv"
    pred = tmp_path / "real_predictions.csv"
    _write_csv(man, MANIFEST_FIELDS, _valid_manifest_rows())
    _write_csv(pred, PRED_FIELDS, _valid_prediction_rows())
    digest = sha256_file(pred)
    defs["real_manifest_path"] = str(man)
    defs["raw_predictions"] = _closed_node({"path": str(pred), "sha256": digest})
    defs["re_inference_path_documented"] = {
        "value": None,
        "status": "UNKNOWN_EVIDENCE_NOT_LOCATED",
        "required_for_real_run": True,
        "sources": [],
    }
    defs["author_attestation"] = {
        "path": "paper/iac2026/reproduction/AUTHOR_ATTESTATION_C05_C06.template.md",
        "status": "pending",
        "author_verified": False,
    }
    return defs, man, pred


def _passing_audit(manifest_sha: str, predictions_sha: str) -> dict:
    return {
        "passed": True,
        "evidence_mode": "real_evidence",
        "claim_ids": ["C05", "C06"],
        "config_id": "test_cfg",
        "config_sha256": _hex("config"),
        "manifest_sha256": manifest_sha,
        "predictions_sha256": predictions_sha,
        "checkpoint_sha256": None,
        "git_head": "a" * 40,
        "git_dirty": False,
        "blockers": [],
        "errors": [],
    }


def test_current_definitions_all_layers_blocked():
    defs = _base_defs()
    result = evaluate(defs, repo=REPO)
    assert result["definition_readiness"] == "blocked"
    assert result["artifact_reference_readiness"] == "blocked"
    assert result["input_audit_readiness"] == "not_run"
    assert result["real_run_allowed"] is False
    assert result["readiness"] == "blocked"
    assert result["author_attestation_status"] == "pending"
    assert result["author_verified"] is False
    assert result["dataset_files_audited"] is False


def test_cli_exits_2_layered_fields():
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/iac2026/check_c05_c06_definition_readiness.py"),
            "--json",
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    payload = json.loads(proc.stdout)
    assert payload["real_run_allowed"] is False
    assert payload["definition_readiness"] == "blocked"
    assert payload["artifact_reference_readiness"] == "blocked"
    assert payload["input_audit_readiness"] == "not_run"


def test_author_verified_true_without_artifacts_still_blocked():
    defs = copy.deepcopy(_base_defs())
    defs["author_attestation"] = {
        "path": "paper/iac2026/reproduction/AUTHOR_ATTESTATION_C05_C06.template.md",
        "status": "provided",
        "author_verified": True,
    }
    result = evaluate(defs, repo=REPO)
    assert result["author_verified"] is True
    assert result["real_run_allowed"] is False
    assert result["definition_readiness"] == "blocked"
    assert result["artifact_reference_readiness"] == "blocked"


def test_partially_status_blocks_definition():
    defs = copy.deepcopy(_base_defs())
    _close_p0(defs)
    defs["task_level"] = {
        "value": "image_binary",
        "status": "PARTIALLY_LOCATED",
        "required_for_real_run": True,
        "sources": [],
    }
    result = evaluate(defs, repo=REPO)
    assert result["definition_readiness"] == "blocked"
    assert any("task_level" in m and "LOCATED" in m for m in result["missing"])


def test_manuscript_claim_only_blocks_definition():
    defs = copy.deepcopy(_base_defs())
    _close_p0(defs)
    defs["c06_baseline_identity"] = {
        "value": "padim_only",
        "status": "MANUSCRIPT_CLAIM_ONLY",
        "required_for_real_run": True,
        "sources": [],
    }
    result = evaluate(defs, repo=REPO)
    assert result["definition_readiness"] == "blocked"
    assert any("c06_baseline_identity" in m for m in result["missing"])


def test_path_reference_only_blocks_predictions(tmp_path):
    defs, man, _ = _ready_defs_with_artifacts(tmp_path)
    defs["raw_predictions"] = {
        "value": {"path": str(tmp_path / "missing.csv"), "sha256": _hex("x")},
        "status": "PATH_REFERENCE_ONLY",
        "required_for_real_run": True,
        "sources": [],
    }
    result = evaluate(defs, repo=REPO)
    assert result["artifact_reference_readiness"] == "blocked"
    assert result["real_run_allowed"] is False


def test_bare_string_predictions_rejected(tmp_path):
    defs, _, _ = _ready_defs_with_artifacts(tmp_path)
    defs["raw_predictions"] = _closed_node("results/fake_predictions.csv")
    result = evaluate(defs, repo=REPO)
    assert result["artifact_reference_readiness"] == "blocked"
    assert any("bare string" in m for m in result["missing"])


def test_nonexistent_predictions_path(tmp_path):
    defs, _, _ = _ready_defs_with_artifacts(tmp_path)
    defs["raw_predictions"] = _closed_node(
        {"path": str(tmp_path / "nope.csv"), "sha256": "a" * 64}
    )
    result = evaluate(defs, repo=REPO)
    assert result["artifact_reference_readiness"] == "blocked"
    assert any("missing" in m for m in result["missing"])


def test_sha_mismatch_blocks(tmp_path):
    defs, _, pred = _ready_defs_with_artifacts(tmp_path)
    defs["raw_predictions"] = _closed_node({"path": str(pred), "sha256": "b" * 64})
    result = evaluate(defs, repo=REPO)
    assert result["artifact_reference_readiness"] == "blocked"
    assert any("sha256 mismatch" in m for m in result["missing"])


def test_synthetic_fixture_predictions_forbidden(tmp_path):
    defs = copy.deepcopy(_base_defs())
    _close_p0(defs)
    man = tmp_path / "real_manifest.csv"
    _write_csv(man, MANIFEST_FIELDS, _valid_manifest_rows())
    fixture = REPO / "reproduction/iac2026/fixtures/synthetic_predictions.csv"
    if not fixture.is_file():
        # create under forbidden path name inside tmp and point relative-like
        fixture = tmp_path / "fixtures" / "synthetic_predictions.csv"
        _write_csv(fixture, PRED_FIELDS, _valid_prediction_rows())
    defs["real_manifest_path"] = str(man)
    defs["raw_predictions"] = _closed_node(
        {"path": str(fixture), "sha256": sha256_file(fixture) if fixture.is_file() else "a" * 64}
    )
    result = evaluate(defs, repo=REPO)
    assert result["artifact_reference_readiness"] == "blocked"
    assert any("forbidden" in m for m in result["missing"])


def test_header_only_manifest_blocked(tmp_path):
    defs = copy.deepcopy(_base_defs())
    _close_p0(defs)
    man = tmp_path / "empty_manifest.csv"
    _write_csv(man, MANIFEST_FIELDS, [])
    pred = tmp_path / "preds.csv"
    _write_csv(pred, PRED_FIELDS, _valid_prediction_rows())
    defs["real_manifest_path"] = str(man)
    defs["raw_predictions"] = _closed_node({"path": str(pred), "sha256": sha256_file(pred)})
    result = evaluate(defs, repo=REPO)
    assert result["artifact_reference_readiness"] == "blocked"
    assert any("no data rows" in m for m in result["missing"])


def test_synthetic_sha_in_manifest_blocked(tmp_path):
    defs, man, pred = _ready_defs_with_artifacts(tmp_path)
    rows = _valid_manifest_rows()
    rows[0]["sha256"] = "SYNTHETIC_fake_digest_00000000000000000000000000000000"
    _write_csv(man, MANIFEST_FIELDS, rows)
    defs["raw_predictions"] = _closed_node({"path": str(pred), "sha256": sha256_file(pred)})
    result = evaluate(defs, repo=REPO)
    assert result["artifact_reference_readiness"] == "blocked"
    assert any("SYNTHETIC" in m for m in result["missing"])


def test_manifest_leakage_blocked(tmp_path):
    defs, man, pred = _ready_defs_with_artifacts(tmp_path)
    rows = _valid_manifest_rows()
    rows[2]["sha256"] = rows[4]["sha256"]  # validation shares test file hash
    _write_csv(man, MANIFEST_FIELDS, rows)
    defs["raw_predictions"] = _closed_node({"path": str(pred), "sha256": sha256_file(pred)})
    result = evaluate(defs, repo=REPO)
    assert result["artifact_reference_readiness"] == "blocked"
    assert any("leakage" in m for m in result["missing"])


def test_incomplete_reinference_blocked(tmp_path):
    defs, man, _ = _ready_defs_with_artifacts(tmp_path)
    defs["raw_predictions"] = {
        "value": None,
        "status": "UNKNOWN_EVIDENCE_NOT_LOCATED",
        "required_for_real_run": True,
        "sources": [],
    }
    defs["re_inference_path_documented"] = _closed_node(
        {
            "inference_script": str(tmp_path / "infer.py"),
            "config_path": str(tmp_path / "cfg.yaml"),
            # missing remaining keys
        }
    )
    result = evaluate(defs, repo=REPO)
    assert result["artifact_reference_readiness"] == "blocked"
    assert any("re_inference missing" in m for m in result["missing"])


def test_bad_checkpoint_sha_blocks_reinference(tmp_path):
    defs, man, _ = _ready_defs_with_artifacts(tmp_path)
    script = tmp_path / "infer.py"
    cfg = tmp_path / "cfg.yaml"
    ckpt = tmp_path / "model.pth"
    env = tmp_path / "env.lock"
    for p, body in ((script, "print(1)\n"), (cfg, "x: 1\n"), (ckpt, b"weights"), (env, "lock\n")):
        if isinstance(body, bytes):
            p.write_bytes(body)
        else:
            p.write_text(body, encoding="utf-8")
    defs["raw_predictions"] = {
        "value": None,
        "status": "UNKNOWN_EVIDENCE_NOT_LOCATED",
        "required_for_real_run": True,
        "sources": [],
    }
    defs["re_inference_path_documented"] = _closed_node(
        {
            "inference_script": str(script),
            "config_path": str(cfg),
            "checkpoint_path": str(ckpt),
            "checkpoint_sha256": "c" * 64,
            "dataset_manifest": str(man),
            "environment_lock": str(env),
            "output_prediction_path": str(tmp_path / "out_preds.csv"),
        }
    )
    result = evaluate(defs, repo=REPO)
    assert result["artifact_reference_readiness"] == "blocked"
    assert any("checkpoint sha256 mismatch" in m for m in result["missing"])


def test_artifacts_ready_without_audit_still_denies_real_run(tmp_path):
    defs, man, pred = _ready_defs_with_artifacts(tmp_path)
    result = evaluate(defs, repo=REPO)
    assert result["definition_readiness"] == "ready"
    assert result["artifact_reference_readiness"] == "ready"
    assert result["manifest_structure_ready"] is True
    assert result["input_audit_readiness"] == "not_run"
    assert result["real_run_allowed"] is False
    assert result["dataset_files_audited"] is False
    assert result["predictions_sha256"] == sha256_file(pred)
    assert result["manifest_sha256"] == sha256_file(man)


def test_matching_input_audit_allows_real_run(tmp_path):
    defs, man, pred = _ready_defs_with_artifacts(tmp_path)
    audit = tmp_path / "input_audit.json"
    audit.write_text(
        json.dumps(_passing_audit(sha256_file(man), sha256_file(pred))),
        encoding="utf-8",
    )
    result = evaluate(defs, repo=REPO, input_audit_json=audit)
    assert result["definition_readiness"] == "ready"
    assert result["artifact_reference_readiness"] == "ready"
    assert result["input_audit_readiness"] == "passed"
    assert result["real_run_allowed"] is True
    assert result["readiness"] == "ready"


def test_stale_audit_hash_fails(tmp_path):
    defs, man, pred = _ready_defs_with_artifacts(tmp_path)
    audit = tmp_path / "input_audit.json"
    audit.write_text(
        json.dumps(_passing_audit("d" * 64, sha256_file(pred))),
        encoding="utf-8",
    )
    result = evaluate(defs, repo=REPO, input_audit_json=audit)
    assert result["input_audit_readiness"] == "failed"
    assert result["real_run_allowed"] is False
    assert any("manifest_sha256" in m for m in result["missing"])
