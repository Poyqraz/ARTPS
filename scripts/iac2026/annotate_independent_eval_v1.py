"""
Model-blind Streamlit annotator for independent_eval_v1.

Shows raw image + annotation guide only. Does not run ARTPS / baselines /
heatmaps / scores. Safe atomic CSV updates; refuses double-write of settled rows.
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Keep imports minimal — tests assert no inference modules are imported.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from independent_eval_annotation_schema import (  # noqa: E402
    ANNOTATION_QUEUE_FIELDS,
    ANNOTATION_VERSION,
    EXCLUSION_REASONS,
    FORBIDDEN_QUEUE_COLUMNS,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUEUE = (
    REPO_ROOT
    / "reproduction"
    / "iac2026"
    / "annotations"
    / "independent_eval_v1_annotation_queue.csv"
)
GUIDE_PATH = (
    REPO_ROOT
    / "paper"
    / "iac2026"
    / "reproduction"
    / "INDEPENDENT_EVAL_V1_ANNOTATION_GUIDE.md"
)


def _assert_clean_modules() -> None:
    banned = ("torch", "torchvision", "src.models", "scripts.iac2026.baselines")
    loaded = set(sys.modules)
    for name in banned:
        if name in loaded or any(m.startswith(name + ".") for m in loaded):
            raise RuntimeError(f"inference module loaded in annotator: {name}")


def load_queue(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("empty queue")
        for bad in FORBIDDEN_QUEUE_COLUMNS:
            if any(bad in (c or "").lower() for c in reader.fieldnames):
                raise ValueError(f"forbidden column in queue: {bad}")
        return list(reader)


def save_queue_atomic(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ANNOTATION_QUEUE_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in ANNOTATION_QUEUE_FIELDS})
    tmp.replace(path)


def row_is_settled(row: Dict[str, str]) -> bool:
    status = (row.get("inclusion_status") or "").strip()
    adj = (row.get("adjudication_status") or "").strip()
    if status in ("included", "excluded", "uncertain") and adj in (
        "resolved",
        "excluded",
    ):
        return True
    return False


def apply_label(
    row: Dict[str, str],
    *,
    binary_label: str,
    inclusion_status: str,
    exclusion_reason: str,
    label_confidence: str,
    notes: str,
    annotator_id: str,
) -> Dict[str, str]:
    if row_is_settled(row) and (row.get("binary_label") or row.get("inclusion_status")):
        raise ValueError(
            f"refusing double-write for settled candidate_id={row.get('candidate_id')}"
        )
    out = dict(row)
    out["binary_label"] = binary_label
    out["inclusion_status"] = inclusion_status
    out["exclusion_reason"] = exclusion_reason
    out["label_confidence"] = label_confidence
    out["annotation_notes"] = notes
    out["annotator_id"] = annotator_id
    out["annotation_timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out["annotation_version"] = ANNOTATION_VERSION
    if inclusion_status == "excluded":
        out["adjudication_status"] = "excluded"
    elif inclusion_status == "uncertain":
        out["adjudication_status"] = "unresolved"
    else:
        out["adjudication_status"] = "resolved"
    return out


def run_streamlit(queue_path: Path, dataset_root: Path) -> None:
    import os

    import streamlit as st
    from PIL import Image

    _assert_clean_modules()
    st.set_page_config(page_title="independent_eval_v1 annotator", layout="wide")
    st.title("independent_eval_v1 — model-blind annotation")
    st.caption("Raw image only. No scores, heatmaps, or model outputs.")

    if "queue_rows" not in st.session_state:
        st.session_state.queue_rows = load_queue(queue_path)
    rows: List[Dict[str, str]] = st.session_state.queue_rows
    if "idx" not in st.session_state:
        # Resume at first unsettled by annotation_order.
        ordered = sorted(range(len(rows)), key=lambda i: int(rows[i].get("annotation_order") or i))
        st.session_state.idx = next((i for i in ordered if not row_is_settled(rows[i])), ordered[0] if ordered else 0)

    col_img, col_side = st.columns([2, 1])
    idx = int(st.session_state.idx)
    row = rows[idx]
    rel = row["relative_path"].replace("\\", "/")
    img_path = dataset_root / rel
    with col_img:
        st.subheader(f"{idx + 1}/{len(rows)} — {row['candidate_id']}")
        if img_path.is_file():
            st.image(Image.open(img_path), use_container_width=True)
        else:
            st.error(f"missing image: {img_path}")
        st.text(rel)

    with col_side:
        if GUIDE_PATH.is_file():
            st.markdown(GUIDE_PATH.read_text(encoding="utf-8")[:4000])
        annotator_id = st.text_input("annotator_id", value=os.environ.get("USER", "annotator"))
        notes = st.text_input("notes", value=row.get("annotation_notes") or "")
        excl = st.selectbox("exclusion_reason", [""] + list(EXCLUSION_REASONS))

        def _save(updated: Dict[str, str]) -> None:
            rows[idx] = updated
            save_queue_atomic(queue_path, rows)
            st.session_state.queue_rows = rows
            st.success("saved")

        c1, c2, c3 = st.columns(3)
        if c1.button("1 positive") or st.session_state.get("_key") == "1":
            try:
                _save(
                    apply_label(
                        row,
                        binary_label="1",
                        inclusion_status="included",
                        exclusion_reason="",
                        label_confidence="medium",
                        notes=notes,
                        annotator_id=annotator_id,
                    )
                )
            except ValueError as exc:
                st.error(str(exc))
        if c2.button("0 negative"):
            try:
                _save(
                    apply_label(
                        row,
                        binary_label="0",
                        inclusion_status="included",
                        exclusion_reason="",
                        label_confidence="medium",
                        notes=notes,
                        annotator_id=annotator_id,
                    )
                )
            except ValueError as exc:
                st.error(str(exc))
        if c3.button("U uncertain"):
            try:
                _save(
                    apply_label(
                        row,
                        binary_label="",
                        inclusion_status="uncertain",
                        exclusion_reason="unresolved_ambiguity",
                        label_confidence="low",
                        notes=notes,
                        annotator_id=annotator_id,
                    )
                )
            except ValueError as exc:
                st.error(str(exc))
        if st.button("X exclude"):
            reason = excl or "other"
            try:
                _save(
                    apply_label(
                        row,
                        binary_label="",
                        inclusion_status="excluded",
                        exclusion_reason=reason,
                        label_confidence="high",
                        notes=notes,
                        annotator_id=annotator_id,
                    )
                )
            except ValueError as exc:
                st.error(str(exc))
        b, n = st.columns(2)
        if b.button("B previous"):
            st.session_state.idx = max(0, idx - 1)
            st.rerun()
        if n.button("N next"):
            st.session_state.idx = min(len(rows) - 1, idx + 1)
            st.rerun()


def main() -> None:
    import os

    _assert_clean_modules()
    root = (os.environ.get("ARTPS_DATASET_ROOT") or "").strip()
    if not root:
        print(
            "DATASET ROOT REQUIRED:\n"
            "Set ARTPS_DATASET_ROOT to the folder containing the real candidate Mars images.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    queue = Path(os.environ.get("IE1_ANNOTATION_QUEUE") or DEFAULT_QUEUE)
    # When executed via `streamlit run`, this module loads as __main__.
    try:
        run_streamlit(queue, Path(root))
    except ModuleNotFoundError as exc:
        if "streamlit" in str(exc):
            print("streamlit is required to run the interactive annotator", file=sys.stderr)
            raise SystemExit(2) from exc
        raise


if __name__ == "__main__":
    main()
