"""
Model-blind Streamlit annotator for validation blind-review pack.

Shows neutral image + guide only. Never loads private_mapping or inference.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from validation_blind_review import (  # noqa: E402
    BLIND_QUEUE_FIELDS,
    FORBIDDEN_VISIBLE_COLUMNS,
    FORBIDDEN_VISIBLE_SUBSTRINGS,
    assert_public_row_blind,
)

GUIDE_PATH = (
    REPO_ROOT
    / "paper"
    / "iac2026"
    / "reproduction"
    / "INDEPENDENT_EVAL_V1_ANNOTATION_GUIDE.md"
)
DEFAULT_PACK = (
    REPO_ROOT / "results" / "iac2026" / "independent_eval_v1" / "blind_review_pack"
)
LABELS = ("positive", "negative", "uncertain", "exclude")


def _assert_clean_modules() -> None:
    banned = ("torch", "torchvision", "src.models", "scripts.iac2026.baselines")
    loaded = set(sys.modules)
    for name in banned:
        if name in loaded or any(m.startswith(name + ".") for m in loaded):
            raise RuntimeError(f"inference module loaded in annotator: {name}")


def load_public_queue(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("empty queue")
        for col in reader.fieldnames:
            if col in FORBIDDEN_VISIBLE_COLUMNS:
                raise ValueError(f"forbidden column in queue: {col}")
            low = (col or "").lower()
            for bad in FORBIDDEN_VISIBLE_SUBSTRINGS:
                if bad in low and col not in BLIND_QUEUE_FIELDS:
                    raise ValueError(f"forbidden column name fragment: {col}")
        rows = list(reader)
    for row in rows:
        assert_public_row_blind(row)
    return rows


def save_public_queue(path: Path, rows: List[Dict[str, str]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BLIND_QUEUE_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in BLIND_QUEUE_FIELDS})
    tmp.replace(path)


def write_review_meta(
    path: Path,
    *,
    independent_annotator: bool,
    repeat_author_review: bool,
) -> None:
    meta = {
        "independent_review_status": "pending",
        "independent_annotator": bool(independent_annotator),
        "repeat_author_review": bool(repeat_author_review),
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "private_mapping.csv must never be loaded by this UI",
    }
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def run_streamlit(pack_dir: Path, *, repeat_author_review: bool) -> None:
    import streamlit as st
    from PIL import Image

    _assert_clean_modules()
    queue_path = pack_dir / "review_queue.csv"
    images_dir = pack_dir / "images"
    private = pack_dir / "private_mapping.csv"
    # ponytail: refuse if UI accidentally pointed at private file as queue
    if queue_path.resolve() == private.resolve():
        raise RuntimeError("refusing to open private_mapping as queue")

    st.set_page_config(page_title="validation blind review", layout="wide")
    st.title("Validation blind review")
    st.caption("Neutral image only. No paths, scores, splits, or model outputs.")

    if "queue_rows" not in st.session_state:
        st.session_state.queue_rows = load_public_queue(queue_path)
    rows: List[Dict[str, str]] = st.session_state.queue_rows
    if "idx" not in st.session_state:
        st.session_state.idx = 0

    write_review_meta(
        pack_dir / "review_meta.json",
        independent_annotator=not repeat_author_review,
        repeat_author_review=repeat_author_review,
    )

    col_img, col_side = st.columns([2, 1])
    idx = int(st.session_state.idx)
    row = rows[idx]
    img_path = images_dir / row["neutral_filename"]
    with col_img:
        st.subheader(f"{idx + 1}/{len(rows)} — {row['review_id']}")
        if img_path.is_file():
            st.image(Image.open(img_path), use_container_width=True)
        else:
            st.error(f"missing image: {row['neutral_filename']}")
    with col_side:
        if GUIDE_PATH.is_file():
            st.markdown(GUIDE_PATH.read_text(encoding="utf-8")[:4000])
        conf = st.selectbox("confidence", ["", "low", "medium", "high"])
        notes = st.text_input("notes", value=row.get("reviewer_notes") or "")

        def _save(label: str) -> None:
            updated = dict(row)
            updated["reviewer_label"] = label
            updated["reviewer_confidence"] = conf
            updated["reviewer_notes"] = notes
            updated["audit_status"] = "reviewed_local"
            rows[idx] = updated
            save_public_queue(queue_path, rows)
            st.session_state.queue_rows = rows
            st.success("saved")

        for lab in LABELS:
            if st.button(lab):
                _save(lab)
        c1, c2 = st.columns(2)
        if c1.button("prev") and idx > 0:
            st.session_state.idx = idx - 1
            st.rerun()
        if c2.button("next") and idx + 1 < len(rows):
            st.session_state.idx = idx + 1
            st.rerun()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK)
    p.add_argument(
        "--repeat-author-review",
        action="store_true",
        help="Mark meta as author repeat (independent_annotator=false)",
    )
    args = p.parse_args(argv)
    if not args.pack_dir.is_dir():
        raise SystemExit(
            f"pack dir missing: {args.pack_dir} (run build_validation_blind_review_pack.py)"
        )
    run_streamlit(args.pack_dir, repeat_author_review=args.repeat_author_review)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
