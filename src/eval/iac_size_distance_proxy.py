"""IAC size/distance policy proxy metrics (no human bbox GT)."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from src.eval.iac_shadow_proxy import MATCH_IOU, TARGET_CLASSES, bbox_iou, count_matched

PSEUDO_GT_SCORE = 0.01
FIELD_SCALE_AREA = 0.10
FAR_THRESH = 0.55
SMALL_AREA = 0.02


def _band(det: dict) -> str:
    return str(det.get("size_distance_band") or "")


def _area_ratio(det: dict) -> float:
    if "area_ratio" in det:
        return float(det["area_ratio"])
    return 0.0


def _relative_far(det: dict) -> float:
    return float(det.get("relative_far", 0.5))


def is_far_small_proxy(det: dict) -> bool:
    if _band(det) == "far_small":
        return True
    return _relative_far(det) >= FAR_THRESH and _area_ratio(det) < SMALL_AREA


def is_near_large_merged(det: dict) -> bool:
    if str(det.get("proposal_source", "")) != "heuristic_merged":
        return False
    return _band(det) == "near_large" or _area_ratio(det) >= 0.08


def is_field_scale_fp(det: dict) -> bool:
    return _area_ratio(det) >= FIELD_SCALE_AREA


def mean_matched_iou(a_list: list[dict], b_list: list[dict], iou_thresh: float = MATCH_IOU) -> float | None:
    """Self-IoU localization proxy between two runs (not GT localization)."""
    used: set[int] = set()
    ious: list[float] = []
    for a in a_list:
        best_i, best = -1, 0.0
        for i, b in enumerate(b_list):
            if i in used:
                continue
            ov = bbox_iou(a, b)
            if ov > best:
                best, best_i = ov, i
        if best_i >= 0 and best >= iou_thresh:
            used.add(best_i)
            ious.append(best)
    if not ious:
        return None
    return float(np.mean(ious))


def summarize_size_distance_image(
    *,
    class_label: str,
    dets_off: list[dict],
    dets_on: list[dict],
) -> dict[str, Any]:
    n_off, n_on = len(dets_off), len(dets_on)

    far_small_recall = None
    if class_label in TARGET_CLASSES:
        pseudo = [
            d for d in dets_off
            if is_far_small_proxy(d)
            and float(d.get("score", d.get("detector_conf", 0.0))) >= PSEUDO_GT_SCORE
        ]
        if pseudo:
            far_small_recall = float(count_matched(pseudo, dets_on) / len(pseudo))
        else:
            far_small_recall = None

    def _over_merge(dets: list[dict]) -> float:
        if not dets:
            return 0.0
        return float(sum(1 for d in dets if is_near_large_merged(d)) / len(dets))

    def _field_fpr(dets: list[dict]) -> float:
        if not dets:
            return 0.0
        return float(sum(1 for d in dets if is_field_scale_fp(d)) / len(dets))

    bands_off = Counter(_band(d) or "unknown" for d in dets_off)
    bands_on = Counter(_band(d) or "unknown" for d in dets_on)

    return {
        "n_det_off": n_off,
        "n_det_on": n_on,
        "far_small_recall": far_small_recall,
        "near_large_over_merge_off": _over_merge(dets_off),
        "near_large_over_merge_on": _over_merge(dets_on),
        "field_scale_fpr_off": _field_fpr(dets_off),
        "field_scale_fpr_on": _field_fpr(dets_on),
        "mean_matched_iou_off_on": mean_matched_iou(dets_off, dets_on),
        "band_hist_off": dict(bands_off),
        "band_hist_on": dict(bands_on),
    }


def aggregate_size_distance_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fs = [r["far_small_recall"] for r in rows if r.get("far_small_recall") is not None]
    ious = [r["mean_matched_iou_off_on"] for r in rows if r.get("mean_matched_iou_off_on") is not None]
    merge_off = [r["near_large_over_merge_off"] for r in rows]
    merge_on = [r["near_large_over_merge_on"] for r in rows]
    field_off = [r["field_scale_fpr_off"] for r in rows]
    field_on = [r["field_scale_fpr_on"] for r in rows]
    return {
        "disclaimer": (
            "Proxy metrics (class labels, OFF-run far-small pseudo-GT, self-IoU). "
            "Not human bbox GT. Lite bench is software verification only - not a performance result."
        ),
        "n_images": len(rows),
        "far_small_recall_mean": float(np.mean(fs)) if fs else None,
        "near_large_over_merge_off": float(np.mean(merge_off)) if merge_off else 0.0,
        "near_large_over_merge_on": float(np.mean(merge_on)) if merge_on else 0.0,
        "near_large_over_merge_delta": (
            float(np.mean(merge_on) - np.mean(merge_off)) if merge_off and merge_on else None
        ),
        "field_scale_fpr_off": float(np.mean(field_off)) if field_off else 0.0,
        "field_scale_fpr_on": float(np.mean(field_on)) if field_on else 0.0,
        "field_scale_fpr_delta": (
            float(np.mean(field_on) - np.mean(field_off)) if field_off and field_on else None
        ),
        "mean_matched_iou_off_on": float(np.mean(ious)) if ious else None,
        "avg_detections_off": float(np.mean([r["n_det_off"] for r in rows])) if rows else 0.0,
        "avg_detections_on": float(np.mean([r["n_det_on"] for r in rows])) if rows else 0.0,
    }
