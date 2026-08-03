"""IAC shadow / FP proxy metrics (no human bbox GT)."""
from __future__ import annotations

from typing import Any

import numpy as np

SHADOW_DENSE_MEAN = 0.25
FLAT_SHADOW_MEAN = 0.35
OBJECT_GATE_LOW = 0.20
ROVER_FP_MEAN = 0.30
SHADOW_ROCK_GATE = 0.35
PSEUDO_GT_SCORE = 0.01
MATCH_IOU = 0.30
TARGET_CLASSES = frozenset({"rocky", "boulder"})


def bbox_iou(a: dict, b: dict) -> float:
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    iw = max(0, min(ax2, bx2) - max(a["x"], b["x"]))
    ih = max(0, min(ay2, by2) - max(a["y"], b["y"]))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return float(inter / max(union, 1e-6))


def _crop_mean(mask: np.ndarray | None, det: dict) -> float:
    if mask is None:
        return 0.0
    H, W = mask.shape[:2]
    x, y = int(det["x"]), int(det["y"])
    w, h = max(1, int(det["w"])), max(1, int(det["h"]))
    y2, x2 = min(H, y + h), min(W, x + w)
    if y2 <= y or x2 <= x:
        return 0.0
    return float(np.mean(mask[y:y2, x:x2]))


def scene_shadow_density(shadow_like: np.ndarray) -> float:
    return float(np.mean(shadow_like))


def is_shadow_dense(shadow_like: np.ndarray, thresh: float = SHADOW_DENSE_MEAN) -> bool:
    return scene_shadow_density(shadow_like) >= thresh


def is_flat_shadow_fp(
    det: dict,
    shadow_like: np.ndarray | None = None,
    object_gate: np.ndarray | None = None,
    *,
    shadow_thresh: float = FLAT_SHADOW_MEAN,
    gate_low: float = OBJECT_GATE_LOW,
) -> bool:
    if "shadow_mean" in det and "object_gate_mean" in det:
        return float(det["shadow_mean"]) >= shadow_thresh and float(det["object_gate_mean"]) <= gate_low
    return _crop_mean(shadow_like, det) >= shadow_thresh and _crop_mean(object_gate, det) <= gate_low


def is_rover_fp(det: dict, rover_body: np.ndarray | None = None, thresh: float = ROVER_FP_MEAN) -> bool:
    if "rover_mean" in det:
        return float(det["rover_mean"]) >= thresh
    return _crop_mean(rover_body, det) >= thresh


def is_shadow_rock(det: dict, object_gate: np.ndarray | None = None, thresh: float = SHADOW_ROCK_GATE) -> bool:
    if "object_gate_mean" in det:
        return float(det["object_gate_mean"]) >= thresh
    return _crop_mean(object_gate, det) >= thresh


def count_matched(gt_list: list[dict], pred_list: list[dict], iou: float = MATCH_IOU) -> int:
    used: set[int] = set()
    n = 0
    for g in gt_list:
        best_i, best = -1, 0.0
        for i, p in enumerate(pred_list):
            if i in used:
                continue
            ov = bbox_iou(g, p)
            if ov > best:
                best, best_i = ov, i
        if best_i >= 0 and best >= iou:
            used.add(best_i)
            n += 1
    return n


def match_dets(gt_list: list[dict], pred_list: list[dict], iou: float = MATCH_IOU) -> int:
    """Alias: number of GT boxes matched at IoU threshold."""
    return count_matched(gt_list, pred_list, iou=iou)

def det_mask_stats(
    det: dict,
    *,
    shadow_like: np.ndarray | None,
    object_gate: np.ndarray | None,
    rover_body: np.ndarray | None,
) -> dict[str, float]:
    return {
        "shadow_mean": _crop_mean(shadow_like, det),
        "object_gate_mean": _crop_mean(object_gate, det),
        "rover_mean": _crop_mean(rover_body, det),
    }


def summarize_image_pair(
    *,
    class_label: str,
    dets_off: list[dict],
    dets_on: list[dict],
    shadow_like: np.ndarray | None = None,
    object_gate: np.ndarray | None = None,
    rover_body: np.ndarray | None = None,
    scene_shadow_density_value: float | None = None,
    shadow_dense_flag: bool | None = None,
) -> dict[str, Any]:
    if shadow_dense_flag is not None:
        dense = bool(shadow_dense_flag)
        dens = float(scene_shadow_density_value or 0.0)
    elif shadow_like is not None:
        dens = scene_shadow_density(shadow_like)
        dense = dens >= SHADOW_DENSE_MEAN
    else:
        dens = float(scene_shadow_density_value or 0.0)
        dense = dens >= SHADOW_DENSE_MEAN
    n_off = len(dets_off)
    n_on = len(dets_on)

    def _fpr(dets: list[dict]) -> float:
        if not dets:
            return 0.0
        n_fp = sum(1 for d in dets if is_flat_shadow_fp(d, shadow_like, object_gate))
        return float(n_fp / len(dets))

    rover_fp_off = sum(1 for d in dets_off if is_rover_fp(d, rover_body))
    rover_fp_on = sum(1 for d in dets_on if is_rover_fp(d, rover_body))

    recall_proxy = None
    if class_label in TARGET_CLASSES:
        pseudo = [
            d for d in dets_off
            if not is_rover_fp(d, rover_body)
            and float(d.get("score", d.get("detector_conf", 0.0))) >= PSEUDO_GT_SCORE
        ]
        if pseudo:
            recall_proxy = float(count_matched(pseudo, dets_on) / len(pseudo))
        else:
            recall_proxy = 1.0 if not dets_on else 0.0

    shadow_rocks = [d for d in dets_off if is_shadow_rock(d, object_gate)]
    if shadow_rocks:
        shadow_rock_loss = float(1.0 - count_matched(shadow_rocks, dets_on) / len(shadow_rocks))
    else:
        shadow_rock_loss = 0.0

    return {
        "shadow_dense": dense,
        "scene_shadow_density": dens,
        "fpr_off": _fpr(dets_off) if dense else None,
        "fpr_on": _fpr(dets_on) if dense else None,
        "n_det_off": n_off,
        "n_det_on": n_on,
        "rover_fp_off": rover_fp_off,
        "rover_fp_on": rover_fp_on,
        "recall_proxy": recall_proxy,
        "shadow_rock_loss": shadow_rock_loss,
        "n_shadow_rock_off": len(shadow_rocks),
    }


def aggregate_shadow_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dense = [r for r in rows if r.get("shadow_dense")]
    fpr_off = [r["fpr_off"] for r in dense if r.get("fpr_off") is not None]
    fpr_on = [r["fpr_on"] for r in dense if r.get("fpr_on") is not None]
    recalls = [r["recall_proxy"] for r in rows if r.get("recall_proxy") is not None]
    losses = [r["shadow_rock_loss"] for r in rows]
    return {
        "disclaimer": (
            "Proxy metrics (class labels, depth/shadow masks, OFF-run pseudo-GT). "
            "Not human bbox GT. Synthetic size/distance tests are software verification only."
        ),
        "n_images": len(rows),
        "n_shadow_dense": len(dense),
        "shadow_dense_fpr_off": float(np.mean(fpr_off)) if fpr_off else None,
        "shadow_dense_fpr_on": float(np.mean(fpr_on)) if fpr_on else None,
        "shadow_dense_fpr_delta": (
            float(np.mean(fpr_on) - np.mean(fpr_off)) if fpr_off and fpr_on else None
        ),
        "rover_fp_count_off": int(sum(r.get("rover_fp_off", 0) for r in rows)),
        "rover_fp_count_on": int(sum(r.get("rover_fp_on", 0) for r in rows)),
        "target_recall_proxy_mean": float(np.mean(recalls)) if recalls else None,
        "shadow_rock_loss_mean": float(np.mean(losses)) if losses else 0.0,
        "avg_detections_off": float(np.mean([r["n_det_off"] for r in rows])) if rows else 0.0,
        "avg_detections_on": float(np.mean([r["n_det_on"] for r in rows])) if rows else 0.0,
    }
