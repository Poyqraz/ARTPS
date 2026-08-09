"""Streamlit-free ARTPS detection / anomaly-map core (shared with app.py)."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cv2
import numpy as np
import torch

from src.core.false_positive_masks import (
    apply_fp_suppression,
    apply_gated_shadow_suppression,
    compute_boundary_shadow_mask,
    compute_object_in_shadow,
    compute_rover_body_mask,
    compute_shadow_like,
)
from src.core.size_distance import (
    area_min_scale,
    compute_size_distance_features,
    edge_min_scale,
    estimate_depth_scale_m,
    features_from_det_fields,
    merge_bridge_floor,
    shadow_cut_delta,
    should_reject_field_scale,
)
from src.models.optimized_autoencoder import OptimizedAutoencoder

_RUNTIME_PARAMS: dict[str, Any] = {}


def set_runtime_params(params: Mapping[str, Any] | None) -> None:
    """Optional UI/runtime overrides (app.py syncs sidebar globals here)."""
    global _RUNTIME_PARAMS
    _RUNTIME_PARAMS = dict(params or {})


def _p(key: str, default: Any) -> Any:
    return _RUNTIME_PARAMS.get(key, default)


def _normalize_map(values: np.ndarray) -> np.ndarray:
    """Harita/yoğunluk matrisini yüzde 2-98 aralığına göre normalize eder (0-1)."""
    arr = values.astype(np.float32)
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    norm = (arr - lo) / (hi - lo)
    return np.clip(norm, 0.0, 1.0)


def _normalize_percentile_map(values: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    """Yüzdelik tabanlı stabil normalizasyon; göreli depth türevleri için kullanılır."""
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.percentile(finite, lo_pct))
    hi = float(np.percentile(finite, hi_pct))
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _compute_protrusion_map(depth_map: np.ndarray | None) -> np.ndarray | None:
    """Göreli derinlikte yerel z-artığını çıkarır; yüksek değerler yerel çıkıntıyı temsil eder."""
    if depth_map is None:
        return None
    depth_norm = _normalize_percentile_map(depth_map)
    h, w = depth_norm.shape[:2]
    kernel = max(9, ((min(h, w) // 12) | 1))
    local_ground = cv2.GaussianBlur(depth_norm, (kernel, kernel), 0)
    residual = depth_norm - local_ground
    return _normalize_percentile_map(np.clip(residual, 0.0, None), 5.0, 99.0)


def _crop_rgb(image_rgb_float: np.ndarray, x: int, y: int, w: int, h: int, margin: float = 0.10) -> np.ndarray:
    """128x128 uzayındaki görüntüden güvenli crop (float RGB [0,1])."""
    H, W = image_rgb_float.shape[:2]
    mx = int(round(w * float(margin)))
    my = int(round(h * float(margin)))
    x1 = max(0, int(x) - mx)
    y1 = max(0, int(y) - my)
    x2 = min(W, int(x) + int(w) + mx)
    y2 = min(H, int(y) + int(h) + my)
    if x2 <= x1 or y2 <= y1:
        return image_rgb_float
    crop = image_rgb_float[y1:y2, x1:x2]
    if crop.ndim != 3 or crop.shape[2] != 3:
        crop = np.repeat(crop[..., None], 3, axis=2)
    return crop.astype(np.float32, copy=False)

def _extract_region_latent(autoencoder: OptimizedAutoencoder, image_rgb_float: np.ndarray, det: dict, device: torch.device) -> np.ndarray:
    """Her aday bölge için AE bottleneck (latent) vektörü çıkar."""
    try:
        x, y, w, h = int(det.get("x", 0)), int(det.get("y", 0)), int(det.get("w", 0)), int(det.get("h", 0))
        crop = _crop_rgb(image_rgb_float, x, y, w, h, margin=float(_p("policy_crop_margin", 0.10)))
        crop_u8 = (np.clip(crop, 0.0, 1.0) * 255.0).astype(np.uint8)
        crop_u8 = cv2.resize(crop_u8, (128, 128), interpolation=cv2.INTER_AREA)
        crop_f = crop_u8.astype(np.float32) / 255.0
        t = torch.from_numpy(crop_f).float().permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    z = autoencoder.encode(t)
            else:
                z = autoencoder.encode(t)
        return z.squeeze(0).detach().cpu().numpy().astype(np.float32)
    except Exception:
        # Fallback: detektör çalışsın; latent yoksa sıfır vektör
        return np.zeros((1024,), dtype=np.float32)


def _pool_region(map_2d: np.ndarray | None, det: dict) -> float:
    if map_2d is None:
        return 0.0
    try:
        x, y, w, h = int(det.get("x", 0)), int(det.get("y", 0)), int(det.get("w", 0)), int(det.get("h", 0))
        hh, ww = map_2d.shape[:2]
        x1 = max(0, min(x, ww - 1))
        y1 = max(0, min(y, hh - 1))
        x2 = max(x1 + 1, min(x + w, ww))
        y2 = max(y1 + 1, min(y + h, hh))
        region = map_2d[y1:y2, x1:x2]
        if region.size == 0:
            return 0.0
        return float(np.mean(region))
    except Exception:
        return 0.0


def _extract_region(map_2d: np.ndarray | None, det: dict) -> np.ndarray | None:
    if map_2d is None:
        return None
    try:
        x, y, w, h = int(det.get("x", 0)), int(det.get("y", 0)), int(det.get("w", 0)), int(det.get("h", 0))
        hh, ww = map_2d.shape[:2]
        x1 = max(0, min(x, ww - 1))
        y1 = max(0, min(y, hh - 1))
        x2 = max(x1 + 1, min(x + w, ww))
        y2 = max(y1 + 1, min(y + h, hh))
        region = np.asarray(map_2d[y1:y2, x1:x2], dtype=np.float32)
        if region.size == 0:
            return None
        return region
    except Exception:
        return None


def _append_detection_geomorph_metrics(det: dict, depth_map: np.ndarray | None, protrusion_map: np.ndarray | None) -> None:
    depth_region = _extract_region(depth_map, det)
    if depth_region is not None:
        depth_norm = _normalize_percentile_map(depth_region, 2.0, 98.0)
        det["depth_span"] = float(np.max(depth_norm) - np.min(depth_norm))
    protrusion_region = _extract_region(protrusion_map, det)
    if protrusion_region is None:
        return
    det["z_peak"] = float(np.max(protrusion_region))
    det["z_mean"] = float(np.mean(protrusion_region))
    det["z_std"] = float(np.std(protrusion_region))


def _annotate_det_size_distance(
    det: dict,
    img_hw: tuple[int, int],
    *,
    proximity_w: np.ndarray | None = None,
    depth_map: np.ndarray | None = None,
) -> None:
    """Write size/distance fields onto a detection dict (proposal + score paths)."""
    H, W = int(img_hw[0]), int(img_hw[1])
    x = int(det.get("x", 0))
    y = int(det.get("y", 0))
    w = max(1, int(det.get("w", 1)))
    h = max(1, int(det.get("h", 1)))
    y2, x2 = min(H, y + h), min(W, x + w)
    prox_crop = proximity_w[y:y2, x:x2] if proximity_w is not None and y2 > y and x2 > x else None
    depth_crop = depth_map[y:y2, x:x2] if depth_map is not None and y2 > y and x2 > x else None
    span = det.get("depth_span")
    feat = compute_size_distance_features(
        w=w,
        h=h,
        img_hw=(H, W),
        depth_crop=depth_crop,
        proximity_crop=prox_crop,
        depth_span=float(span) if span is not None else None,
        depth_scale_m=estimate_depth_scale_m(depth_map) if depth_map is not None else None,
    )
    det["relative_far"] = feat.relative_far
    det["apparent_size"] = feat.apparent_size
    det["area_ratio"] = feat.area_ratio
    det["metric_proxy"] = feat.metric_proxy
    det["metric_size"] = feat.metric_size
    det["size_distance_band"] = feat.band


def _fuse_object_scores(
    *,
    detector_conf: float,
    combined_pool: float,
    padim_pool: float,
    patchcore_pool: float,
    depth_pool: float,
    global_known_value: float,
) -> tuple[float, float, float]:
    """Nesne puanini anomaly destegiyle kapili olarak birlestir."""
    local_value = float(
        np.clip(
            0.55 * global_known_value + 0.25 * depth_pool + 0.20 * combined_pool,
            0.0,
            1.0,
        )
    )
    anomaly_score = float(
        np.clip(
            0.50 * combined_pool
            + 0.20 * padim_pool
            + 0.15 * patchcore_pool
            + 0.10 * detector_conf
            + 0.05 * (1.0 - depth_pool),
            0.0,
            1.0,
        )
    )
    # ponytail: image-level known value zayif anomaly kutularini sisirmesin.
    final_score = float(np.clip(anomaly_score * (0.70 + 0.30 * local_value), 0.0, 1.0))
    return local_value, anomaly_score, final_score


def _size_distance_policy_enabled() -> bool:
    return bool(_p("size_distance_policy", True))


def _fp_suppression_enabled() -> bool:
    return bool(_p("fp_suppression_enabled", True))


def _should_keep_detection(det: dict, image_shape: tuple[int, int, int]) -> tuple[bool, str]:
    """Benchmark kaynakli hafif post-filter. Returns (keep, reason_code)."""
    h_img, w_img = image_shape[:2]
    y = int(det.get("y", 0))
    w = max(1, int(det.get("w", 1)))
    h = max(1, int(det.get("h", 1)))
    area_ratio = float((w * h) / max(1, h_img * w_img))
    combined_pool = float(det.get("combined_pool", 0.0))
    comb_mean = float(det.get("comb_mean", 0.0))
    edge_mean = float(det.get("edge_mean", 0.0))
    detector_conf = float(det.get("detector_conf", det.get("score", 0.0)))
    support = max(
        combined_pool,
        detector_conf,
        float(det.get("padim_pool", 0.0)),
        float(det.get("patchcore_pool", 0.0)),
    )
    sd_on = _size_distance_policy_enabled()
    feat = features_from_det_fields(det)
    if feat is None:
        feat = compute_size_distance_features(w=w, h=h, img_hw=(h_img, w_img))
    if sd_on and should_reject_field_scale(feat, support):
        return False, "field_scale_rejection"
    near_top = y <= max(12, int(0.06 * h_img))
    very_wide = (w / max(1, h)) >= 8.0
    src = str(det.get("proposal_source", ""))
    if area_ratio >= 0.12 and support < 0.02:
        return False, "candidate_score_filtering"
    if near_top and very_wide and combined_pool < 0.18:
        return False, "border_mask"
    if src == "heuristic_merged":
        if area_ratio >= 0.10:
            return False, "size_distance_policy_rejection"
        if area_ratio >= 0.18 and y < int(0.20 * h_img):
            return False, "border_mask"
        if (w / max(1, h)) >= 5.0 and combined_pool < 0.04:
            return False, "candidate_score_filtering"
        if h >= int(0.5 * h_img) and support < 0.03:
            return False, "candidate_score_filtering"
    if src == "heuristic_detail_first":
        if (w / max(1, w_img)) > 0.85 and (h / max(1, h_img)) < 0.08:
            return False, "invalid_geometry"
        if y > int(0.80 * h_img) and (w / max(1, w_img)) > 0.70:
            return False, "border_mask"
        fine_proxy = edge_mean + comb_mean
        fine_local = float(det.get("fine_local", 0.0))
        recall_signal = max(fine_proxy, fine_local, float(det.get("score", 0.0)))
        if recall_signal >= 0.010 or support >= 0.003:
            return True, "kept"
        return False, "all_proposals_below_localization_threshold"
    if src == "heuristic_plateau":
        fill_ratio = float(det.get("fill_ratio", 1.0))
        if area_ratio >= 0.12 and fill_ratio < 0.70:
            return False, "invalid_geometry"
        if float(det.get("plateau_mass", 0.0)) > 0 and float(det.get("score", 0.0)) >= 0.006:
            return True, "kept"
    if src == "heuristic":
        far_small_ok = sd_on and feat.band == "far_small"
        if area_ratio < 0.003 and support < 0.035 and not far_small_ok:
            return False, "size_distance_policy_rejection"
        if near_top and area_ratio < 0.01 and support < 0.05:
            return False, "border_mask"
    if support < 0.015:
        fine_proxy = edge_mean + comb_mean
        if src in {"heuristic_relaxed", "heuristic_peaks", "heuristic_plateau", "heuristic_detail_first"} and (
            support >= 0.006 or fine_proxy >= 0.035
        ):
            return True, "kept"
        # far_small: weak-support soft keep (recall); do not harden
        if sd_on and feat.band == "far_small" and (support >= 0.004 or fine_proxy >= 0.025):
            return True, "kept"
        return False, "candidate_score_filtering"
    return True, "kept"


def _boxes_axis_overlap_ratio(a: dict, b: dict) -> tuple[float, float]:
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    x_overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
    y_overlap = max(0, min(ay2, by2) - max(ay1, by1))
    x_ratio = x_overlap / max(1.0, min(a["w"], b["w"]))
    y_ratio = y_overlap / max(1.0, min(a["h"], b["h"]))
    return float(x_ratio), float(y_ratio)


def _bridge_strength(combined: np.ndarray, a: dict, b: dict, pad: int = 8) -> float:
    x1 = max(0, min(a["x"], b["x"]) - pad)
    y1 = max(0, min(a["y"], b["y"]) - pad)
    x2 = min(combined.shape[1], max(a["x"] + a["w"], b["x"] + b["w"]) + pad)
    y2 = min(combined.shape[0], max(a["y"] + a["h"], b["y"] + b["h"]) + pad)
    region = combined[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0
    return float(np.mean(region))


# ponytail: recall_ablation preset — export/benchmark only
_RECALL_ABLATION_PRESETS: dict[str, dict[str, bool]] = {
    "full": {"boost": True, "cap": True},
    "slim": {"boost": True, "cap": True},
    "no_boost": {"boost": False, "cap": True},
    "no_cap": {"boost": True, "cap": False},
}


def _recall_ablation_flags() -> dict[str, bool]:
    preset = str(_p("recall_ablation", "slim")).lower()
    return dict(_RECALL_ABLATION_PRESETS.get(preset, _RECALL_ABLATION_PRESETS["slim"]))


def _recall_tier(combined: np.ndarray, fine_detail: np.ndarray) -> str:
    """off | sparse (rocky recall) | clutter (penalty gevşetme)."""
    try:
        rocky_sparse = bool(
            float(np.percentile(combined, 50)) < 0.08
            and float(np.max(fine_detail)) > 0.12
            and float(np.mean(fine_detail > np.percentile(fine_detail, 90))) > 0.012
        )
        if rocky_sparse:
            return "sparse"
        global_clutter = bool(
            float(np.percentile(combined, 95)) < 0.12
            and float(np.percentile(fine_detail, 90)) > 0.08
        )
        local_clutter = bool(
            float(np.percentile(combined, 50)) < 0.08
            and float(np.max(fine_detail)) > 0.15
            and float(np.mean(fine_detail > np.percentile(fine_detail, 85))) > 0.02
        )
        if global_clutter or local_clutter:
            return "clutter"
        return "off"
    except Exception:
        return "off"


def _is_clutter_mode(combined: np.ndarray, fine_detail: np.ndarray) -> bool:
    """ponytail: combined zayıf ama fine_detail güçlü → rocky/clutter recall modu."""
    return _recall_tier(combined, fine_detail) in {"sparse", "clutter"}


def _is_rocky_recall_mode(combined: np.ndarray, fine_detail: np.ndarray) -> bool:
    """Dar recall modu: yalnız düşük combined + seyrek fine-detail tepeleri."""
    return _recall_tier(combined, fine_detail) == "sparse"


def _should_run_detail_first_recall(
    combined: np.ndarray,
    fine_detail: np.ndarray,
    detections: list[dict],
) -> bool:
    """ponytail: detail-first yalnız boş veya rocky-benzeri seyrek sahnede."""
    if len(detections) == 0:
        return True
    if len(detections) >= 2:
        return False
    return _is_rocky_recall_mode(combined, fine_detail)


def _hysteresis_mask(
    seed_map: np.ndarray,
    hi_pct: float,
    lo_pct: float,
    *,
    max_iter: int = 10,
) -> np.ndarray:
    hi = float(np.percentile(seed_map, hi_pct))
    lo = float(np.percentile(seed_map, lo_pct))
    high_mask = (seed_map >= hi).astype(np.uint8)
    low_mask = (seed_map >= lo).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    seeds = (high_mask * 255).astype(np.uint8)
    low = (low_mask * 255).astype(np.uint8)
    prev = np.zeros_like(seeds)
    for _ in range(max_iter):
        dil = cv2.dilate(seeds, kernel, iterations=1)
        seeds = cv2.bitwise_and(dil, low)
        if np.array_equal(seeds, prev):
            break
        prev = seeds.copy()
    mask = cv2.morphologyEx(seeds, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def _nms_topk(detections: list[dict], *, nms_iou: float, top_k: int) -> list[dict]:
    ordered = sorted(detections, key=lambda d: float(d.get("score", 0.0)), reverse=True)
    kept: list[dict] = []
    for det in ordered:
        if all(_bbox_iou_xywh(det, k) < nms_iou for k in kept):
            kept.append(det)
    return kept[:max(1, int(top_k))]


def _append_unique_detections(
    detections: list[dict],
    extra: list[dict],
    *,
    nms_iou: float,
    max_add: int,
) -> list[dict]:
    if not extra:
        return detections

    out = list(detections)
    added = 0
    for det in sorted(extra, key=lambda d: float(d.get("score", 0.0)), reverse=True):
        if added >= max_add:
            break
        if all(_bbox_iou_xywh(det, kept) < nms_iou for kept in out):
            out.append(det)
            added += 1
    return out


def _bbox_iou_xywh(a: dict, b: dict) -> float:
    ax1, ay1, aw, ah = a["x"], a["y"], a["w"], a["h"]
    bx1, by1, bw, bh = b["x"], b["y"], b["w"], b["h"]
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / max(union, 1e-6))


def _collect_plateau_detections(
    combined: np.ndarray,
    *,
    area_min: float,
    percentile: float = 91.0,
    max_plateaus: int = 3,
    min_fill_ratio: float = 0.40,
    max_area_ratio: float = 0.12,
) -> list[dict]:
    """Heatmap plato CC pass: boulder fragmentation için tek bbox / plato.

    Seyrek rocky field'ları (MORPH_CLOSE ile birleşen küçük tepeler) alan-ölçekli
    kutuya çevirmemek için fill_ratio + max_area_ratio kapısı uygular.
    """
    H, W = combined.shape[:2]
    img_area = float(H * W)
    th = float(np.percentile(combined, percentile))
    mask = (combined >= th).astype(np.uint8)
    # ponytail: 5x5 CLOSE ayrı kayaları tek CC yapıyordu; 3x3 yeterli
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n_labels, labels = cv2.connectedComponents(mask)
    out: list[dict] = []
    for lab in range(1, n_labels):
        ys, xs = np.where(labels == lab)
        if ys.size == 0:
            continue
        if float(ys.size) < area_min:
            continue
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        bbox_area = float(max(1, (x2 - x1) * (y2 - y1)))
        fill_ratio = float(ys.size) / bbox_area
        area_ratio = float(bbox_area / max(1.0, img_area))
        # Seyrek / alan-ölçekli plato = rocky clutter, nesne değil
        if fill_ratio < float(min_fill_ratio):
            continue
        if area_ratio > float(max_area_ratio) and fill_ratio < 0.70:
            continue
        region = combined[y1:y2, x1:x2]
        mass = float(np.sum(region))
        score = mass / max(1.0, float(region.size))
        out.append(
            {
                "x": x1,
                "y": y1,
                "w": x2 - x1,
                "h": y2 - y1,
                "score": float(score),
                "poly": None,
                "proposal_source": "heuristic_plateau",
                "plateau_mass": mass,
                "area_ratio": area_ratio,
                "fill_ratio": fill_ratio,
                "comb_mean": float(score),
                "edge_mean": float(np.percentile(region, 90)) if region.size else float(score),
            }
        )
        _annotate_det_size_distance(out[-1], (H, W))
    out = sorted(out, key=lambda d: float(d.get("plateau_mass", d.get("score", 0.0))), reverse=True)
    return out[:max_plateaus]


def _fuse_with_plateau_detections(
    contour_dets: list[dict],
    plateau_dets: list[dict],
    *,
    iou_replace: float = 0.35,
    min_standalone_area_ratio: float = 0.015,
) -> list[dict]:
    """Plato kutuları contour fragmanlarının üzerine bindir; küçük plato FP ekleme."""
    if not plateau_dets:
        return contour_dets
    kept = [d for d in contour_dets]
    for plat in plateau_dets:
        overlap_idx = [
            i for i, det in enumerate(kept)
            if _bbox_iou_xywh(det, plat) >= iou_replace
        ]
        if overlap_idx:
            for i in sorted(overlap_idx, reverse=True):
                kept.pop(i)
            kept.append(plat)
        elif float(plat.get("area_ratio", 0.0)) >= min_standalone_area_ratio:
            kept.append(plat)
    return kept


def _region_proposal_score(region: np.ndarray) -> float:
    if region.size == 0:
        return 0.0
    return float(0.65 * np.percentile(region, 75) + 0.35 * (np.sum(region) / region.size))


def _should_merge_proposals(a: dict, b: dict, combined: np.ndarray, diag: float, merge_iou: float, merge_tol: float) -> bool:
    axc = a["x"] + a["w"] / 2.0
    ayc = a["y"] + a["h"] / 2.0
    bxc = b["x"] + b["w"] / 2.0
    byc = b["y"] + b["h"] / 2.0
    center_dist = float(np.hypot(axc - bxc, ayc - byc))
    x_overlap_ratio, y_overlap_ratio = _boxes_axis_overlap_ratio(a, b)
    bridge = _bridge_strength(combined, a, b)
    ax2 = a["x"] + a["w"]
    ay2 = a["y"] + a["h"]
    bx2 = b["x"] + b["w"]
    by2 = b["y"] + b["h"]
    gap_x = max(0.0, max(a["x"], b["x"]) - min(ax2, bx2))
    gap_y = max(0.0, max(a["y"], b["y"]) - min(ay2, by2))
    size_scale = max(a["w"], a["h"], b["w"], b["h"])
    close_centers = center_dist < merge_tol * diag * 0.02
    # ponytail: 1.5*size_scale rocky field'da zincir merge → tek dev kutu
    close_gap = max(gap_x, gap_y) < max(16.0, 0.8 * size_scale)
    aligned = x_overlap_ratio > 0.35 or y_overlap_ratio > 0.35
    if _size_distance_policy_enabled():
        img_hw = (int(combined.shape[0]), int(combined.shape[1]))
        feat_a = features_from_det_fields(a) or compute_size_distance_features(
            w=int(a["w"]), h=int(a["h"]), img_hw=img_hw
        )
        feat_b = features_from_det_fields(b) or compute_size_distance_features(
            w=int(b["w"]), h=int(b["h"]), img_hw=img_hw
        )
        bridge_need = merge_bridge_floor(feat_a, feat_b, 0.05)
        bridge_need_tight = merge_bridge_floor(feat_a, feat_b, 0.06)
    else:
        bridge_need = 0.05
        bridge_need_tight = 0.06
    return bool(
        close_centers
        or ((aligned and close_gap) and bridge > bridge_need)
        or (
            _bridge_strength(combined, a, b, pad=4) > bridge_need_tight
            and center_dist < max(22.0, 0.40 * size_scale)
        )
    )


def _collect_detection_from_contour(
    cnt,
    *,
    combined: np.ndarray,
    grad_mag_n: np.ndarray,
    depth_edge_n: np.ndarray,
    depth_n_for_region: np.ndarray,
    proximity_w: np.ndarray,
    shadow_like: np.ndarray | None,
    illumination_edge: np.ndarray | None,
    spec_mask: np.ndarray | None,
    boundary_shadow: np.ndarray | None,
    rover_body: np.ndarray | None,
    lowvar_mask: np.ndarray | None,
    fine_detail: np.ndarray,
    area_min: float,
    clutter_mode: bool = False,
) -> dict | None:
    H, W = combined.shape[:2]
    x, y, w, h = cv2.boundingRect(cnt)
    y1c, y2c = max(0, y), min(H, y + h)
    x1c, x2c = max(0, x), min(W, x + w)
    prox_crop = proximity_w[y1c:y2c, x1c:x2c] if (y2c > y1c and x2c > x1c) else None
    depth_crop = depth_n_for_region[y1c:y2c, x1c:x2c] if (y2c > y1c and x2c > x1c) else None
    feat = compute_size_distance_features(
        w=w, h=h, img_hw=(H, W),
        depth_crop=depth_crop,
        proximity_crop=prox_crop,
    )
    region_far = float(feat.relative_far)
    if _size_distance_policy_enabled():
        local_area_min = area_min * area_min_scale(feat)
    else:
        local_area_min = area_min * (0.35 + 0.65 * (1.0 - region_far))
    if cv2.contourArea(cnt) < local_area_min:
        return None
    rect = cv2.minAreaRect(cnt)
    box_pts = cv2.boxPoints(rect).astype(np.intp)
    y1, y2 = y1c, y2c
    x1, x2 = x1c, x2c
    region = combined[y1:y2, x1:x2]
    region_edges = grad_mag_n[y1:y2, x1:x2]
    region_shadow = shadow_like[y1:y2, x1:x2] if shadow_like is not None else None
    region_illum = illumination_edge[y1:y2, x1:x2] if illumination_edge is not None else None
    region_spec = spec_mask[y1:y2, x1:x2] if spec_mask is not None else None
    region_boundary = boundary_shadow[y1:y2, x1:x2] if boundary_shadow is not None else None
    region_rover = rover_body[y1:y2, x1:x2] if rover_body is not None else None
    region_prox = proximity_w[y1:y2, x1:x2]
    prox_mean = float(np.mean(region_prox)) if region_prox.size else 0.0
    comb_mean = float(np.mean(region)) if region.size else 0.0
    edge_mean = float(np.mean(region_edges)) if region_edges.size else 0.0
    shadow_pen = float(np.mean(region_shadow)) if (region_shadow is not None and region_shadow.size) else 0.0
    illum_pen = float(np.mean(region_illum)) if (region_illum is not None and region_illum.size) else 0.0
    spec_pen = float(np.mean(region_spec)) if (region_spec is not None and region_spec.size) else 0.0
    boundary_pen = float(np.mean(region_boundary)) if (region_boundary is not None and region_boundary.size) else 0.0
    rover_pen = float(np.mean(region_rover)) if (region_rover is not None and region_rover.size) else 0.0
    lowvar_pen = float(np.mean(lowvar_mask[y1:y2, x1:x2])) if lowvar_mask is not None else 0.0
    fine_local = float(np.mean(fine_detail[y1:y2, x1:x2])) if (y2 > y1 and x2 > x1) else 0.0
    prox_weight = 0.20 * (1.0 - 0.6 * region_far)
    lowvar_pen *= 1.0 - 0.45 * region_far
    penalty_scale = 0.8 if clutter_mode else 1.0
    score = 0.5 * comb_mean + 0.25 * edge_mean + prox_weight * prox_mean + 0.05 * fine_local - penalty_scale * (0.35 * shadow_pen + 0.20 * illum_pen + 0.30 * spec_pen + 0.30 * boundary_pen + 0.45 * rover_pen + 0.25 * lowvar_pen)
    score = float(max(0.0, score))
    if _size_distance_policy_enabled():
        e_scale = edge_min_scale(feat)
        sh_cut = float(_p('shadow_cut', 0.45)) + shadow_cut_delta(feat)
        im_edge_min = float(_p('img_edge_min', 0.10)) * e_scale
        dp_edge_min = float(_p('depth_edge_min', 0.08)) * (0.55 + 0.45 * e_scale)
        sp_cut = float(_p('spec_cut', 0.50)) + 0.8 * shadow_cut_delta(feat)
    else:
        sh_cut = float(_p('shadow_cut', 0.45)) + 0.10 * region_far
        im_edge_min = float(_p('img_edge_min', 0.10)) * (1.0 - 0.35 * region_far)
        dp_edge_min = float(_p('depth_edge_min', 0.08)) * (1.0 - 0.45 * region_far)
        sp_cut = float(_p('spec_cut', 0.50)) + 0.08 * region_far
    depth_edge_local = float(np.mean(depth_edge_n[y1:y2, x1:x2])) if (y2 > y1 and x2 > x1) else 0.0
    if shadow_pen > sh_cut and edge_mean < im_edge_min and depth_edge_local < dp_edge_min:
        return None
    if spec_pen > sp_cut and edge_mean < im_edge_min and depth_edge_local < dp_edge_min:
        return None
    if boundary_pen > 0.35 or rover_pen > 0.30:
        return None
    if lowvar_pen > 0.6 and edge_mean < im_edge_min:
        return None
    return {
        "x": int(x), "y": int(y), "w": int(w), "h": int(h),
        "score": float(score),
        "poly": box_pts.tolist(),
        "comb_mean": float(comb_mean),
        "edge_mean": float(edge_mean),
        "prox_mean": float(prox_mean),
        "shadow_pen": float(shadow_pen),
        "illum_pen": float(illum_pen),
        "spec_pen": float(spec_pen),
        "lowvar_pen": float(lowvar_pen),
        "proposal_source": "heuristic",
        "relative_far": feat.relative_far,
        "apparent_size": feat.apparent_size,
        "area_ratio": feat.area_ratio,
        "metric_proxy": feat.metric_proxy,
        "metric_size": feat.metric_size,
        "size_distance_band": feat.band,
    }


def _collect_peak_window_detections(
    seed_map: np.ndarray,
    *,
    top_k: int,
    rover_body: np.ndarray | None,
    boundary_shadow: np.ndarray | None,
    peak_percentile: float = 99.3,
    window_scale: float = 0.08,
) -> list[dict]:
    """Bos kalan sahnelerde local peak pencereleri ile recall kurtar."""
    H, W = seed_map.shape[:2]
    work = seed_map.copy().astype(np.float32)
    if rover_body is not None:
        work *= (1.0 - 0.9 * rover_body.astype(np.float32))
    if boundary_shadow is not None:
        work *= (1.0 - 0.8 * boundary_shadow.astype(np.float32))
    border = max(12, int(0.10 * min(H, W)))
    work[:border, :] *= 0.5
    work[-border:, :] *= 0.7
    work[:, :border] *= 0.7
    work[:, -border:] *= 0.7
    peak_thresh = max(0.03, float(np.percentile(work, peak_percentile)))
    if peak_thresh <= 0.0:
        return []
    dil = cv2.dilate(work, np.ones((9, 9), np.uint8), iterations=1)
    peak_mask = (work >= peak_thresh) & (work >= dil - 1e-6)
    ys, xs = np.where(peak_mask)
    if len(xs) == 0:
        return []
    order = np.argsort(work[ys, xs])[::-1]
    winsz = max(32, int(window_scale * min(H, W)))
    out: list[dict] = []
    for idx in order[: max(1, int(top_k))]:
        xc = int(xs[idx])
        yc = int(ys[idx])
        x1 = max(0, xc - winsz // 2)
        y1 = max(0, yc - winsz // 2)
        x2 = min(W, x1 + winsz)
        y2 = min(H, y1 + winsz)
        region = work[y1:y2, x1:x2]
        rover_pen = float(np.mean(rover_body[y1:y2, x1:x2])) if rover_body is not None else 0.0
        boundary_pen = float(np.mean(boundary_shadow[y1:y2, x1:x2])) if boundary_shadow is not None else 0.0
        if region.size == 0 or float(np.mean(region)) < 0.003:
            continue
        if rover_pen > 0.15 or boundary_pen > 0.20:
            continue
        out.append(
            {
                "x": int(x1),
                "y": int(y1),
                "w": int(x2 - x1),
                "h": int(y2 - y1),
                "score": float(np.mean(region)),
                "poly": None,
                "proposal_source": "heuristic_peaks",
            }
        )
        _annotate_det_size_distance(out[-1], (H, W))
    return out


def _collect_detail_first_detections(
    seed_map: np.ndarray,
    *,
    top_k: int,
    area_min: float,
    rover_body: np.ndarray | None,
    boundary_shadow: np.ndarray | None,
) -> list[dict]:
    """Fine-detail odaklı contour-free recall pass."""
    H, W = seed_map.shape[:2]
    work = seed_map.copy().astype(np.float32)
    if rover_body is not None:
        work *= (1.0 - 0.9 * rover_body.astype(np.float32))
    if boundary_shadow is not None:
        work *= (1.0 - 0.8 * boundary_shadow.astype(np.float32))
    mask = _hysteresis_mask(work, 88.0, 82.0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[dict] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if (w * h) < area_min:
            continue
        region = work[y:y + h, x:x + w]
        if region.size == 0:
            continue
        out.append(
            {
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "score": float(np.mean(region)),
                "detector_conf": float(np.mean(region)),
                "comb_mean": float(np.mean(region)),
                "edge_mean": float(np.percentile(region, 90)),
                "fine_local": float(np.percentile(region, 85)),
                "poly": None,
                "proposal_source": "heuristic_detail_first",
            }
        )
        _annotate_det_size_distance(out[-1], (H, W))
    out.sort(key=lambda d: float(d["score"]), reverse=True)
    return out[:max(1, int(top_k))]


def _merge_backend_detections(primary: list[dict], secondary: list[dict], *, iou_threshold: float) -> list[dict]:
    merged = list(primary)
    for cand in secondary:
        replaced = False
        for idx, det in enumerate(merged):
            if _bbox_iou_xywh(det, cand) >= iou_threshold:
                left = float(det.get("object_anomaly_score", det.get("score", 0.0)))
                right = float(cand.get("object_anomaly_score", cand.get("score", 0.0)))
                if right > left:
                    merged[idx] = cand
                replaced = True
                break
        if not replaced:
            merged.append(cand)
    return merged


def _boost_recall_detection_pools(det: dict) -> None:
    """Recall pass kutularinda combined_pool/detector_conf'i seed sinyaliyle destekle."""
    src = str(det.get("proposal_source", ""))
    if src not in {"heuristic_detail_first", "heuristic_relaxed", "heuristic_peaks", "heuristic_plateau"}:
        return
    seed_strength = max(
        float(det.get("comb_mean", 0.0)),
        float(det.get("edge_mean", 0.0)),
        float(det.get("fine_local", 0.0)),
        float(det.get("score", 0.0)),
        float(det.get("detector_conf", 0.0)),
    )
    det["combined_pool"] = max(float(det.get("combined_pool", 0.0)), seed_strength * 0.85)
    det["detector_conf"] = max(float(det.get("detector_conf", 0.0)), seed_strength)


def _cap_detections_if_needed(
    detections: list[dict],
    combined: np.ndarray,
    fine_detail: np.ndarray,
    *,
    max_default: int = 4,
) -> list[dict]:
    """ponytail: hills/boulder fragment spray; gerçek rocky clutter hariç üst sınır."""
    if len(detections) <= max_default:
        return detections
    detail_cnt = sum(1 for d in detections if str(d.get("proposal_source")) == "heuristic_detail_first")
    rocky_rich = float(np.percentile(combined, 90)) > 0.08
    rocky_sparse = _is_rocky_recall_mode(combined, fine_detail) and detail_cnt >= 1
    if rocky_sparse or rocky_rich:
        return detections
    return sorted(detections, key=lambda d: float(d.get("score", 0.0)), reverse=True)[:max_default]


def _score_object_detections(
    detections: list,
    *,
    original_rgb_float: np.ndarray,
    autoencoder: OptimizedAutoencoder,
    device: torch.device,
    combined_map: np.ndarray,
    depth_map: np.ndarray | None,
    protrusion_map: np.ndarray | None,
    padim_map: np.ndarray | None,
    patchcore_map: np.ndarray | None,
    global_known_value: float,
    diagnostics_candidates: list | None = None,
) -> list:
    """Detector veya heuristic proposal'lari object-level anomaly/value skoruyla yeniden puanla."""
    if not detections:
        return detections

    depth_norm = _normalize_map(depth_map) if depth_map is not None else None
    proximity_w = _normalize_map(1.0 - depth_map) if depth_map is not None else None
    kept: list[dict] = []
    for det in detections:
        det["combined_pool"] = _pool_region(combined_map, det)
        det["padim_pool"] = _pool_region(padim_map, det)
        det["patchcore_pool"] = _pool_region(patchcore_map, det)
        det["depth_pool"] = _pool_region(depth_norm, det)
        _append_detection_geomorph_metrics(det, depth_map, protrusion_map)
        _annotate_det_size_distance(
            det,
            (int(original_rgb_float.shape[0]), int(original_rgb_float.shape[1])),
            proximity_w=proximity_w,
            depth_map=depth_map,
        )
        if _recall_ablation_flags()["boost"]:
            _boost_recall_detection_pools(det)
        det["detector_conf"] = float(det.get("detector_conf", det.get("score", 0.0)))
        det["latent_z"] = _extract_region_latent(autoencoder, original_rgb_float, det, device)
        det["object_value_score"], det["object_anomaly_score"], det["score_raw"] = _fuse_object_scores(
            detector_conf=float(det["detector_conf"]),
            combined_pool=float(det["combined_pool"]),
            padim_pool=float(det["padim_pool"]),
            patchcore_pool=float(det["patchcore_pool"]),
            depth_pool=float(det["depth_pool"]),
            global_known_value=float(global_known_value),
        )
        det["score"] = det["score_raw"]
        keep, reason = _should_keep_detection(det, original_rgb_float.shape)
        det["keep_or_drop"] = "keep" if keep else "drop"
        det["drop_reason"] = "" if keep else reason
        det["mask_reason"] = reason if not keep else ""
        if diagnostics_candidates is not None:
            diagnostics_candidates.append(
                {
                    "x": int(det.get("x", 0)),
                    "y": int(det.get("y", 0)),
                    "w": int(det.get("w", 0)),
                    "h": int(det.get("h", 0)),
                    "combined_pool": float(det["combined_pool"]),
                    "depth_pool": float(det["depth_pool"]),
                    "detector_confidence": float(det["detector_conf"]),
                    "padim_pool": float(det["padim_pool"]),
                    "patchcore_pool": float(det["patchcore_pool"]),
                    "local_value": float(det["object_value_score"]),
                    "anomaly_score_before_gate": float(det["object_anomaly_score"]),
                    "final_candidate_score": float(det["score"]),
                    "keep_or_drop": det["keep_or_drop"],
                    "drop_reason": det["drop_reason"],
                    "mask_reason": det["mask_reason"],
                }
            )
        if keep:
            kept.append(det)

    return sorted(kept, key=lambda d: float(d.get("score", 0.0)), reverse=True)


def _run_detector_backend(
    detector: Any,
    image_rgb_float: np.ndarray,
    *,
    conf_threshold: float,
    nms_iou: float,
    top_k: int,
) -> list[dict]:
    image_u8 = (np.clip(image_rgb_float, 0.0, 1.0) * 255.0).astype(np.uint8)
    boxes = detector.detect(
        image_u8,
        conf_threshold=conf_threshold,
        nms_iou=nms_iou,
        max_detections=top_k,
    )
    detections: list[dict] = []
    for box in boxes:
        det = {
            "x": int(box.x),
            "y": int(box.y),
            "w": int(box.w),
            "h": int(box.h),
            "score": float(box.score),
            "detector_conf": float(box.score),
            "class_id": int(box.class_id),
            "class_name": str(box.class_name),
            "proposal_source": "yolo",
            "poly": None,
        }
        _annotate_det_size_distance(
            det, (int(image_rgb_float.shape[0]), int(image_rgb_float.shape[1]))
        )
        detections.append(det)
    return detections
def compute_combined_anomaly_map(
    original_rgb: np.ndarray,
    reconstructed_rgb: np.ndarray,
    depth_map: np.ndarray,
    *,
    hyst_high_pct: int = 97,
    hyst_low_pct: int = 92,
    nms_iou: float = 0.35,
    top_k: int = 25,
    w_recon: float = 0.50,
    w_depth: float = 0.30,
    w_texture: float = 0.20,
    edge_reinforce: float = 0.35,
):
    """Rekonstrüksiyon farkı + derinlik süreksizliği + gölge/kenar farkındalığı birleşik haritası.

    Ek olarak kenar rehberli yeniden keskinleştirme ve mühendislik odaklı kutulama uygular.

    Döndürür: (combined_map[H,W] in 0..1, detections[list of dict])
    """
    # Hedef çözünürlüğü derinlik haritası boyutu
    H, W = depth_map.shape[:2]
    orig = cv2.resize(original_rgb.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    recon = cv2.resize(reconstructed_rgb.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)

    # Rekonstrüksiyon farkı (MSE kanal başına)
    recon_diff = ((orig - recon) ** 2).mean(axis=2)
    recon_diff_n = _normalize_map(recon_diff)

    # Görüntü gri, kenar/kontrast ve gölge göstergesi
    img_u8 = (orig * 255.0).astype(np.uint8)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    Hc, Sc, Vc = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    grad_mag_n = _normalize_map(grad_mag)
    shadow_n = _normalize_map(1.0 - gray)  # koyu bölgeler yüksek

    # Derinlik süreksizliği ve yakınlık ağırlığı
    depth = depth_map.astype(np.float32)
    depth_n_for_region = _normalize_map(depth)
    dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    depth_edge = np.sqrt(dx ** 2 + dy ** 2)
    depth_edge_n = _normalize_map(depth_edge)
    proximity_w = _normalize_map(1.0 - depth)  # yakın bölgeler yüksek ağırlık

    # Derinlik Laplacian (çöküntü/çıkıntı vurgusu)
    depth_lap = cv2.Laplacian(depth, cv2.CV_32F, ksize=3)
    depth_lap_n = _normalize_map(np.abs(depth_lap))

    # Birleşik skor (ayarlanabilir ağırlıklar)
    # Not: Gölge bölgeleri sahte anomaliye yol açabildiğinden, texture_term
    # doğrudan gölgeyi yükseltmek yerine kenar ağırlıklı tutulur.
    texture_term = 0.35 * shadow_n + 0.65 * grad_mag_n
    # Laplacian katkısı UI'dan gelebilir; yoksa 0.08 varsay
    w_lap = float(_p('w_lap', 0.08))
    # İnce detay vurgusu (küçük taş, kum hatları için): çok ölçekli Laplacian + DoG
    try:
        lap3 = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        lap5 = cv2.Laplacian(gray, cv2.CV_32F, ksize=5)
        dog = cv2.GaussianBlur(gray, (0, 0), 0.8) - cv2.GaussianBlur(gray, (0, 0), 1.6)
        fine_detail = _normalize_map(np.abs(lap3) + 0.6 * np.abs(lap5) + 0.8 * np.abs(dog))
    except Exception:
        fine_detail = np.zeros_like(recon_diff_n)

    w_detail = float(_p('w_detail', 0.12))
    raw_combined = (
        w_recon * recon_diff_n
        + w_depth * depth_edge_n
        + w_texture * texture_term
        + w_lap * depth_lap_n
        + w_detail * fine_detail
    )
    # Uzak alanlari tamamen bastirmadan yakinlik etkisini daha yumusak uygula.
    proximity_mix = 0.55 * proximity_w + 0.45 * (1.0 - proximity_w)
    combined = np.clip(raw_combined * (0.5 + 0.5 * proximity_mix), 0.0, 1.0)
    raw_combined_pre_mask = combined.copy()

    # Gölge bastırma: (koyu) AND (düşük görüntü gradyanı) AND (düşük derinlik kenarı)
    # ve aydınlatma-kenar etkisi azaltımı: görüntü kenarı yüksek ama derinlik kenarı düşükse etkisini düşür.
    try:
        illumination_edge = np.clip(grad_mag_n - depth_edge_n, 0.0, 1.0)
        shadow_like = compute_shadow_like(orig, depth)
        boundary_shadow = compute_boundary_shadow_mask(orig, depth)
        rover_body = compute_rover_body_mask(depth)
        # Speküler/parlak nokta maskesi: yüksek V, düşük S ve düşük kenar
        spec_mask = np.clip(Vc * (1.0 - Sc) * (1.0 - grad_mag_n) * (1.0 - depth_edge_n), 0.0, 1.0)
        spec_mask = cv2.GaussianBlur(spec_mask, (3, 3), 0)
        # Düşük doku (varyans) haritası: küçük pencere varyansı
        gray_f32 = gray.astype(np.float32)
        k = 5
        mean = cv2.boxFilter(gray_f32, ddepth=-1, ksize=(k, k), normalize=True)
        mean_sq = cv2.boxFilter(gray_f32 * gray_f32, ddepth=-1, ksize=(k, k), normalize=True)
        variance = np.clip(mean_sq - mean * mean, 0.0, 1.0)
        var_norm = variance / max(variance.max(), 1e-6)

        # Saha ayarlı katsayılar
        fp_on = _fp_suppression_enabled()
        alpha_shad = float(_p('alpha_shad', 0.65)) if fp_on else 0.0
        beta_shadow_obj = float(_p('beta_shadow_obj', 0.5)) if fp_on else 0.0
        alpha_boundary = 0.85 if fp_on else 0.0
        alpha_rover = 0.90 if fp_on else 0.0
        beta_illum = float(_p('beta_illum', 0.25))
        spec_gamma = float(_p('spec_gamma', 0.35))
        spec_lowvar_gamma = float(_p('spec_lowvar_gamma', 0.35))
        spec_var_thresh = float(_p('spec_var_thresh', 0.005))
        # Düşük varyans bölgeleri için ek azaltım (speküler düz alanlar)
        lowvar_mask = (var_norm < spec_var_thresh).astype(np.float32)
        lowvar_mask = cv2.GaussianBlur(lowvar_mask, (3, 3), 0)
        object_gate = compute_object_in_shadow(orig, depth)
        combined = apply_gated_shadow_suppression(
            combined,
            shadow_like,
            object_gate,
            alpha_shad=alpha_shad,
            beta=beta_shadow_obj,
            gamma_recall=0.08,
        )
        combined = np.clip(
            combined
            - beta_illum * illumination_edge
            - spec_gamma * spec_mask
            - spec_lowvar_gamma * lowvar_mask,
            0.0,
            1.0,
        )
        combined = apply_fp_suppression(
            combined,
            shadow_like=None,
            boundary_shadow=boundary_shadow,
            rover_body=rover_body,
            alpha_boundary=alpha_boundary,
            alpha_rover=alpha_rover,
        )
    except Exception:
        pass

    # Kenar rehberli yeniden keskinleştirme (overlay ve kutu netliği için)
    try:
        guide_u8 = (orig * 255.0).astype(np.uint8)
        guide_gray = cv2.cvtColor(guide_u8, cv2.COLOR_RGB2GRAY)
        if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'jointBilateralFilter'):
            joint = cv2.ximgproc.jointBilateralFilter(guide_gray, (combined * 255).astype(np.uint8), d=9, sigmaColor=25, sigmaSpace=25)
            combined = joint.astype(np.float32) / 255.0
        else:
            combined = cv2.bilateralFilter((combined * 255).astype(np.uint8), d=9, sigmaColor=25, sigmaSpace=25).astype(np.float32) / 255.0
        # Guided filter ile hizalama (varsa)
        if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'guidedFilter'):
            gf = cv2.ximgproc.guidedFilter(guide_u8, (combined * 255).astype(np.uint8), radius=8, eps=1e-2)
            combined = gf.astype(np.float32) / 255.0
        # Unsharp mask + kenar vurgusu
        edges = cv2.Canny(guide_gray, 50, 150).astype(np.float32) / 255.0
        combined = np.clip(combined + edge_reinforce * (edges * (combined - cv2.GaussianBlur(combined, (0, 0), 1.0))), 0.0, 1.0)
    except Exception:
        combined = cv2.GaussianBlur(combined, (3, 3), 0.0)

    clutter_mode = _is_clutter_mode(combined, fine_detail)
    rocky_recall_mode = _is_rocky_recall_mode(combined, fine_detail)
    diagnostics = {
        "clutter_mode": bool(clutter_mode),
        "rocky_recall_mode": bool(rocky_recall_mode),
        "pre_filter_proposal_count": 0,
        "proposal_sources_breakdown": {},
        "recon_diff_n": recon_diff_n.astype(np.float32, copy=True),
        "depth_edge_n": depth_edge_n.astype(np.float32, copy=True),
        "texture_term": texture_term.astype(np.float32, copy=True),
        "raw_combined_pre_mask": raw_combined_pre_mask.astype(np.float32, copy=True),
        "w_recon": float(w_recon),
        "w_depth": float(w_depth),
        "w_texture": float(w_texture),
        "w_lap": float(w_lap),
        "w_detail": float(w_detail),
    }

    # Histerezis eşikleme ile aday bölgeler (seed-grow): daha sağlam tespit
    mask = _hysteresis_mask(combined, float(hyst_high_pct), float(hyst_low_pct))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    area_min_pct = float(_p('min_area_pct', 0.10)) / 100.0
    area_min = max(1.0, area_min_pct * H * W)
    for cnt in contours:
        det = _collect_detection_from_contour(
            cnt,
            combined=combined,
            grad_mag_n=grad_mag_n,
            depth_edge_n=depth_edge_n,
            depth_n_for_region=depth_n_for_region,
            proximity_w=proximity_w,
            shadow_like=shadow_like if 'shadow_like' in locals() else None,
            illumination_edge=illumination_edge if 'illumination_edge' in locals() else None,
            spec_mask=spec_mask if 'spec_mask' in locals() else None,
            boundary_shadow=boundary_shadow if 'boundary_shadow' in locals() else None,
            rover_body=rover_body if 'rover_body' in locals() else None,
            lowvar_mask=lowvar_mask if 'lowvar_mask' in locals() else None,
            fine_detail=fine_detail,
            area_min=area_min,
            clutter_mode=clutter_mode,
        )
        if det is not None:
            detections.append(det)

    detections = _nms_topk(detections, nms_iou=nms_iou, top_k=int(top_k))

    # Plato CC pass: boulder fragmentation azalt
    try:
        plateau_dets = _collect_plateau_detections(combined, area_min=area_min, percentile=91.0, max_plateaus=3)
        detections = _fuse_with_plateau_detections(detections, plateau_dets)
        detections = _nms_topk(detections, nms_iou=nms_iou, top_k=int(top_k))
        # ponytail: aynı plato içinde >3 fragman varsa en büyük kutuyu koru
        if len(detections) > 3 and plateau_dets:
            dominant = plateau_dets[0]
            inside = [d for d in detections if _bbox_iou_xywh(d, dominant) >= 0.2]
            outside = [d for d in detections if d not in inside]
            if len(inside) > 3:
                inside = sorted(inside, key=lambda d: d["w"] * d["h"], reverse=True)[:2]
            detections = outside + inside
    except Exception:
        pass

    # Recall kurtarma geçişi: kaya yoğun sahnelerde first-pass zayıfsa
    # fine-detail ağırlıklı contour-free recall dene.
    if _should_run_detail_first_recall(combined, fine_detail, detections):
        try:
            detail_seed = np.clip(0.25 * combined + 0.75 * fine_detail, 0.0, 1.0)
            detail_cap = min(int(top_k), 4 if len(detections) == 0 else 2)
            detail_extra = _collect_detail_first_detections(
                detail_seed,
                top_k=detail_cap,
                area_min=max(1.0, 0.15 * area_min),
                rover_body=rover_body if 'rover_body' in locals() else None,
                boundary_shadow=boundary_shadow if 'boundary_shadow' in locals() else None,
            )
            detections = _append_unique_detections(
                detections,
                detail_extra,
                nms_iou=nms_iou,
                max_add=detail_cap,
            )
        except Exception:
            pass

    # Yakın kutuları birleştir (merkez yakın + heatmap köprüsü varsa tek kutu yap)
    try:
        miou = float(_p('merge_iou', 0.15))
        mtol = float(_p('merge_tol', 0.5))
        for det in detections:
            _annotate_det_size_distance(
                det, (H, W), proximity_w=proximity_w, depth_map=depth_n_for_region
            )
        merged = []
        used = [False] * len(detections)
        diag = float(np.hypot(W, H))
        for i, a in enumerate(detections):
            if used[i]:
                continue
            axc = a['x'] + a['w'] / 2.0
            ayc = a['y'] + a['h'] / 2.0
            group = [i]
            for j, b in enumerate(detections[i + 1:], start=i + 1):
                if used[j]:
                    continue
                iou_ab = _bbox_iou_xywh(a, b)
                if iou_ab >= miou or _should_merge_proposals(a, b, combined, diag, miou, mtol):
                    group.append(j)
                    used[j] = True
            # Grupları tek kutuya birleştir
            xs = [detections[g]['x'] for g in group]
            ys = [detections[g]['y'] for g in group]
            ws = [detections[g]['w'] for g in group]
            hs = [detections[g]['h'] for g in group]
            x1 = int(min(xs))
            y1 = int(min(ys))
            x2 = int(max(xs[k] + ws[k] for k in range(len(xs))))
            y2 = int(max(ys[k] + hs[k] for k in range(len(ys))))
            region = combined[y1:y2, x1:x2]
            group_area_ratio = float(((x2 - x1) * (y2 - y1)) / max(1.0, H * W))
            # Geniş birleşik kutu yerine en güçlü fragmanı tut (nokta atışı)
            if len(group) > 1 and group_area_ratio > 0.08:
                best_idx = max(
                    group,
                    key=lambda g: _region_proposal_score(
                        combined[
                            detections[g]['y']: detections[g]['y'] + detections[g]['h'],
                            detections[g]['x']: detections[g]['x'] + detections[g]['w'],
                        ]
                    ),
                )
                merged.append(detections[best_idx])
                used[i] = True
                continue
            merged_det = {
                'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1,
                'score': _region_proposal_score(region) if region.size else a['score'],
                'poly': None,
                'proposal_source': 'heuristic_merged' if len(group) > 1 else a.get('proposal_source', 'heuristic'),
            }
            _annotate_det_size_distance(
                merged_det, (H, W), proximity_w=proximity_w, depth_map=depth_n_for_region
            )
            merged.append(merged_det)
            used[i] = True
        detections = merged
    except Exception:
        pass

    ablation_flags = _recall_ablation_flags()
    if ablation_flags["cap"]:
        detections = _cap_detections_if_needed(detections, combined, fine_detail, max_default=4)

    diagnostics["pre_filter_proposal_count"] = int(len(detections))
    src_counts: dict[str, int] = {}
    for det in detections:
        src = str(det.get("proposal_source", "unknown"))
        src_counts[src] = src_counts.get(src, 0) + 1
    diagnostics["proposal_sources_breakdown"] = src_counts
    # Ufuk maskesi: derin ve düşük gradyan alanları (genelde üst kısım)
    try:
        horizon_mask = ((depth > 0.8) & (depth_edge_n < 0.05)).astype(np.uint8)
        upper_band = np.zeros_like(horizon_mask, dtype=np.uint8)
        upper_band[: max(1, H // 3), :] = 1
        horizon_penalty = cv2.GaussianBlur((horizon_mask & upper_band).astype(np.float32), (5, 5), 0)
        combined = np.clip(combined * (1.0 - 0.12 * horizon_penalty), 0.0, 1.0)
    except Exception:
        horizon_mask = None

    try:
        fp_on = _fp_suppression_enabled()
        combined = apply_fp_suppression(
            combined,
            shadow_like=None,
            boundary_shadow=boundary_shadow if 'boundary_shadow' in locals() else None,
            rover_body=rover_body if 'rover_body' in locals() else None,
            alpha_boundary=0.85 if fp_on else 0.0,
            alpha_rover=0.95 if fp_on else 0.0,
        )
    except Exception:
        pass

    return combined.astype(np.float32), detections, diagnostics
