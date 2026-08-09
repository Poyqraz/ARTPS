"""Visualization-only candidate-support overlay (ROI brackets + proposal CC + anchor)."""
from __future__ import annotations

import cv2
import numpy as np

BRACKET_COLOR = (0, 200, 80)
CONTOUR_COLOR = (40, 190, 220)
POLY_COLOR = (80, 170, 200)
ANCHOR_FILL = (255, 255, 255)
ANCHOR_EDGE = (25, 25, 25)
LABEL_MAX = 3


def overlay_geometry_counts(detections: list[dict]) -> dict[str, int]:
    n_cc = n_poly = n_fb = 0
    for det in detections:
        if det.get("support_geometry") == "cc" and det.get("support_contour"):
            n_cc += 1
        elif det.get("poly"):
            n_poly += 1
        else:
            n_fb += 1
    return {
        "n_support_contour": n_cc,
        "n_oriented_poly": n_poly,
        "n_bracket_fallback": n_fb,
    }


def candidate_xywh_scores(detections: list[dict]) -> list[dict]:
    return [
        {
            "x": int(d["x"]),
            "y": int(d["y"]),
            "w": int(d["w"]),
            "h": int(d["h"]),
            "score": float(d["score"]),
        }
        for d in detections
    ]


def _draw_open_corners(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    half = max(2, min(bw, bh) // 2)
    length = int(np.clip(round(0.20 * min(bw, bh)), 8, 28))
    length = min(length, half)
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness, cv2.LINE_AA)


def _anchor_map_xy(det: dict, combined_map: np.ndarray) -> tuple[int, int]:
    h, w = combined_map.shape[:2]
    x = int(det["x"])
    y = int(det["y"])
    bw = int(det["w"])
    bh = int(det["h"])
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + bw), min(h, y + bh)
    peak = det.get("peak_xy")
    if peak is not None:
        px, py = int(peak[0]), int(peak[1])
        if x1 <= px < x2 and y1 <= py < y2:
            return px, py
    mask = np.zeros((h, w), np.uint8)
    contour = det.get("support_contour") if det.get("support_geometry") == "cc" else None
    if contour:
        pts = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 1)
        if int(mask.sum()) == 0 and y2 > y1 and x2 > x1:
            mask[y1:y2, x1:x2] = 1
    elif y2 > y1 and x2 > x1:
        mask[y1:y2, x1:x2] = 1
    work = np.where(mask > 0, combined_map.astype(np.float32), -np.inf)
    iy, ix = np.unravel_index(int(np.argmax(work)), work.shape)
    return int(ix), int(iy)


def draw_candidate_support_overlay(
    rgb_u8: np.ndarray,
    detections: list[dict],
    combined_map: np.ndarray,
    map_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    out = rgb_u8.copy()
    mh, mw = map_hw if map_hw is not None else combined_map.shape[:2]
    h, w = out.shape[:2]
    sx = w / float(mw)
    sy = h / float(mh)
    n = len(detections)
    for i, det in enumerate(detections):
        x1 = int(round(float(det["x"]) * sx))
        y1 = int(round(float(det["y"]) * sy))
        x2 = int(round((float(det["x"]) + float(det["w"])) * sx))
        y2 = int(round((float(det["y"]) + float(det["h"])) * sy))
        _draw_open_corners(out, x1, y1, x2, y2, BRACKET_COLOR, thickness=2)
        if det.get("support_geometry") == "cc" and det.get("support_contour"):
            pts = np.asarray(det["support_contour"], dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= sx
            pts[:, 1] *= sy
            cv2.polylines(out, [np.round(pts).astype(np.int32)], True, CONTOUR_COLOR, 1, cv2.LINE_AA)
        elif det.get("poly"):
            pts = np.asarray(det["poly"], dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= sx
            pts[:, 1] *= sy
            cv2.polylines(out, [np.round(pts).astype(np.int32)], True, POLY_COLOR, 1, cv2.LINE_AA)
        ax, ay = _anchor_map_xy(det, combined_map)
        px, py = int(round(ax * sx)), int(round(ay * sy))
        cv2.circle(out, (px, py), 3, ANCHOR_EDGE, -1, cv2.LINE_AA)
        cv2.circle(out, (px, py), 2, ANCHOR_FILL, -1, cv2.LINE_AA)
        if n <= LABEL_MAX:
            cv2.putText(
                out,
                f"T{i + 1}",
                (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                BRACKET_COLOR,
                1,
                cv2.LINE_AA,
            )
    return out
