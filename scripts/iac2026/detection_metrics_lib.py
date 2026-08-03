"""Pure detection metrics (AUROC, AP, trapezoidal PR-AUC, F1) with tie-safe ranking."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def map_positive_label(y_raw: np.ndarray, positive_label: int | str) -> np.ndarray:
    pos = int(positive_label)
    return (y_raw.astype(np.int64) == pos).astype(np.int32)


def orient_scores(scores: np.ndarray, higher_means_anomalous: bool) -> np.ndarray:
    s = scores.astype(np.float64)
    if higher_means_anomalous:
        return s
    return -s


def _require_finite(scores: np.ndarray, y: np.ndarray) -> Optional[str]:
    if not np.all(np.isfinite(scores)):
        return "scores contain non-finite values"
    if not np.all(np.isfinite(y.astype(np.float64))):
        return "labels contain non-finite values"
    return None


def binary_auroc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    err = _require_finite(scores, y_true)
    if err:
        return None
    y = y_true.astype(np.int32)
    s = scores.astype(np.float64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(s, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        if j > i:
            avg = 0.5 * (ranks[order[i]] + ranks[order[j]])
            ranks[order[i : j + 1]] = avg
        i = j + 1
    sum_pos_ranks = float(ranks[y == 1].sum())
    val = (sum_pos_ranks - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    if not (0.0 <= val <= 1.0) or not np.isfinite(val):
        return None
    return float(val)


def _score_groups_descending(scores: np.ndarray) -> List[np.ndarray]:
    """Return index arrays for equal-score groups, highest score first (tie-safe)."""
    order = np.argsort(-scores, kind="mergesort")
    groups: List[np.ndarray] = []
    i = 0
    n = len(order)
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        groups.append(order[i : j + 1])
        i = j + 1
    return groups


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """Average precision with equal-score groups (row-order invariant within ties)."""
    err = _require_finite(scores, y_true)
    if err:
        return None
    y = y_true.astype(np.int32)
    s = scores.astype(np.float64)
    n_pos = int((y == 1).sum())
    if n_pos == 0 or n_pos == len(y):
        return None
    tp = 0
    fp = 0
    ap = 0.0
    for g in _score_groups_descending(s):
        g_y = y[g]
        g_pos = int((g_y == 1).sum())
        g_neg = int((g_y == 0).sum())
        tp += g_pos
        fp += g_neg
        if g_pos:
            ap += (tp / float(tp + fp)) * g_pos
    val = ap / float(n_pos)
    if not (0.0 <= val <= 1.0) or not np.isfinite(val):
        return None
    return float(val)


def average_precision_sklearn_ref(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    try:
        from sklearn.metrics import average_precision_score
    except ImportError:
        return None
    y = y_true.astype(np.int32)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    return float(average_precision_score(y, scores.astype(np.float64)))


def trapezoidal_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """Trapezoidal area under precision-recall curve with tie-grouped thresholds."""
    err = _require_finite(scores, y_true)
    if err:
        return None
    y = y_true.astype(np.int32)
    s = scores.astype(np.float64)
    n_pos = int((y == 1).sum())
    if n_pos == 0 or n_pos == len(y):
        return None
    groups = _score_groups_descending(s)
    tp = 0
    fp = 0
    recalls = [0.0]
    precisions = [1.0]
    for g in groups:
        g_y = y[g]
        tp += int((g_y == 1).sum())
        fp += int((g_y == 0).sum())
        recalls.append(tp / float(n_pos))
        precisions.append(tp / float(tp + fp))
    trapz = getattr(np, "trapezoid", None) or np.trapz
    val = float(trapz(np.asarray(precisions), np.asarray(recalls)))
    if not (0.0 <= val <= 1.0) or not np.isfinite(val):
        return None
    return val


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    return {
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
    }


def f1_precision_recall(cm: Dict[str, int]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is None or recall is None or (precision + recall) == 0:
        return None, precision, recall
    return 2.0 * precision * recall / (precision + recall), precision, recall


def select_threshold_on_validation(
    y_val: np.ndarray,
    scores_val: np.ndarray,
    *,
    metric: str = "f1",
    tie_break: str = "highest_threshold",
) -> Tuple[Optional[float], Optional[float]]:
    """Select threshold on validation only. Returns (threshold, metric_value)."""
    y = y_val.astype(np.int32)
    s = scores_val.astype(np.float64)
    if y.size == 0:
        return None, None
    if int((y == 1).sum()) == 0 or int((y == 0).sum()) == 0:
        return None, None
    if metric != "f1":
        raise ValueError(f"unsupported threshold_selection_metric: {metric}")

    candidates = np.unique(s)
    best_t: Optional[float] = None
    best_metric = -1.0
    best_precision = -1.0
    best_recall = -1.0

    def better(t: float, f1: float, prec: Optional[float], rec: Optional[float]) -> bool:
        nonlocal best_t, best_metric, best_precision, best_recall
        p = -1.0 if prec is None else float(prec)
        r = -1.0 if rec is None else float(rec)
        if best_t is None:
            return True
        if f1 > best_metric:
            return True
        if f1 < best_metric:
            return False
        if tie_break == "highest_threshold":
            return t > float(best_t)
        if tie_break == "lowest_threshold":
            return t < float(best_t)
        if tie_break == "highest_precision":
            return p > best_precision or (p == best_precision and t > float(best_t))
        if tie_break == "highest_recall":
            return r > best_recall or (r == best_recall and t > float(best_t))
        raise ValueError(f"unsupported threshold_tie_break: {tie_break}")

    for t in candidates:
        pred = (s >= t).astype(np.int32)
        f1, prec, rec = f1_precision_recall(confusion(y, pred))
        score = -1.0 if f1 is None else float(f1)
        if better(float(t), score, prec, rec):
            best_t = float(t)
            best_metric = score
            best_precision = -1.0 if prec is None else float(prec)
            best_recall = -1.0 if rec is None else float(rec)

    return best_t, (None if best_t is None else best_metric)
