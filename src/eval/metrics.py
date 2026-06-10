from __future__ import annotations

from typing import Iterable, Tuple
import numpy as np


def rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def abs_rel(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    denom = np.maximum(np.abs(gt), eps)
    return float(np.mean(np.abs(pred - gt) / denom))


def delta_accuracy(pred: np.ndarray, gt: np.ndarray, threshold: float = 1.25) -> float:
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    ratio = np.maximum(pred / np.maximum(gt, 1e-6), gt / np.maximum(pred, 1e-6))
    return float(np.mean(ratio < threshold))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(labels, scores))


def ndcg_at_k(relevances: Iterable[float], k: int = 10) -> float:
    rel = np.asarray(list(relevances), dtype=np.float32)[:k]
    if rel.size == 0:
        return 0.0
    dcg = float(np.sum((2 ** rel - 1) / np.log2(np.arange(2, rel.size + 2))))
    ideal = np.sort(rel)[::-1]
    idcg = float(np.sum((2 ** ideal - 1) / np.log2(np.arange(2, ideal.size + 2))))
    return 0.0 if idcg == 0.0 else dcg / idcg


def spearmanr(x: Iterable[float], y: Iterable[float]) -> Tuple[float, float]:
    from scipy.stats import spearmanr as _spr
    return _spr(list(x), list(y))


def kendalltau(x: Iterable[float], y: Iterable[float]) -> Tuple[float, float]:
    from scipy.stats import kendalltau as _kt
    return _kt(list(x), list(y))


