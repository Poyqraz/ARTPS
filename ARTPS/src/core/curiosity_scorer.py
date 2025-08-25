from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import math


@dataclass
class CuriosityWeights:
    """İlginçlik (Curiosity) skor bileşen ağırlıkları.

    Tüm ağırlıklar [0, 1] aralığında olmalı ve toplamlarının 0'dan büyük olması beklenir.
    """
    w_known: float = 0.4               # Bilinen değer (sınıf) katkısı
    w_anomaly: float = 0.6             # AE tabanlı anomali MSE katkısı
    w_combined: float = 0.0            # Birleşik anomali haritası ortalaması katkısı
    w_depth_variance: float = 0.0      # Derinlik varyansı katkısı
    w_roughness: float = 0.0           # Pürüzlülük katkısı


class CuriosityScorer:
    """İlginçlik puanı hesaplayıcı.

    Kullanılabilir sinyalleri (bilinen değer skoru, AE MSE, birleşik anomali skoru,
    derinlik özellikleri) normalize edip ağırlıklı bir skor üretir ve katkı dökümü verir.
    """

    def __init__(self, weights: Optional[CuriosityWeights] = None) -> None:
        self.weights = weights or CuriosityWeights()

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _safe(value: Optional[float], default: float = 0.0) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _normalize_mse(mse: float, reference: float = 0.003) -> float:
        """AE MSE'yi yaklaşık [0,1] aralığına sıkıştır.
        reference: tipik eşik; 2×referans ≈ 1.0 civarı olacak şekilde tanh ölçekleme.
        """
        if reference <= 0.0:
            reference = 1e-6
        x = mse / (2.0 * reference)
        return max(0.0, min(1.0, math.tanh(x)))

    @staticmethod
    def _normalize_stat(x: float, scale: float, kind: str = "tanh") -> float:
        """Pozitif istatistiği [0,1] aralığına getirir.
        - kind == 'tanh': tanh(x/scale)
        - kind == 'log':  log(1+x)/log(1+scale)
        """
        x = max(0.0, float(x))
        scale = max(1e-6, float(scale))
        if kind == 'log':
            return max(0.0, min(1.0, math.log1p(x) / math.log1p(scale)))
        return max(0.0, min(1.0, math.tanh(x / scale)))

    def compute(
        self,
        *,
        known_value_score: Optional[float],
        anomaly_mse: Optional[float],
        combined_anomaly_score: Optional[float] = None,
        depth_features: Optional[Dict[str, float]] = None,
        reference_mse: float = 0.003,
    ) -> Tuple[float, Dict[str, float]]:
        """İlginçlik skorunu hesapla ve katkı dökümünü döndür.

        Returns: (score, breakdown)
        breakdown anahtarları: 'known', 'anomaly', 'combined', 'depth_variance', 'roughness'
        """
        depth_features = depth_features or {}
        w = self.weights

        # Bileşenleri normalize et
        known = self._clip01(self._safe(known_value_score, 0.0))
        anom = self._normalize_mse(self._safe(anomaly_mse, 0.0), reference=reference_mse)
        comb = self._clip01(self._safe(combined_anomaly_score, 0.0))

        depth_var = self._safe(depth_features.get('depth_variance'), 0.0)
        depth_var_n = self._normalize_stat(depth_var, scale=0.05, kind='tanh')  # ölçek deneysel

        rough = self._safe(depth_features.get('roughness'), 0.0)
        rough_n = self._normalize_stat(rough, scale=0.10, kind='tanh')  # ölçek deneysel

        # Ağırlıklı toplam (ağırlık toplamı > 0 ise normalize et)
        contrib_known = w.w_known * known
        contrib_anom = w.w_anomaly * anom
        contrib_comb = w.w_combined * comb
        contrib_dv = w.w_depth_variance * depth_var_n
        contrib_rough = w.w_roughness * rough_n

        weight_sum = float(w.w_known + w.w_anomaly + w.w_combined + w.w_depth_variance + w.w_roughness)
        raw = contrib_known + contrib_anom + contrib_comb + contrib_dv + contrib_rough
        score = (raw / weight_sum) if weight_sum > 1e-9 else 0.0
        score = self._clip01(score)

        breakdown = {
            'known': contrib_known,
            'anomaly': contrib_anom,
            'combined': contrib_comb,
            'depth_variance': contrib_dv,
            'roughness': contrib_rough,
            'weight_sum': weight_sum,
            'normalized_score': score,
        }
        return score, breakdown

    # --- Öğrenilebilir mod: basit lineer regresyon ile ağırlık öğrenme ---
    def fit(self, X: List[Dict[str, float]], y: List[float]) -> CuriosityWeights:
        """Basit lineer regresyon ile ağırlıkları öğren.

        X: her örnek için özellik sözlüğü. Beklenen anahtarlar:
           - known_value_score, anomaly_mse, combined_anomaly_score,
             depth_variance, roughness
        y: [0,1] aralığında ground-truth curiosity/önem skoru (veya proxy label)
        """
        import numpy as np
        from sklearn.linear_model import Ridge

        def _row(d: Dict[str, float]) -> List[float]:
            known = self._clip01(float(d.get('known_value_score', 0.0)))
            anom = self._normalize_mse(float(d.get('anomaly_mse', 0.0)))
            comb = self._clip01(float(d.get('combined_anomaly_score', 0.0)))
            dv = self._normalize_stat(float(d.get('depth_variance', 0.0)), scale=0.05)
            rf = self._normalize_stat(float(d.get('roughness', 0.0)), scale=0.10)
            return [known, anom, comb, dv, rf]

        X_mat = np.asarray([_row(r) for r in X], dtype=np.float32)
        y_vec = np.asarray(list(y), dtype=np.float32)
        # Ridge ile kararlı katsayılar; negatif katsayıları 0'a klipsleyeceğiz
        model = Ridge(alpha=1.0, fit_intercept=False, positive=True)
        model.fit(X_mat, y_vec)
        coef = model.coef_.astype(np.float32)
        # Normalizasyon: toplamı 0 değilse 1'e ölçekle
        s = float(coef.sum())
        if s <= 1e-9:
            coef = np.ones_like(coef) / float(coef.size)
        else:
            coef = coef / s
        self.weights = CuriosityWeights(
            w_known=float(coef[0]),
            w_anomaly=float(coef[1]),
            w_combined=float(coef[2]),
            w_depth_variance=float(coef[3]),
            w_roughness=float(coef[4]),
        )
        return self.weights

    def save(self, path: str) -> None:
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.weights.__dict__, f, indent=2, ensure_ascii=False)

    def load(self, path: str) -> CuriosityWeights:
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.weights = CuriosityWeights(**data)
        return self.weights


