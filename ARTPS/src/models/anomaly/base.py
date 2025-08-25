from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class AnomalyModel(ABC):
    """Anomali modelleri için basit arayüz.

    Tüm modeller 0..1 aralığında bir anomali haritası (H,W) üretmelidir.
    """

    @abstractmethod
    def prepare(self, dataset_dir: str, save_path: Optional[str] = None) -> None:
        """Modeli yalnız-normal veri kümesi ile hazırla (istatistik/özellik bankası oluştur).

        Args:
            dataset_dir: Yalnızca normal örnekleri içeren eğitim klasörü
            save_path: İsteğe bağlı istatistik kaydetme yolu
        """

    @abstractmethod
    def load(self, stats_path: str) -> None:
        """Önceden hazırlanmış istatistikleri yükle."""

    @abstractmethod
    def predict_anomaly_map(self, image_rgb_uint8: np.ndarray) -> np.ndarray:
        """0..255 RGB girişten 0..1 aralığında anomali haritası döndür.

        Args:
            image_rgb_uint8: (H,W,3) uint8 RGB
        Returns:
            (H,W) float32 0..1
        """


