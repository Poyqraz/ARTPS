"""
Gerçek Rover Veri İşlemcisi
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RealDataProcessor:
    """Gerçek Mars rover verilerini işleyen sınıf"""
    
    def __init__(self, raw_data_dir: str, processed_dir: str = "data/real_mars_data/processed"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Alt dizinler
        (self.processed_dir / "images").mkdir(exist_ok=True)
        (self.processed_dir / "metadata").mkdir(exist_ok=True)
        
        # Desteklenen formatlar
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        
    def scan_raw_data(self) -> List[Path]:
        """Ham veri dizinini tarar"""
        logger.info(f"Ham veri dizini taranıyor: {self.raw_data_dir}")
        
        image_files = []
        for file_path in self.raw_data_dir.rglob('*'):
            if file_path.suffix.lower() in self.supported_formats:
                image_files.append(file_path)
        
        logger.info(f"Toplam {len(image_files)} görüntü dosyası bulundu")
        return image_files
    
    def preprocess_image(self, image_path: Path, target_size: Tuple[int, int] = (128, 128)) -> Optional[np.ndarray]:
        """Görüntüyü ön işler"""
        try:
            # Görüntüyü yükle
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Boyutlandır
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Normalize et
            image = image.astype(np.float32) / 255.0
            
            # BGR'den RGB'ye çevir
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
            
        except Exception as e:
            logger.error(f"Ön işleme hatası ({image_path}): {e}")
            return None
    
    def process_dataset(self) -> Dict:
        """Tüm veri setini işler"""
        logger.info("Veri seti işleme başlıyor...")
        
        # Ham verileri tara
        raw_images = self.scan_raw_data()
        
        if not raw_images:
            logger.error("Hiç görüntü bulunamadı!")
            return {}
        
        # Ön işleme
        processed_data = []
        
        for i, image_path in enumerate(raw_images):
            logger.info(f"İşleniyor: {i+1}/{len(raw_images)} - {image_path.name}")
            
            # Ön işleme
            processed_image = self.preprocess_image(image_path)
            if processed_image is None:
                continue
            
            # İşlenmiş görüntüyü kaydet
            processed_filename = f"processed_{i:04d}.npy"
            processed_path = self.processed_dir / "images" / processed_filename
            np.save(processed_path, processed_image)
            
            # Metadata kaydet
            metadata = {
                "original_path": str(image_path),
                "processed_path": str(processed_path),
                "processing_date": datetime.now().isoformat()
            }
            
            processed_data.append(metadata)
        
        # Sonuçları kaydet
        self._save_processing_results(processed_data)
        
        logger.info(f"İşleme tamamlandı: {len(processed_data)} görüntü işlendi")
        return {
            "total_processed": len(processed_data),
            "total_raw": len(raw_images)
        }
    
    def _save_processing_results(self, processed_data: List[Dict]):
        """İşleme sonuçlarını kaydeder"""
        # Metadata CSV
        df = pd.DataFrame(processed_data)
        df.to_csv(self.processed_dir / "metadata" / "processed_images.csv", index=False)
        
        logger.info("İşleme sonuçları kaydedildi")


def main():
    """Ana fonksiyon"""
    print("🚀 Gerçek Rover Veri İşlemcisi Başlıyor...")
    
    processor = RealDataProcessor(
        raw_data_dir="data/raw_rover_images",
        processed_dir="data/real_mars_data/processed"
    )
    
    results = processor.process_dataset()
    
    print("\n📊 İşleme Sonuçları:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main() 