"""
NASA Public Mars Rover Veri İndirici
"""

import os
import requests
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import logging
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import re
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NASAPublicDownloader:
    """NASA'nın public web sitesinden Mars rover verilerini indiren sınıf"""

    def __init__(self, base_dir: str = "data/real_mars_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # NASA Public Mars 2020 Raw Images URL'leri
        self.mars2020_raw_images_url = "https://mars.nasa.gov/mars2020/multimedia/raw-images/"
        self.mars2020_api_url = "https://mars.nasa.gov/api/v1/raw_image_items/"
        
        # Curiosity Raw Images URL'leri
        self.curiosity_raw_images_url = "https://mars.nasa.gov/msl/multimedia/raw-images/"
        self.curiosity_api_url = "https://mars.nasa.gov/api/v1/raw_image_items/"

        self.rovers = {
            'perseverance': {
                'mission': 'Mars 2020',
                'instruments': ['Mastcam-Z', 'SuperCam', 'PIXL'],
                'data_dir': self.base_dir / 'perseverance',
                'image_raw_dir': self.base_dir / 'perseverance' / 'images' / 'mastcam_z_raw',
                'web_url': self.mars2020_raw_images_url,
                'api_url': self.mars2020_api_url
            },
            'curiosity': {
                'mission': 'MSL',
                'instruments': ['Mastcam', 'ChemCam'],
                'data_dir': self.base_dir / 'curiosity',
                'image_raw_dir': self.base_dir / 'curiosity' / 'images' / 'mastcam_raw',
                'web_url': self.curiosity_raw_images_url,
                'api_url': self.curiosity_api_url
            }
        }

        self._create_directory_structure()

    def _create_directory_structure(self):
        """Veri dizin yapısını oluştur"""
        for rover_name, rover_info in self.rovers.items():
            rover_info['data_dir'].mkdir(parents=True, exist_ok=True)
            rover_info['image_raw_dir'].mkdir(parents=True, exist_ok=True)
            
            # Alt dizinler
            (rover_info['data_dir'] / 'metadata').mkdir(exist_ok=True)
            (rover_info['data_dir'] / 'logs').mkdir(exist_ok=True)

    def get_mars2020_api_data(self, sol: int, instrument: str = "Mastcam-Z") -> Optional[List[Dict]]:
        """Mars 2020 API'sinden belirli bir Sol için görüntü verilerini alır"""
        
        api_url = f"{self.mars2020_api_url}?order=sol+desc&search=&page=0&per_page=50&mission=mars2020&sol={sol}"
        
        try:
            logger.info(f"Mars 2020 API'den Sol {sol} verileri alınıyor: {api_url}")
            response = requests.get(api_url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'items' in data:
                # Sadece belirtilen enstrümanın görüntülerini filtrele
                filtered_items = []
                for item in data['items']:
                    if instrument.lower() in item.get('instrument', '').lower():
                        filtered_items.append(item)
                
                logger.info(f"Sol {sol} için {len(filtered_items)} {instrument} görüntüsü bulundu")
                return filtered_items
            else:
                logger.warning(f"Sol {sol} için API yanıtında 'items' alanı bulunamadı")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Mars 2020 API isteği hatası (Sol {sol}): {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Mars 2020 API JSON ayrıştırma hatası (Sol {sol}): {e}")
            return None

    def download_mars2020_images(self, sol_range: Tuple[int, int] = (1, 10), 
                                max_images_per_sol: int = 5) -> int:
        """Mars 2020 Perseverance görüntülerini indirir"""
        
        logger.info(f"Mars 2020 Perseverance görüntüleri indiriliyor (Sol {sol_range[0]}-{sol_range[1]})...")
        
        total_downloaded = 0
        
        for sol in range(sol_range[0], sol_range[1] + 1):
            logger.info(f"Sol {sol} işleniyor...")
            
            # API'den görüntü verilerini al
            images_data = self.get_mars2020_api_data(sol, "Mastcam-Z")
            
            if not images_data:
                logger.warning(f"Sol {sol} için görüntü verisi bulunamadı")
                continue
            
            # Sol dizinini oluştur
            sol_dir = self.rovers['perseverance']['image_raw_dir'] / f"sol_{sol:05d}"
            sol_dir.mkdir(exist_ok=True)
            
            # Görüntüleri indir
            downloaded_count = 0
            for image_data in images_data[:max_images_per_sol]:
                try:
                    # Görüntü URL'sini al
                    image_url = image_data.get('image_files', {}).get('full_res', '')
                    if not image_url:
                        continue
                    
                    # Dosya adını oluştur
                    image_id = image_data.get('image_id', f"unknown_{downloaded_count}")
                    file_extension = Path(urlparse(image_url).path).suffix or '.jpg'
                    filename = f"{image_id}{file_extension}"
                    filepath = sol_dir / filename
                    
                    # Dosya zaten varsa atla
                    if filepath.exists():
                        logger.debug(f"Dosya zaten mevcut: {filename}")
                        downloaded_count += 1
                        continue
                    
                    # Görüntüyü indir
                    logger.info(f"İndiriliyor: {filename}")
                    response = requests.get(image_url, timeout=60, stream=True)
                    response.raise_for_status()
                    
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Metadata'yı kaydet
                    metadata_file = sol_dir / f"{image_id}_metadata.json"
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(image_data, f, indent=2, ensure_ascii=False)
                    
                    downloaded_count += 1
                    total_downloaded += 1
                    
                    # Rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Görüntü indirme hatası (Sol {sol}, {image_id}): {e}")
                    continue
            
            logger.info(f"Sol {sol} tamamlandı: {downloaded_count} görüntü indirildi")
            
            # Sol'lar arası bekleme
            time.sleep(1)
        
        logger.info(f"Mars 2020 indirme tamamlandı: Toplam {total_downloaded} görüntü")
        return total_downloaded

    def download_all_data(self, sol_range: Tuple[int, int] = (1, 10)) -> Dict[str, int]:
        """Tüm rover verilerini indirir"""
        
        results = {}
        
        # Perseverance verilerini indir
        logger.info("🚀 Perseverance verileri indiriliyor...")
        perseverance_count = self.download_mars2020_images(sol_range)
        results['perseverance_images'] = perseverance_count
        
        # Curiosity verilerini indir (gelecekte eklenecek)
        logger.info("🔍 Curiosity verileri henüz desteklenmiyor...")
        results['curiosity_images'] = 0
        
        return results


def main():
    """Ana fonksiyon - veri indirme işlemini başlatır."""
    print("🚀 NASA Public Mars Rover Veri İndirici Başlıyor...\n")
    
    downloader = NASAPublicDownloader()
    
    # Gerçek veri indirme için başlangıç ve bitiş Sol'lerini belirle
    sol_start = 80 
    sol_end = 85  # Daha küçük aralık ile test
    sol_range = (sol_start, sol_end)
    
    print(f"Mars Sol Aralığı: {sol_range[0]}-{sol_range[1]}")
    print("Bu işlem ağ hızınıza ve seçilen Sol aralığına bağlı olarak zaman alabilir...\n")
    
    try:
        results = downloader.download_all_data(sol_range)
        
        print("\n📊 İndirme Sonuçları:")
        for key, count in results.items():
            print(f"  {key}: {count} dosya")
        
        print(f"\n✅ Veri indirme tamamlandı!")
        print(f"📁 Veriler şu dizinde: {downloader.base_dir}")
        
    except Exception as e:
        logger.error(f"❌ Hata oluştu: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 