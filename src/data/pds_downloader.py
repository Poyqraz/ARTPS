"""
NASA PDS Mars Rover Veri İndirici
"""

import os
import requests
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import logging
from urllib.parse import urljoin
import io # StringIO için gerekli

# PDS4 Tools kütüphanesini import ediyoruz (eğer kuruluysa)
try:
    from pds4_tools import pds4_read, pds4_to_bytes, pds4_info
except ImportError:
    pass 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PDSDownloader:
    """NASA PDS'den Mars rover verilerini indiren sınıf"""

    def __init__(self, base_dir: str = "data/real_mars_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # NASA PDS Imaging Node temel URL'i
        self.pds_imaging_node_base = "https://pds-imaging.jpl.nasa.gov/"
        
        # Perseverance Mastcam-Z Ham Görüntü Bundle URL'si ve Envanter CSV URL'i
        self.mastcamz_raw_bundle_url = urljoin(self.pds_imaging_node_base, "data/mars2020/mars2020_mastcamz/data_raw/")
        self.mastcamz_raw_inventory_url = urljoin(self.mastcamz_raw_bundle_url, "collection_data_raw_inventory.csv")

        self.rovers = {
            'perseverance': {
                'mission': 'Mars 2020',
                'instruments': ['Mastcam-Z', 'SuperCam', 'PIXL'],
                'data_dir': self.base_dir / 'perseverance',
                'image_raw_dir': self.base_dir / 'perseverance' / 'images' / 'mastcam_z_raw'
            },
            'curiosity': {
                'mission': 'MSL',
                'instruments': ['Mastcam', 'ChemCam'],
                'data_dir': self.base_dir / 'curiosity',
                'image_raw_dir': self.base_dir / 'curiosity' / 'images' / 'mastcam_raw' 
            }
        }

        # HTTP oturumu (User-Agent ile)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ARTPS/1.0 (+https://example.org)",
            "Accept": "*/*",
        })

        self._create_directory_structure()

    def _create_directory_structure(self):
        """Veri dizin yapısını oluştur"""
        for rover_name, rover_info in self.rovers.items():
            rover_dir = rover_info['data_dir']
            
            (rover_dir / 'images' / 'interesting').mkdir(parents=True, exist_ok=True)
            (rover_dir / 'images' / 'ordinary').mkdir(parents=True, exist_ok=True)
            rover_info['image_raw_dir'].mkdir(parents=True, exist_ok=True) 
            
            (rover_dir / 'science_data').mkdir(parents=True, exist_ok=True)
            (rover_dir / 'metadata').mkdir(parents=True, exist_ok=True)
            
            for instrument in rover_info['instruments']:
                (rover_dir / 'science_data' / instrument.lower()).mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, local_path: Path) -> bool:
        """Tek bir dosyayı indirir."""
        if local_path.exists():
            logger.info(f"Dosya zaten mevcut, atlanıyor: {local_path.name}")
            return True 
            
        try:
            logger.info(f"İndiriliyor: {url} -> {local_path.name}")
            response = self.session.get(url, stream=True, timeout=120, verify=False)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"İndirme başarılı: {local_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"İndirme hatası ({url}): {e}")
            if local_path.exists():
                local_path.unlink()
            return False

    def download_mastcamz_inventory(self) -> Optional[pd.DataFrame]:
        """Mastcam-Z ham görüntü envanter CSV'sini indirir ve ayrıştırır."""
        logger.info(f"Mastcam-Z ham görüntü envanteri indiriliyor: {self.mastcamz_raw_inventory_url}")
        
        try:
            response = self.session.get(self.mastcamz_raw_inventory_url, timeout=60, verify=False) # verify=False eklendi
            response.raise_for_status()
            
            # İndirilen CSV içeriğini pandas DataFrame'e oku
            # PDS inventory CSV'leri genellikle özel formatlara sahiptir (örn. dikey çizgi ile ayrılmış, yorum satırları)
            # `inspect_csv.py` çıktısına göre, HTML döndüğü için bu metodun direkt pandas ile okunması hataya neden oldu.
            # Eğer inventory CSV URL'si doğrudan bir CSV dosyası değilse, bu metod başarısız olacaktır.
            # Bu durumda, PDS'den doğrudan erişilebilir inventory CSV URL'sini bulmamız gerekir.
            # Geçici olarak, response.text içinde 'product_file_path' olup olmadığını kontrol ederek ilerleyelim.

            # Eğer HTML sayfası dönüyorsa, `read_csv` hata verecektir. 
            # Bu durumda, loglara bir uyarı düşüp None dönmeliyiz.
            if "<!DOCTYPE HTML" in response.text or "<html" in response.text:
                logger.warning(f"Envanter URL'si {self.mastcamz_raw_inventory_url} bir HTML sayfası döndü, CSV değil.")
                return None
            
            # Gerçek CSV formatına göre oku
            # `inspect_csv.py` çıktılarına göre, `skiprows` ve `sep` değerleri burada hassas olmalı.
            # Önceki denemelerden, `sep='|'` ve `skiprows`un dikkatli ayarlanması gerektiği biliniyor.
            # Ancak HTML döndüğü için bu kısım gerçekte çalışmadı.
            # Doğru CSV formatına sahip bir envanter dosyası bulana kadar bu kısım sorunlu kalacaktır.
            inventory_df = pd.read_csv(io.StringIO(response.text),
                                       sep='|',
                                       comment='#',
                                       header=None,
                                       skipinitialspace=True,
                                       skiprows=5) 
            
            inventory_df.columns = [
                'member_status', 
                'logical_identifier', 
                'version_id', 
                'product_file_path',
                'file_name', 
                'file_checksum', 
                'file_creation_date_time'
            ]
            logger.info(f"Envanterde toplam {len(inventory_df)} ürün bulundu.")
            return inventory_df

        except Exception as e:
            logger.error(f"Mastcam-Z envanter CSV indirme/ayrıştırma hatası: {e}")
            return None

    def download_mastcamz_raw_images(self, sol_range: Tuple[int, int] = (1, 10), max_images_per_sol: int = 5) -> int:
        """Perseverance Mastcam-Z ham görüntülerini belirtilen sol aralığında indirir."""
        logger.info(f"Perseverance Mastcam-Z ham görüntüleri indiriliyor (Sol {sol_range[0]}-{sol_range[1]})...")
        
        inventory_df = self.download_mastcamz_inventory()
        if inventory_df is None:
            logger.warning("Envanter indirilemedi. Dizin listeleme ile yedek indirime geçiliyor.")
            return self._fallback_download_by_listing(sol_range, max_images_per_sol)

        downloaded_count = 0
        rover_info = self.rovers['perseverance']
        base_save_dir = rover_info['image_raw_dir']

        # Sadece .IMG uzantılı ham görüntüleri filtrele
        filtered_images = inventory_df[
            (inventory_df['file_name'].str.lower().str.endswith('.img', na=False)) 
        ].copy()
        
        # Sol numarasına göre filtrele (logical_identifier içinde sol bilgisi bulunur)
        # Örnek logical_identifier: urn:nasa:pds:mars2020.mastcamz:data_raw:sol:sol_00089
        filtered_images['sol'] = filtered_images['logical_identifier'].apply(lambda x: int(x.split(':')[-1].replace('sol_', '')))
        filtered_images = filtered_images[(filtered_images['sol'] >= sol_range[0]) & (filtered_images['sol'] <= sol_range[1])]

        logger.info(f"Filtrelenen Mastcam-Z ham görüntü sayısı (Sol {sol_range[0]}-{sol_range[1]}): {len(filtered_images)}")
        
        if filtered_images.empty:
            logger.info(f"Belirtilen sol aralığında (Sol {sol_range[0]}-{sol_range[1]}) hiç Mastcam-Z ham görüntü bulunamadı.")
            return 0

        for sol in range(sol_range[0], sol_range[1] + 1):
            sol_str = f"sol_{sol:05d}"
            current_sol_images = filtered_images[filtered_images['sol'] == sol]
            
            if current_sol_images.empty:
                logger.info(f"Sol {sol} için filtrelenmiş görüntü bulunamadı.")
                continue

            img_count_in_sol = 0
            # Her sol için maksimum `max_images_per_sol` kadar görüntü al
            for idx, row in current_sol_images.head(max_images_per_sol).iterrows():
                file_name = row['file_name']
                product_file_path = row['product_file_path'] # Bundle içindeki göreceli yol

                # PDS4 standardına göre doğrudan indirilebilir URL'yi oluştur.
                # product_file_path, bundle'ın kökünden itibaren göreceli bir yol olmalıdır.
                # Örnek product_file_path: data/sol/sol_00089/r_f2560_0716653303_000rtx_n02800001947735050005075j01.img
                file_url = urljoin(self.mastcamz_raw_bundle_url, product_file_path)

                local_sol_dir = base_save_dir / sol_str
                local_sol_dir.mkdir(parents=True, exist_ok=True)
                local_path = local_sol_dir / file_name

                if self.download_file(file_url, local_path):
                    downloaded_count += 1
                    img_count_in_sol += 1

                # İlgili .xml etiket dosyasını da indir
                xml_file_name = Path(file_name).stem + '.xml'
                # product_file_path'den xml yolu oluştur
                xml_product_file_path = str(Path(product_file_path).parent / xml_file_name)
                xml_url = urljoin(self.mastcamz_raw_bundle_url, xml_product_file_path)
                xml_local_path = local_sol_dir / xml_file_name
                self.download_file(xml_url, xml_local_path) 

                time.sleep(0.5) 
                
        logger.info(f"Toplam {downloaded_count} Mastcam-Z ham dosya indirildi.")
        return downloaded_count

    def _fallback_download_by_listing(self, sol_range: Tuple[int, int], max_images_per_sol: int) -> int:
        """Envanter yoksa, sol dizinlerini listeleyerek .img ve .xml indir.

        Not: PDS dizin yapısı tipik olarak .../data_raw/sol/sol_00050/ altında dosyalar içerir.
        """
        downloaded_count = 0
        rover_info = self.rovers['perseverance']
        base_save_dir = rover_info['image_raw_dir']

        for sol in range(sol_range[0], sol_range[1] + 1):
            sol_str = f"sol_{sol:05d}"
            sol_url = urljoin(self.mastcamz_raw_bundle_url, f"sol/{sol_str}/")
            logger.info(f"Listeleme: {sol_url}")
            try:
                r = self.session.get(sol_url, timeout=60, verify=False)
                r.raise_for_status()
                html = r.text
            except Exception as e:
                logger.warning(f"Sol dizini okunamadı ({sol_url}): {e}")
                continue

            # Basit link çıkartımı
            import re
            links = re.findall(r'href=["\']([^"\']+)["\']', html)
            # .img dosyalarını seç
            img_files = [ln for ln in links if ln.lower().endswith('.img')]
            if not img_files:
                logger.info(f"Sol {sol} için .img bulunamadı.")
                continue
            local_sol_dir = base_save_dir / sol_str
            local_sol_dir.mkdir(parents=True, exist_ok=True)
            count = 0
            for fname in img_files:
                if count >= max_images_per_sol:
                    break
                # Tam URL
                if not fname.lower().startswith('http'):
                    file_url = urljoin(sol_url, fname)
                else:
                    file_url = fname
                local_path = local_sol_dir / Path(fname).name
                if self.download_file(file_url, local_path):
                    downloaded_count += 1
                    count += 1
                # XML eş dosyası
                xml_name = f"{Path(fname).stem}.xml"
                xml_url = urljoin(sol_url, xml_name)
                xml_local = local_sol_dir / xml_name
                self.download_file(xml_url, xml_local)

        logger.info(f"Fallback ile indirilen toplam .img sayısı: {downloaded_count}")
        return downloaded_count

    def create_target_database(self, rover: str) -> pd.DataFrame:
        """Hedef veritabanı oluştur (simülasyon)."""
        logger.info(f"{rover.upper()} hedef veritabanı oluşturuluyor...")
        
        if rover == 'perseverance':
            targets = [
                {'sol': 10, 'target_name': 'Rochette', 'instrument': 'SuperCam', 
                 'image_file': 'perseverance_mastcam-z_001.jpg', 'science_value': 0.9},
                {'sol': 15, 'target_name': 'Bellegarde', 'instrument': 'SuperCam',
                 'image_file': 'perseverance_mastcam-z_002.jpg', 'science_value': 0.8},
                {'sol': 20, 'target_name': 'Montdenier', 'instrument': 'PIXL',
                 'image_file': 'perseverance_mastcam-z_003.jpg', 'science_value': 0.7}
            ]
        else:
            targets = [
                {'sol': 5, 'target_name': 'Jake_Matijevic', 'instrument': 'ChemCam',
                 'image_file': 'curiosity_mastcam_001.jpg', 'science_value': 0.9},
                {'sol': 12, 'target_name': 'Coronation', 'instrument': 'ChemCam',
                 'image_file': 'curiosity_mastcam_002.jpg', 'science_value': 0.8},
                {'sol': 18, 'target_name': 'Rocknest', 'instrument': 'ChemCam',
                 'image_file': 'curiosity_mastcam_003.jpg', 'science_value': 0.7}
            ]
        
        df = pd.DataFrame(targets)
        
        save_path = self.rovers[rover]['data_dir'] / 'metadata' / 'targets.csv'
        df.to_csv(save_path, index=False)
        
        logger.info(f"Hedef veritabanı kaydedildi: {save_path}")
        return df

    def download_all_data(self, sol_range: Tuple[int, int] = (1, 10)) -> Dict[str, int]:
        """Tüm önemli verileri indirir (şimdilik sadece Perseverance Mastcam-Z raw)."""
        results = {}
        
        mastcamz_raw_count = self.download_mastcamz_raw_images(sol_range)
        results['perseverance_mastcamz_raw_images'] = mastcamz_raw_count
        
        self.create_target_database('perseverance')
        self.create_target_database('curiosity') 

        return results


def main():
    """Ana fonksiyon - veri indirme işlemini başlatır."""
    import argparse
    print("🚀 NASA PDS Mars Rover Veri İndirici Başlıyor...\n")

    parser = argparse.ArgumentParser()
    parser.add_argument('--sol_start', type=int, default=80)
    parser.add_argument('--sol_end', type=int, default=90)
    parser.add_argument('--max_per_sol', type=int, default=5)
    args = parser.parse_args()

    downloader = PDSDownloader()
    sol_range = (int(args.sol_start), int(args.sol_end))

    print(f"Mars Sol Aralığı: {sol_range[0]}-{sol_range[1]}")
    print("Bu işlem ağ hızınıza ve seçilen Sol aralığına bağlı olarak zaman alabilir...\n")

    try:
        count = downloader.download_mastcamz_raw_images(sol_range, max_images_per_sol=int(args.max_per_sol))
        print(f"\n📊 İndirilen ham dosya sayısı: {count}")
        print(f"\n✅ Veri indirme tamamlandı!")
        print(f"📁 Veriler şu dizinde: {downloader.base_dir}")

    except Exception as e:
        logger.error(f"❌ Hata oluştu: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 