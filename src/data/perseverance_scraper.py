"""
Perseverance Mastcam-Z Web Scraper
"""

import requests
from bs4 import BeautifulSoup
import os
import time
from pathlib import Path
from typing import List, Dict, Optional
import logging
from urllib.parse import urljoin
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PerseveranceScraper:
    """Perseverance Mastcam-Z verilerini indiren scraper"""
    
    def __init__(self, base_url: str = "https://pds-imaging.jpl.nasa.gov/data/mars2020/mars2020_mastcamz/data_edr/", 
                 output_dir: str = "data/real_mars_data/raw_images"):
        self.base_url = base_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Session oluştur
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # İstatistikler
        self.stats = {
            'total_sols_checked': 0,
            'total_files_found': 0,
            'total_files_downloaded': 0,
            'failed_downloads': 0,
            'start_time': datetime.now()
        }
        
        # Desteklenen formatlar
        self.supported_formats = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.xml', '.lbl'}
        
    def get_sol_url(self, sol_number: int) -> str:
        """Sol numarasından URL oluşturur"""
        sol_str = f"{sol_number:05d}"
        return urljoin(self.base_url, f"sol/{sol_str}/")
    
    def fetch_page_content(self, url: str) -> Optional[BeautifulSoup]:
        """Web sayfasının içeriğini getirir"""
        try:
            logger.info(f"Sayfa getiriliyor: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            return soup
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Sayfa getirme hatası ({url}): {e}")
            return None
    
    def extract_file_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict]:
        """Sayfadaki dosya linklerini çıkarır"""
        file_links = []
        
        links = soup.find_all('a', href=True)
        
        for link in links:
            href = link['href']
            filename = link.get_text().strip()
            
            file_ext = Path(filename).suffix.lower()
            if file_ext in self.supported_formats:
                full_url = urljoin(base_url, href)
                
                file_info = {
                    'filename': filename,
                    'url': full_url,
                    'extension': file_ext
                }
                
                file_links.append(file_info)
        
        return file_links
    
    def download_file(self, file_info: Dict, sol_dir: Path) -> bool:
        """Dosyayı indirir"""
        try:
            url = file_info['url']
            filename = file_info['filename']
            file_path = sol_dir / filename
            
            if file_path.exists():
                return True
            
            logger.info(f"İndiriliyor: {filename}")
            
            response = self.session.get(url, timeout=60, stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            self.stats['total_files_downloaded'] += 1
            return True
            
        except Exception as e:
            logger.error(f"İndirme hatası ({filename}): {e}")
            self.stats['failed_downloads'] += 1
            return False
    
    def process_sol(self, sol_number: int) -> Dict:
        """Tek bir sol'u işler"""
        logger.info(f"🔍 Sol {sol_number} işleniyor...")
        
        sol_url = self.get_sol_url(sol_number)
        soup = self.fetch_page_content(sol_url)
        
        if soup is None:
            return {'sol': sol_number, 'status': 'failed', 'files_found': 0, 'files_downloaded': 0}
        
        file_links = self.extract_file_links(soup, sol_url)
        
        if not file_links:
            return {'sol': sol_number, 'status': 'no_files', 'files_found': 0, 'files_downloaded': 0}
        
        sol_dir = self.output_dir / f"sol_{sol_number:05d}"
        sol_dir.mkdir(exist_ok=True)
        
        downloaded_count = 0
        for file_info in file_links:
            if self.download_file(file_info, sol_dir):
                downloaded_count += 1
        
        self.stats['total_files_found'] += len(file_links)
        
        return {
            'sol': sol_number,
            'status': 'success',
            'files_found': len(file_links),
            'files_downloaded': downloaded_count
        }
    
    def scrape_sol_range(self, start_sol: int, end_sol: int, delay: float = 1.0) -> Dict:
        """Sol aralığını tarar"""
        logger.info(f"🚀 Sol aralığı: {start_sol} - {end_sol}")
        
        results = []
        
        for sol in range(start_sol, end_sol + 1):
            self.stats['total_sols_checked'] += 1
            result = self.process_sol(sol)
            results.append(result)
            
            if delay > 0:
                time.sleep(delay)
        
        self.stats['end_time'] = datetime.now()
        duration = self.stats['end_time'] - self.stats['start_time']
        self.stats['duration'] = str(duration)
        
        self.save_results(results)
        
        return {'results': results, 'stats': self.stats}
    
    def save_results(self, results: List[Dict]):
        """Sonuçları kaydeder"""
        output_file = self.output_dir / "scraping_results.json"
        
        # datetime objelerini string'e çevir
        stats_copy = self.stats.copy()
        if 'start_time' in stats_copy:
            stats_copy['start_time'] = stats_copy['start_time'].isoformat()
        if 'end_time' in stats_copy:
            stats_copy['end_time'] = stats_copy['end_time'].isoformat()
        
        summary = {
            'scraping_date': datetime.now().isoformat(),
            'statistics': stats_copy,
            'results': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Sonuçlar kaydedildi: {output_file}")
    
    def print_summary(self):
        """Özet yazdırır"""
        print("\n" + "="*60)
        print("📊 İNDİRME ÖZETİ")
        print("="*60)
        print(f"Kontrol edilen Sol: {self.stats['total_sols_checked']}")
        print(f"Bulunan dosya: {self.stats['total_files_found']}")
        print(f"İndirilen dosya: {self.stats['total_files_downloaded']}")
        print(f"Başarısız: {self.stats['failed_downloads']}")
        
        if 'duration' in self.stats:
            print(f"Süre: {self.stats['duration']}")
        
        success_rate = (self.stats['total_files_downloaded'] / max(self.stats['total_files_found'], 1)) * 100
        print(f"Başarı oranı: {success_rate:.1f}%")
        print("="*60)


def main():
    """Ana fonksiyon"""
    print("🚀 Perseverance Mastcam-Z Web Scraper Başlıyor...")
    
    scraper = PerseveranceScraper(
        base_url="https://pds-imaging.jpl.nasa.gov/data/mars2020/mars2020_mastcamz/data_edr/",
        output_dir="data/real_mars_data/raw_images"
    )
    
    # Test: Sol 1-10
    print("Test modu: Sol 1-10 arası taranıyor...")
    results = scraper.scrape_sol_range(
        start_sol=1,
        end_sol=10,
        delay=2.0
    )
    
    scraper.print_summary()
    print("\n✅ İndirme tamamlandı!")


if __name__ == "__main__":
    main() 