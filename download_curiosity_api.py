"""
API'den Curiosity Verilerini İndir
"""

import requests
import json
import os
from pathlib import Path
import time
from urllib.parse import urljoin

def download_curiosity_images():
    """API'den Curiosity görüntülerini indir"""
    
    print("🔄 API'den Curiosity Verileri İndiriliyor...")
    
    base_url = "https://mars.nasa.gov/api/v1/raw_image_items/"
    output_dir = Path("data/curiosity_api_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    # Curiosity parametreleri
    params = {
        'mission': 'msl',  # Curiosity
        'limit': 50,       # 50 görüntü
        'page': 1
    }
    
    total_downloaded = 0
    
    try:
        print(f"API'ye istek gönderiliyor: {params}")
        response = session.get(base_url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            items = data.get('items', [])
            
            print(f"API'den {len(items)} Curiosity görüntüsü bulundu")
            
            for i, item in enumerate(items):
                try:
                    # Görüntü bilgilerini al
                    title = item.get('title', f'curiosity_{i}')
                    url = item.get('url', '')
                    
                    if not url:
                        continue
                    
                    # Dosya adını oluştur
                    filename = f"curiosity_{i:04d}_{title.replace(' ', '_').replace(':', '_')[:50]}.jpg"
                    filepath = output_dir / filename
                    
                    # Dosya zaten varsa atla
                    if filepath.exists():
                        print(f"  {i+1}. Zaten mevcut: {filename}")
                        total_downloaded += 1
                        continue
                    
                    print(f"  {i+1}. İndiriliyor: {title}")
                    
                    # Görüntüyü indir
                    img_response = session.get(url, timeout=60, stream=True)
                    img_response.raise_for_status()
                    
                    with open(filepath, 'wb') as f:
                        for chunk in img_response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    total_downloaded += 1
                    print(f"     ✅ İndirildi: {filename}")
                    
                    # Rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"  {i+1}. İndirme hatası: {e}")
                    continue
            
            print(f"\n✅ Toplam {total_downloaded} görüntü indirildi")
            print(f"📁 Kayıt dizini: {output_dir}")
            
            return total_downloaded
            
        else:
            print(f"❌ API hatası: {response.status_code}")
            return 0
            
    except Exception as e:
        print(f"❌ API isteği hatası: {e}")
        return 0

def analyze_curiosity_data():
    """İndirilen Curiosity verilerini analiz et"""
    
    print("\n📊 İndirilen Curiosity Verilerini Analiz Ediliyor...")
    
    output_dir = Path("data/curiosity_api_images")
    
    if not output_dir.exists():
        print("❌ İndirilen veri dizini bulunamadı")
        return
    
    # Dosyaları listele
    image_files = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
    
    print(f"Toplam indirilen görüntü: {len(image_files)}")
    
    if image_files:
        print("İlk 5 dosya:")
        for i, file in enumerate(image_files[:5]):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {i+1}. {file.name} ({size_mb:.1f} MB)")
    
    return len(image_files)

if __name__ == "__main__":
    print("🚀 Curiosity API İndirme Başlıyor...")
    
    # Curiosity görüntülerini indir
    downloaded_count = download_curiosity_images()
    
    # İndirilen verileri analiz et
    if downloaded_count > 0:
        analyze_curiosity_data()
    
    print("\n✅ Curiosity API indirme tamamlandı!")
    print("Bu verileri de model eğitiminde kullanabilirsiniz.") 