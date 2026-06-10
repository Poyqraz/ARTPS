"""
Perseverance Verilerini Ara
"""

import requests
import json
from bs4 import BeautifulSoup

def search_perseverance():
    """Perseverance verilerini ara"""
    
    print("🔍 Perseverance Verilerini Arıyor...")
    
    # 1. API'de farklı sol'ları test et
    base_url = "https://mars.nasa.gov/api/v1/raw_image_items/"
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    # Perseverance'ın bilinen sol'ları
    test_sols = [1, 100, 500, 1000, 1500, 2000]
    
    for sol in test_sols:
        params = {'mission': 'mars2020', 'sol': sol, 'limit': 5}
        
        try:
            response = session.get(base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('items', [])
                
                if items:
                    print(f"Sol {sol}: {len(items)} görüntü bulundu")
                    
                    first_item = items[0]
                    mission = first_item.get('mission', 'Unknown')
                    instrument = first_item.get('instrument', 'Unknown')
                    
                    print(f"  Mission: {mission}")
                    print(f"  Instrument: {instrument}")
                    
                    if 'mars2020' in mission.lower() or 'perseverance' in mission.lower():
                        print(f"  ✅ Perseverance verisi!")
                        return True
                        
        except Exception as e:
            print(f"Sol {sol}: Hata - {e}")
    
    # 2. PDS Archive Explorer test et
    archive_url = "https://pds-imaging.jpl.nasa.gov/beta/archive-explorer?mission=mars_2020&bundle=mars2020_mastcamz_sci_calibrated"
    
    try:
        response = session.get(archive_url, timeout=30)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            title = soup.find('title')
            title_text = title.get_text() if title else "Başlık yok"
            print(f"Archive Explorer: {title_text}")
            
            # Dosya linklerini ara
            links = soup.find_all('a', href=True)
            file_count = 0
            
            for link in links:
                text = link.get_text().strip()
                if any(ext in text.lower() for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']):
                    file_count += 1
            
            if file_count > 0:
                print(f"✅ {file_count} dosya bulundu!")
                return True
                
    except Exception as e:
        print(f"Archive Explorer hatası: {e}")
    
    return False

if __name__ == "__main__":
    print("🚀 Perseverance Arama Başlıyor...")
    
    found = search_perseverance()
    
    if found:
        print("\n✅ Perseverance verisi bulundu!")
    else:
        print("\n❌ Perseverance verisi bulunamadı")
        print("Curiosity verileriyle devam edebilirsiniz.")
    
    print("\n✅ Arama tamamlandı!") 