"""
Perseverance Verilerini Bul - Gelişmiş Yöntemler
"""

import requests
import json
from urllib.parse import urljoin
import time
from bs4 import BeautifulSoup

def test_perseverance_specific_sols():
    """Perseverance'ın bilinen sol'larını test et"""
    
    print("🔍 Perseverance Bilinen Sol'ları Test Ediliyor...")
    
    base_url = "https://mars.nasa.gov/api/v1/raw_image_items/"
    
    # Perseverance'ın bilinen sol aralıkları
    perseverance_sols = [
        1, 2, 3, 4, 5, 10, 15, 20, 25, 30,  # İlk günler
        50, 75, 100, 125, 150, 175, 200,    # Erken dönem
        300, 400, 500, 600, 700, 800,       # Orta dönem
        900, 1000, 1100, 1200, 1300, 1400,  # Geç dönem
        1500, 1600, 1700, 1800, 1900, 2000  # En son dönem
    ]
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    for sol in perseverance_sols:
        print(f"\n📋 Sol {sol} test ediliyor...")
        
        # Farklı parametre kombinasyonları
        test_params = [
            {'mission': 'mars2020', 'sol': sol, 'limit': 5},
            {'mission': 'perseverance', 'sol': sol, 'limit': 5},
            {'rover': 'perseverance', 'sol': sol, 'limit': 5},
            {'rover': 'mars2020', 'sol': sol, 'limit': 5},
            {'instrument': 'mastcamz', 'sol': sol, 'limit': 5},
            {'camera': 'mastcamz', 'sol': sol, 'limit': 5}
        ]
        
        for params in test_params:
            try:
                response = session.get(base_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    items = data.get('items', [])
                    
                    if items:
                        print(f"  {params}: {len(items)} görüntü bulundu")
                        
                        # İlk öğeyi kontrol et
                        first_item = items[0]
                        mission = first_item.get('mission', 'Unknown')
                        instrument = first_item.get('instrument', 'Unknown')
                        title = first_item.get('title', 'Unknown')
                        
                        print(f"    Mission: {mission}")
                        print(f"    Instrument: {instrument}")
                        print(f"    Title: {title}")
                        
                        # Perseverance verisi kontrolü
                        if any(keyword in mission.lower() or keyword in instrument.lower() or keyword in title.lower()
                               for keyword in ['mars2020', 'perseverance', 'mastcamz']):
                            print(f"    ✅ Perseverance verisi bulundu!")
                            return True
                    else:
                        print(f"  {params}: Görüntü bulunamadı")
                        
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"  {params}: Hata - {e}")
    
    return False

def test_pds_archive_explorer():
    """PDS Archive Explorer'ı detaylı test et"""
    
    print("\n🔍 PDS Archive Explorer Detaylı Test...")
    
    # Farklı bundle kombinasyonları
    bundles = [
        'mars2020_mastcamz_sci_calibrated',
        'mars2020_mastcamz_ops_raw',
        'mars2020_mastcamz_ops_calibrated',
        'mars2020_mastcamz_ops_stereo',
        'mars2020_mastcamz_ops_mesh',
        'mars2020_mastcamz_ops_mosaic',
        'mars2020_navcam_ops_raw',
        'mars2020_navcam_ops_calibrated',
        'mars2020_navcam_ops_stereo',
        'mars2020_hazcam_ops_raw',
        'mars2020_hazcam_ops_calibrated'
    ]
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    for bundle in bundles:
        url = f"https://pds-imaging.jpl.nasa.gov/beta/archive-explorer?mission=mars_2020&bundle={bundle}"
        
        print(f"\n🔗 Test: {bundle}")
        
        try:
            response = session.get(url, timeout=30)
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                title = soup.find('title')
                title_text = title.get_text() if title else "Başlık yok"
                print(f"  Title: {title_text}")
                
                # Dosya linklerini ara
                links = soup.find_all('a', href=True)
                file_links = []
                
                for link in links:
                    href = link['href']
                    text = link.get_text().strip()
                    if any(ext in text.lower() for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.xml', '.lbl']):
                        file_links.append((text, href))
                
                if file_links:
                    print(f"  ✅ {len(file_links)} dosya bulundu!")
                    for text, href in file_links[:3]:
                        print(f"    - {text}")
                    return url
                else:
                    print(f"  ❌ Dosya bulunamadı")
                    
        except Exception as e:
            print(f"  ❌ Hata: {e}")
    
    return None

def test_nasa_public_datasets():
    """NASA'nın public dataset'lerini test et"""
    
    print("\n🔍 NASA Public Dataset'leri Test Ediliyor...")
    
    # NASA'nın farklı veri kaynakları
    nasa_urls = [
        "https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=MARS2020-MASTCAMZ-2-EDR-V1.0",
        "https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=MARS2020-NAVCAM-2-EDR-V1.0",
        "https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=MARS2020-HAZCAM-2-EDR-V1.0",
        "https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=MARS2020-SUPERCAM-2-EDR-V1.0",
        "https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=MARS2020-PIXL-2-EDR-V1.0",
        "https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=MARS2020-SHERLOC-2-EDR-V1.0",
        "https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=MARS2020-MEDI-2-EDR-V1.0",
        "https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=MARS2020-MOXIE-2-EDR-V1.0"
    ]
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    for url in nasa_urls:
        print(f"\n🔗 Test: {url}")
        
        try:
            response = session.get(url, timeout=30)
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                title = soup.find('title')
                title_text = title.get_text() if title else "Başlık yok"
                print(f"  Title: {title_text}")
                
                # Perseverance içeriği kontrolü
                content = soup.get_text().lower()
                if any(keyword in content for keyword in ['perseverance', 'mars2020', 'mastcamz', 'mars 2020']):
                    print(f"  ✅ Perseverance içeriği bulundu!")
                    
                    # Linkleri kontrol et
                    links = soup.find_all('a', href=True)
                    for link in links[:5]:
                        href = link['href']
                        text = link.get_text().strip()
                        if any(keyword in text.lower() for keyword in ['mastcamz', 'perseverance', 'mars2020', 'download']):
                            print(f"    - {text} -> {href}")
                    
                    return url
                else:
                    print(f"  ❌ Perseverance içeriği bulunamadı")
                    
        except Exception as e:
            print(f"  ❌ Hata: {e}")
    
    return None

def test_alternative_apis():
    """Alternatif API'leri test et"""
    
    print("\n🔍 Alternatif API'ler Test Ediliyor...")
    
    # Farklı API endpoint'leri
    api_urls = [
        "https://mars.nasa.gov/api/v1/raw_image_items/",
        "https://mars.nasa.gov/api/v1/images/",
        "https://mars.nasa.gov/api/v1/rovers/",
        "https://mars.nasa.gov/api/v1/missions/",
        "https://mars.nasa.gov/api/v1/instruments/",
        "https://mars.nasa.gov/api/v1/cameras/"
    ]
    
    # Farklı parametre kombinasyonları
    test_params = [
        {'mission': 'mars2020', 'limit': 10},
        {'rover': 'perseverance', 'limit': 10},
        {'instrument': 'mastcamz', 'limit': 10},
        {'camera': 'mastcamz', 'limit': 10},
        {'mission': 'mars2020', 'instrument': 'mastcamz', 'limit': 10},
        {'rover': 'perseverance', 'instrument': 'mastcamz', 'limit': 10},
        {'mission': 'mars2020', 'camera': 'mastcamz', 'limit': 10},
        {'rover': 'perseverance', 'camera': 'mastcamz', 'limit': 10}
    ]
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    for url in api_urls:
        print(f"\n🔗 Test URL: {url}")
        
        for params in test_params:
            try:
                response = session.get(url, params=params, timeout=30)
                print(f"  {params}: Status {response.status_code}")
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        
                        if isinstance(data, dict) and 'items' in data:
                            items = data['items']
                            print(f"    Items found: {len(items)}")
                            
                            if items:
                                # İlk öğeyi kontrol et
                                first_item = items[0]
                                mission = first_item.get('mission', 'Unknown')
                                instrument = first_item.get('instrument', 'Unknown')
                                
                                print(f"    Mission: {mission}")
                                print(f"    Instrument: {instrument}")
                                
                                # Perseverance verisi kontrolü
                                if any(keyword in mission.lower() or keyword in instrument.lower()
                                       for keyword in ['mars2020', 'perseverance', 'mastcamz']):
                                    print(f"    ✅ Perseverance verisi bulundu!")
                                    return url, params
                                    
                    except json.JSONDecodeError:
                        print(f"    Not JSON response")
                        
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"  {params}: Hata - {e}")
    
    return None, None

if __name__ == "__main__":
    print("🚀 Perseverance Gelişmiş Arama Başlıyor...")
    
    # 1. Perseverance bilinen sol'larını test et
    perseverance_found = test_perseverance_specific_sols()
    
    if perseverance_found:
        print(f"\n✅ Perseverance verisi bulundu!")
    else:
        print(f"\n❌ Perseverance verisi bulunamadı")
    
    # 2. PDS Archive Explorer'ı test et
    archive_url = test_pds_archive_explorer()
    
    if archive_url:
        print(f"\n✅ Archive Explorer URL bulundu: {archive_url}")
    else:
        print(f"\n❌ Archive Explorer URL bulunamadı")
    
    # 3. NASA Public Dataset'lerini test et
    dataset_url = test_nasa_public_datasets()
    
    if dataset_url:
        print(f"\n✅ Dataset URL bulundu: {dataset_url}")
    else:
        print(f"\n❌ Dataset URL bulunamadı")
    
    # 4. Alternatif API'leri test et
    api_url, api_params = test_alternative_apis()
    
    if api_url and api_params:
        print(f"\n✅ Alternatif API bulundu!")
        print(f"URL: {api_url}")
        print(f"Params: {api_params}")
    else:
        print(f"\n❌ Alternatif API bulunamadı")
    
    print("\n✅ Gelişmiş arama tamamlandı!") 