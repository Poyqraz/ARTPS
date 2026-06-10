"""
Perseverance Verilerini Bul
"""

import requests
import json
from urllib.parse import urljoin
import time

def test_perseverance_specific_apis():
    """Perseverance'a özel API'leri test et"""
    
    print("🔍 Perseverance Özel API'leri Test Ediliyor...")
    
    # Perseverance'a özel URL'ler
    perseverance_urls = [
        "https://mars.nasa.gov/api/v1/raw_image_items/",
        "https://mars.nasa.gov/api/v1/raw_images/",
        "https://mars.nasa.gov/api/v1/images/",
        "https://mars.nasa.gov/api/v1/perseverance/",
        "https://mars.nasa.gov/api/v1/mars2020/",
        "https://mars.nasa.gov/api/v1/mastcamz/",
        "https://mars.nasa.gov/api/v1/rovers/perseverance/",
        "https://mars.nasa.gov/api/v1/rovers/mars2020/"
    ]
    
    # Farklı parametre kombinasyonları
    test_params = [
        {'mission': 'mars2020', 'limit': 5},
        {'mission': 'perseverance', 'limit': 5},
        {'rover': 'perseverance', 'limit': 5},
        {'rover': 'mars2020', 'limit': 5},
        {'instrument': 'mastcamz', 'limit': 5},
        {'camera': 'mastcamz', 'limit': 5},
        {'mission': 'mars2020', 'instrument': 'mastcamz', 'limit': 5},
        {'mission': 'mars2020', 'camera': 'mastcamz', 'limit': 5},
        {'rover': 'perseverance', 'instrument': 'mastcamz', 'limit': 5},
        {'rover': 'perseverance', 'camera': 'mastcamz', 'limit': 5},
        {'mission': 'mars2020', 'instrument': 'mastcamz', 'sol': 1, 'limit': 5},
        {'mission': 'mars2020', 'instrument': 'mastcamz', 'sol': 100, 'limit': 5},
        {'mission': 'mars2020', 'instrument': 'mastcamz', 'sol': 500, 'limit': 5},
        {'mission': 'mars2020', 'instrument': 'mastcamz', 'sol': 1000, 'limit': 5}
    ]
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    for url in perseverance_urls:
        print(f"\n🔗 Test URL: {url}")
        
        for i, params in enumerate(test_params):
            print(f"  📋 Test {i+1}: {params}")
            
            try:
                response = session.get(url, params=params, timeout=30)
                print(f"    Status: {response.status_code}")
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        
                        if isinstance(data, dict) and 'items' in data:
                            items = data['items']
                            print(f"    Items found: {len(items)}")
                            
                            # Perseverance verisi kontrolü
                            perseverance_found = False
                            for item in items[:3]:
                                mission = item.get('mission', '').lower()
                                instrument = item.get('instrument', '').lower()
                                title = item.get('title', '').lower()
                                
                                if any(keyword in mission or keyword in instrument or keyword in title 
                                       for keyword in ['mars2020', 'perseverance', 'mastcamz']):
                                    perseverance_found = True
                                    print(f"    ✅ Perseverance verisi bulundu!")
                                    print(f"      Mission: {item.get('mission')}")
                                    print(f"      Instrument: {item.get('instrument')}")
                                    print(f"      Title: {item.get('title')}")
                                    break
                            
                            if perseverance_found:
                                return url, params
                                
                    except json.JSONDecodeError:
                        print(f"    Not JSON response")
                        
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"    ❌ Hata: {e}")
    
    return None, None

def test_pds_alternative_urls():
    """PDS alternatif URL'lerini test et"""
    
    print("\n📁 PDS Alternatif URL'leri Test Ediliyor...")
    
    pds_urls = [
        "https://pds-imaging.jpl.nasa.gov/data/mars2020/",
        "https://pds-geosciences.wustl.edu/missions/mars2020/",
        "https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=MARS2020-MASTCAMZ-2-EDR-V1.0",
        "https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=MARS2020-MASTCAMZ-2-EDR-V1.0",
        "https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=MARS2020-MASTCAMZ-2-EDR-V1.0",
        "https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=MARS2020-MASTCAMZ-2-EDR-V1.0",
        "https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=MARS2020-MASTCAMZ-2-EDR-V1.0",
        "https://pds.nasa.gov/ds-view/pds/viewDataset.jsp?dsid=MARS2020-MASTCAMZ-2-EDR-V1.0"
    ]
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    for url in pds_urls:
        print(f"\n🔗 Test: {url}")
        
        try:
            response = session.get(url, timeout=30)
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                title = soup.find('title')
                title_text = title.get_text() if title else "Başlık yok"
                print(f"  Title: {title_text}")
                
                # Perseverance/Mastcam-Z içeriği ara
                content = soup.get_text().lower()
                if any(keyword in content for keyword in ['perseverance', 'mars2020', 'mastcamz']):
                    print(f"  ✅ Perseverance içeriği bulundu!")
                    
                    # Linkleri kontrol et
                    links = soup.find_all('a', href=True)
                    for link in links[:5]:
                        href = link['href']
                        text = link.get_text().strip()
                        if any(keyword in text.lower() for keyword in ['mastcamz', 'perseverance', 'mars2020']):
                            print(f"    - {text} -> {href}")
                    
                    return url
                    
        except Exception as e:
            print(f"  ❌ Hata: {e}")
    
    return None

def test_nasa_archive_explorer():
    """NASA Archive Explorer'ı test et"""
    
    print("\n🔍 NASA Archive Explorer Test Ediliyor...")
    
    archive_urls = [
        "https://pds-imaging.jpl.nasa.gov/beta/archive-explorer?mission=mars_2020&bundle=mars2020_mastcamz_sci_calibrated",
        "https://pds-imaging.jpl.nasa.gov/beta/archive-explorer?mission=mars_2020&bundle=mars2020_mastcamz_ops_raw",
        "https://pds-imaging.jpl.nasa.gov/beta/archive-explorer?mission=mars_2020&bundle=mars2020_mastcamz_ops_calibrated",
        "https://pds-imaging.jpl.nasa.gov/beta/archive-explorer?mission=mars_2020&bundle=mars2020_mastcamz_ops_stereo",
        "https://pds-imaging.jpl.nasa.gov/beta/archive-explorer?mission=mars_2020&bundle=mars2020_mastcamz_ops_mesh",
        "https://pds-imaging.jpl.nasa.gov/beta/archive-explorer?mission=mars_2020&bundle=mars2020_mastcamz_ops_mosaic"
    ]
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    for url in archive_urls:
        print(f"\n🔗 Test: {url}")
        
        try:
            response = session.get(url, timeout=30)
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                from bs4 import BeautifulSoup
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
                    
        except Exception as e:
            print(f"  ❌ Hata: {e}")
    
    return None

if __name__ == "__main__":
    print("🚀 Perseverance Veri Arama Başlıyor...")
    
    # 1. Perseverance özel API'leri test et
    api_url, api_params = test_perseverance_specific_apis()
    
    if api_url and api_params:
        print(f"\n✅ Perseverance API bulundu!")
        print(f"URL: {api_url}")
        print(f"Params: {api_params}")
    else:
        print(f"\n❌ Perseverance API bulunamadı")
    
    # 2. PDS alternatif URL'lerini test et
    pds_url = test_pds_alternative_urls()
    
    if pds_url:
        print(f"\n✅ PDS URL bulundu: {pds_url}")
    else:
        print(f"\n❌ PDS URL bulunamadı")
    
    # 3. NASA Archive Explorer'ı test et
    archive_url = test_nasa_archive_explorer()
    
    if archive_url:
        print(f"\n✅ Archive Explorer URL bulundu: {archive_url}")
    else:
        print(f"\n❌ Archive Explorer URL bulunamadı")
    
    print("\n✅ Arama tamamlandı!") 