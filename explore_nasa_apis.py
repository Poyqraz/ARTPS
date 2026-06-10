"""
NASA API'lerini Keşfet
"""

import requests
import json
from urllib.parse import urljoin

def test_nasa_apis():
    """NASA'nın farklı API'lerini test eder"""
    
    print("🚀 NASA API'leri Test Ediliyor...")
    
    # Test edilecek API'ler
    apis = [
        {
            'name': 'Mars 2020 Public API',
            'url': 'https://mars.nasa.gov/api/v1/raw_image_items/',
            'params': {'mission': 'mars2020', 'limit': 10}
        },
        {
            'name': 'PDS Search API',
            'url': 'https://pds.nasa.gov/api/search/',
            'params': {'q': 'mission:mars2020', 'limit': 10}
        },
        {
            'name': 'PDS Imaging Node',
            'url': 'https://pds-imaging.jpl.nasa.gov/api/',
            'params': {}
        }
    ]
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    for api in apis:
        print(f"\n🔗 Test: {api['name']}")
        print(f"URL: {api['url']}")
        
        try:
            response = session.get(api['url'], params=api['params'], timeout=30)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"Response type: JSON")
                    print(f"Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                    
                    # İlk birkaç öğeyi göster
                    if isinstance(data, dict) and 'items' in data:
                        items = data['items'][:3]
                        for i, item in enumerate(items):
                            print(f"  Item {i+1}: {item}")
                    elif isinstance(data, list):
                        for i, item in enumerate(data[:3]):
                            print(f"  Item {i+1}: {item}")
                            
                except json.JSONDecodeError:
                    print(f"Response type: Text (first 200 chars)")
                    print(f"Content: {response.text[:200]}...")
            else:
                print(f"Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Hata: {e}")

def test_pds_directories():
    """PDS dizinlerini doğrudan test et"""
    
    print("\n📁 PDS Dizinleri Test Ediliyor...")
    
    # Farklı PDS URL'leri
    pds_urls = [
        "https://pds-imaging.jpl.nasa.gov/data/mars2020/",
        "https://pds-geosciences.wustl.edu/missions/mars2020/",
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
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                # HTML içeriğini kontrol et
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                title = soup.find('title')
                title_text = title.get_text() if title else "Başlık yok"
                print(f"Title: {title_text}")
                
                # Linkleri kontrol et
                links = soup.find_all('a', href=True)
                print(f"Total links: {len(links)}")
                
                # İlk 5 linki göster
                for i, link in enumerate(links[:5]):
                    href = link['href']
                    text = link.get_text().strip()
                    print(f"  {i+1}. {text} -> {href}")
                    
        except Exception as e:
            print(f"❌ Hata: {e}")

def test_mars_public_api():
    """Mars public API'yi detaylı test et"""
    
    print("\n🔍 Mars Public API Detaylı Test...")
    
    base_url = "https://mars.nasa.gov/api/v1/raw_image_items/"
    
    # Farklı parametreler
    test_params = [
        {'mission': 'mars2020', 'limit': 5},
        {'mission': 'msl', 'limit': 5},
        {'limit': 5},
        {'mission': 'mars2020', 'camera': 'mastcamz', 'limit': 5},
        {'mission': 'mars2020', 'sol': 1, 'limit': 5}
    ]
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    for i, params in enumerate(test_params):
        print(f"\n📋 Test {i+1}: {params}")
        
        try:
            response = session.get(base_url, params=params, timeout=30)
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                    
                    if isinstance(data, dict):
                        if 'items' in data:
                            items = data['items']
                            print(f"Items found: {len(items)}")
                            
                            for j, item in enumerate(items[:2]):
                                print(f"  Item {j+1}:")
                                for key, value in item.items():
                                    print(f"    {key}: {value}")
                        else:
                            print(f"Data: {data}")
                            
                except json.JSONDecodeError:
                    print(f"Not JSON: {response.text[:200]}...")
            else:
                print(f"Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Hata: {e}")

if __name__ == "__main__":
    print("🚀 NASA Veri Kaynakları Keşfi Başlıyor...")
    
    # API'leri test et
    test_nasa_apis()
    
    # PDS dizinlerini test et
    test_pds_directories()
    
    # Mars public API'yi detaylı test et
    test_mars_public_api()
    
    print("\n✅ Keşif tamamlandı!") 