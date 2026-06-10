"""
Doğru URL Yapısını Bul
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def find_correct_url():
    """Doğru URL yapısını bulur"""
    
    print("🔍 Doğru URL Yapısını Arıyor...")
    
    # Ana URL'ler
    base_urls = [
        "https://pds-imaging.jpl.nasa.gov/data/mars2020/mars2020_mastcamz/data_edr/",
        "https://pds-imaging.jpl.nasa.gov/data/mars2020/mars2020_mastcamz/",
        "https://pds-imaging.jpl.nasa.gov/data/mars2020/",
        "https://pds-imaging.jpl.nasa.gov/data/"
    ]
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    for base_url in base_urls:
        print(f"\n🔗 Test URL: {base_url}")
        
        try:
            response = session.get(base_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            title = soup.find('title')
            title_text = title.get_text() if title else "Başlık yok"
            
            print(f"  Status: {response.status_code}")
            print(f"  Başlık: {title_text}")
            
            # Sol dizinlerini ara
            links = soup.find_all('a', href=True)
            sol_links = []
            
            for link in links:
                href = link['href']
                text = link.get_text().strip()
                
                # Sol dizini kontrolü
                if 'sol' in href.lower() or 'sol' in text.lower():
                    sol_links.append((text, href))
            
            print(f"  Sol linkleri: {len(sol_links)}")
            for text, href in sol_links[:3]:  # İlk 3'ü göster
                print(f"    - {text} -> {href}")
            
            if sol_links:
                # İlk sol linkini test et
                first_sol_link = sol_links[0][1]
                sol_url = urljoin(base_url, first_sol_link)
                print(f"  İlk sol URL: {sol_url}")
                
                # Sol sayfasını test et
                sol_response = session.get(sol_url, timeout=30)
                if sol_response.status_code == 200:
                    sol_soup = BeautifulSoup(sol_response.content, 'html.parser')
                    sol_links = sol_soup.find_all('a', href=True)
                    
                    # Dosya linklerini ara
                    file_links = []
                    for link in sol_links:
                        href = link['href']
                        text = link.get_text().strip()
                        if any(ext in text.lower() for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.xml', '.lbl']):
                            file_links.append((text, href))
                    
                    print(f"  Dosya bulundu: {len(file_links)}")
                    if file_links:
                        print("  ✅ Bu URL yapısı çalışıyor!")
                        return base_url, sol_url
                
        except Exception as e:
            print(f"  ❌ Hata: {e}")
    
    print("\n❌ Çalışan URL yapısı bulunamadı")
    return None, None

def test_alternative_urls():
    """Alternatif URL'leri test et"""
    
    print("\n🔄 Alternatif URL'ler Test Ediliyor...")
    
    # Farklı URL yapıları
    test_urls = [
        "https://pds-imaging.jpl.nasa.gov/data/mars2020/mars2020_mastcamz/data_edr/sol/00001/",
        "https://pds-imaging.jpl.nasa.gov/data/mars2020/mars2020_mastcamz/data_edr/sol/00001",
        "https://pds-imaging.jpl.nasa.gov/data/mars2020/mars2020_mastcamz/sol/00001/",
        "https://pds-imaging.jpl.nasa.gov/data/mars2020/mars2020_mastcamz/sol/00001",
        "https://pds-imaging.jpl.nasa.gov/data/mars2020/sol/00001/",
        "https://pds-imaging.jpl.nasa.gov/data/mars2020/sol/00001"
    ]
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    for url in test_urls:
        print(f"\n🔗 Test: {url}")
        
        try:
            response = session.get(url, timeout=30)
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                title = soup.find('title')
                title_text = title.get_text() if title else "Başlık yok"
                print(f"  Başlık: {title_text}")
                
                # Dosya linklerini ara
                links = soup.find_all('a', href=True)
                file_links = []
                
                for link in links:
                    href = link['href']
                    text = link.get_text().strip()
                    if any(ext in text.lower() for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.xml', '.lbl']):
                        file_links.append((text, href))
                
                print(f"  Dosya bulundu: {len(file_links)}")
                if file_links:
                    print("  ✅ Bu URL çalışıyor!")
                    for text, href in file_links[:3]:
                        print(f"    - {text}")
                    return url
                
        except Exception as e:
            print(f"  ❌ Hata: {e}")
    
    return None

if __name__ == "__main__":
    print("🚀 URL Yapısı Araştırması Başlıyor...")
    
    # Ana URL'leri test et
    base_url, sol_url = find_correct_url()
    
    if not base_url:
        # Alternatif URL'leri test et
        working_url = test_alternative_urls()
        
        if working_url:
            print(f"\n✅ Çalışan URL bulundu: {working_url}")
        else:
            print("\n❌ Hiçbir URL çalışmıyor")
    else:
        print(f"\n✅ Çalışan URL yapısı: {base_url}")
        print(f"Örnek sol URL: {sol_url}") 