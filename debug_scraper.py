"""
Debug Scraper - Sayfa içeriğini kontrol et
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def debug_page_content():
    """Sayfa içeriğini debug eder"""
    
    print("🔍 Sayfa İçeriği Debug Başlıyor...")
    
    # Test URL'si
    base_url = "https://pds-imaging.jpl.nasa.gov/data/mars2020/mars2020_mastcamz/data_edr/"
    test_url = urljoin(base_url, "sol/00001/")
    
    print(f"Test URL: {test_url}")
    
    # Session oluştur
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    try:
        # Sayfayı getir
        print("Sayfa getiriliyor...")
        response = session.get(test_url, timeout=30)
        response.raise_for_status()
        
        print(f"Status Code: {response.status_code}")
        print(f"Content Length: {len(response.content)} bytes")
        
        # HTML içeriğini parse et
        soup = BeautifulSoup(response.content, 'html.parser')
        
        print("\n📄 Sayfa Başlığı:")
        title = soup.find('title')
        if title:
            print(f"  {title.get_text()}")
        else:
            print("  Başlık bulunamadı")
        
        print("\n🔗 Tüm Linkler:")
        links = soup.find_all('a', href=True)
        print(f"Toplam {len(links)} link bulundu")
        
        for i, link in enumerate(links[:10]):  # İlk 10 linki göster
            href = link['href']
            text = link.get_text().strip()
            print(f"  {i+1}. {text} -> {href}")
        
        if len(links) > 10:
            print(f"  ... ve {len(links) - 10} link daha")
        
        print("\n📁 Dosya Linkleri:")
        file_links = []
        for link in links:
            href = link['href']
            text = link.get_text().strip()
            
            # Dosya uzantısı kontrolü
            if any(ext in text.lower() for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.xml', '.lbl']):
                file_links.append((text, href))
        
        print(f"Toplam {len(file_links)} dosya linki bulundu")
        for i, (text, href) in enumerate(file_links[:5]):  # İlk 5 dosyayı göster
            print(f"  {i+1}. {text} -> {href}")
        
        if len(file_links) > 5:
            print(f"  ... ve {len(file_links) - 5} dosya daha")
        
        # Sayfa içeriğinin bir kısmını göster
        print("\n📝 Sayfa İçeriği (İlk 500 karakter):")
        content_preview = soup.get_text()[:500]
        print(f"  {content_preview}...")
        
    except Exception as e:
        print(f"❌ Hata: {e}")

if __name__ == "__main__":
    debug_page_content() 