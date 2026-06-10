"""
Perseverance Scraper Test Script
"""

from src.data.perseverance_scraper import PerseveranceScraper
import time

def test_scraper():
    """Scraper'ı test eder"""
    
    print("🧪 Perseverance Scraper Test Başlıyor...")
    
    # Scraper oluştur
    scraper = PerseveranceScraper(
        base_url="https://pds-imaging.jpl.nasa.gov/data/mars2020/mars2020_mastcamz/data_edr/",
        output_dir="data/real_mars_data/raw_images"
    )
    
    # Test 1: Tek sol testi
    print("\n📋 Test 1: Tek Sol Testi (Sol 1)")
    result = scraper.process_sol(1)
    print(f"Sonuç: {result}")
    
    # Test 2: Küçük aralık testi
    print("\n📋 Test 2: Küçük Aralık Testi (Sol 1-3)")
    results = scraper.scrape_sol_range(
        start_sol=1,
        end_sol=3,
        delay=1.0  # 1 saniye bekle
    )
    
    # Özet yazdır
    scraper.print_summary()
    
    print("\n✅ Test tamamlandı!")

if __name__ == "__main__":
    test_scraper() 