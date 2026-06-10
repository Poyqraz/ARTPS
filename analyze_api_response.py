"""
API Yanıtlarını Detaylı Analiz Et
"""

import requests
import json
from collections import Counter

def analyze_api_responses():
    """API yanıtlarını detaylı analiz eder"""
    
    print("🔍 API Yanıtları Detaylı Analiz Ediliyor...")
    
    base_url = "https://mars.nasa.gov/api/v1/raw_image_items/"
    
    # Test parametreleri
    test_params = [
        {'mission': 'mars2020', 'limit': 10},
        {'mission': 'msl', 'limit': 10},
        {'instrument': 'mastcamz', 'limit': 10},
        {'instrument': 'navcam', 'limit': 10},
        {'sol': 1, 'limit': 10},
        {'sol': 1000, 'limit': 10},
        {'limit': 10}  # Parametresiz
    ]
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    all_missions = []
    all_instruments = []
    all_sols = []
    all_titles = []
    
    for i, params in enumerate(test_params):
        print(f"\n📋 Test {i+1}: {params}")
        
        try:
            response = session.get(base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('items', [])
                
                print(f"  Items found: {len(items)}")
                
                # İlk 3 öğeyi detaylı analiz et
                for j, item in enumerate(items[:3]):
                    mission = item.get('mission', 'Unknown')
                    instrument = item.get('instrument', 'Unknown')
                    sol = item.get('sol', 'Unknown')
                    title = item.get('title', 'Unknown')
                    
                    all_missions.append(mission)
                    all_instruments.append(instrument)
                    all_sols.append(sol)
                    all_titles.append(title)
                    
                    print(f"    Item {j+1}:")
                    print(f"      Mission: {mission}")
                    print(f"      Instrument: {instrument}")
                    print(f"      Sol: {sol}")
                    print(f"      Title: {title}")
                    
                    # URL'yi kontrol et
                    url = item.get('url', '')
                    if 'mars.jpl.nasa.gov' in url:
                        print(f"      URL: {url}")
                        
                        # URL'den mission bilgisi çıkar
                        if 'msl' in url:
                            print(f"      URL Mission: MSL (Curiosity)")
                        elif 'mars2020' in url:
                            print(f"      URL Mission: Mars 2020 (Perseverance)")
                        else:
                            print(f"      URL Mission: Unknown")
                
                # Tüm öğelerin mission dağılımını kontrol et
                missions_in_batch = [item.get('mission', 'Unknown') for item in items]
                mission_counts = Counter(missions_in_batch)
                print(f"  Mission distribution: {dict(mission_counts)}")
                
            else:
                print(f"  Status: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Hata: {e}")
    
    # Genel analiz
    print(f"\n📊 GENEL ANALİZ")
    print(f"=" * 50)
    
    mission_counts = Counter(all_missions)
    instrument_counts = Counter(all_instruments)
    sol_counts = Counter(all_sols)
    
    print(f"Toplam öğe: {len(all_missions)}")
    print(f"Mission dağılımı: {dict(mission_counts)}")
    print(f"Instrument dağılımı: {dict(instrument_counts)}")
    print(f"Sol dağılımı: {dict(sol_counts)}")
    
    # Perseverance verisi var mı kontrol et
    perseverance_found = any('mars2020' in mission.lower() or 'perseverance' in mission.lower() 
                           for mission in all_missions)
    
    if perseverance_found:
        print(f"\n✅ Perseverance verisi bulundu!")
    else:
        print(f"\n❌ Perseverance verisi bulunamadı")
        print(f"Tüm veriler: {all_missions}")
    
    return perseverance_found

def test_different_sol_ranges():
    """Farklı sol aralıklarını test et"""
    
    print(f"\n🔍 Farklı Sol Aralıkları Test Ediliyor...")
    
    base_url = "https://mars.nasa.gov/api/v1/raw_image_items/"
    
    # Perseverance'ın bilinen sol aralıkları
    sol_ranges = [
        (1, 50),      # İlk günler
        (100, 150),   # Erken dönem
        (500, 550),   # Orta dönem
        (1000, 1050), # Geç dönem
        (1500, 1550), # En son dönem
    ]
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    for start_sol, end_sol in sol_ranges:
        print(f"\n📋 Sol {start_sol}-{end_sol} arası test ediliyor...")
        
        for sol in range(start_sol, min(start_sol + 5, end_sol + 1)):  # Her aralıktan 5 sol test et
            params = {'mission': 'mars2020', 'sol': sol, 'limit': 5}
            
            try:
                response = session.get(base_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    items = data.get('items', [])
                    
                    if items:
                        print(f"  Sol {sol}: {len(items)} görüntü bulundu")
                        
                        # İlk öğeyi kontrol et
                        first_item = items[0]
                        mission = first_item.get('mission', 'Unknown')
                        instrument = first_item.get('instrument', 'Unknown')
                        
                        print(f"    Mission: {mission}")
                        print(f"    Instrument: {instrument}")
                        
                        if 'mars2020' in mission.lower() or 'perseverance' in mission.lower():
                            print(f"    ✅ Perseverance verisi!")
                            return True
                    else:
                        print(f"  Sol {sol}: Görüntü bulunamadı")
                        
            except Exception as e:
                print(f"  Sol {sol}: Hata - {e}")
    
    return False

if __name__ == "__main__":
    print("🚀 API Detaylı Analiz Başlıyor...")
    
    # 1. Genel API yanıtlarını analiz et
    perseverance_found = analyze_api_responses()
    
    # 2. Farklı sol aralıklarını test et
    if not perseverance_found:
        print(f"\n🔄 Farklı sol aralıkları test ediliyor...")
        perseverance_found = test_different_sol_ranges()
    
    # Sonuç
    if perseverance_found:
        print(f"\n✅ Perseverance verisi bulundu!")
    else:
        print(f"\n❌ Perseverance verisi bulunamadı")
        print(f"API sadece Curiosity verisi döndürüyor gibi görünüyor.")
    
    print(f"\n✅ Analiz tamamlandı!") 