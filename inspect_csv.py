import requests
import pandas as pd
import io
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mastcam-Z ham görüntü envanter CSV URL'si
inventory_url = "https://pds-imaging.jpl.nasa.gov/data/mars2020/mars2020_mastcamz/data_raw/collection_data_raw_inventory.csv"

print(f"Envanter CSV URL'si: {inventory_url}")

try:
    logger.info("Envanter CSV içeriği indiriliyor...")
    response = requests.get(inventory_url, timeout=60, verify=False) # SSL doğrulaması kapatıldı
    response.raise_for_status()
    raw_csv_content = response.text
    logger.info("Envanter CSV içeriği başarıyla indirildi.")

    # İlk 10 satırı yazdır (hata ayıklama için)
    print("\n--- İlk 10 satır (ham içerik) ---")
    for i, line in enumerate(raw_csv_content.splitlines()):
        print(line)
        if i >= 9: # İlk 10 satır
            break
    print("--------------------------------\n")

    # Farklı skiprows değerleriyle denemeler
    test_skiprows = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for skip_val in test_skiprows:
        print(f"--- skiprows={skip_val} ile deneme ---")
        try:
            df = pd.read_csv(io.StringIO(raw_csv_content),
                             sep='|',
                             comment='#',
                             header=None, # Başlık yok varsayımı
                             skipinitialspace=True,
                             skiprows=skip_val)
            
            print(f"DataFrame boyutu: {df.shape}")
            print("DataFrame ilk 5 satır:")
            print(df.head())
            print("\n")
            
        except Exception as e:
            print(f"Hata (skiprows={skip_val}): {e}")
            print("\n")

except requests.exceptions.RequestException as e:
    logger.error(f"İstek hatası oluştu: {e}")
except Exception as e:
    logger.error(f"Beklenmeyen bir hata oluştu: {e}")
    import traceback
    traceback.print_exc()