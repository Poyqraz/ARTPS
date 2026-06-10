import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import sys

url = 'https://pds-imaging.jpl.nasa.gov/data/mars2020/mars2020_mastcamz/data_raw/'

print(f"PDS dizin URL'si: {url}")

try:
    print("Dizin içeriği alınıyor...")
    response = requests.get(url, timeout=30)
    response.raise_for_status() # HTTP hataları için hata fırlat
    print(f"HTTP Durum Kodu: {response.status_code}")
    
    soup = BeautifulSoup(response.text, 'html.parser')
    print("HTML ayrıştırma başarılı.")
    
    found_links = False
    for link in soup.find_all('a'):
        href = link.get('href')
        if href:
            print(href)
            found_links = True

    if not found_links:
        print("Dizinde hiçbir bağlantı bulunamadı.")

except requests.exceptions.HTTPError as http_err:
    print(f"HTTP hatası oluştu: {http_err}")
except requests.exceptions.ConnectionError as conn_err:
    print(f"Bağlantı hatası oluştu: {conn_err}")
except requests.exceptions.Timeout as timeout_err:
    print(f"İstek zaman aşımına uğradı: {timeout_err}")
except requests.exceptions.RequestException as req_err:
    print(f"İstek sırasında bir hata oluştu: {req_err}")
except Exception as e:
    print(f"Beklenmeyen bir hata oluştu: {e}")
    import traceback
    traceback.print_exc()