# Yerel model ağırlıkları (git dışı)

Bu klasördeki büyük dosyalar `.gitignore` ile sürüm kontrolüne **dahil edilmez**.
Uygulama yerelde çalışırken bu dosyaları otomatik arar.

## DPT_Large derinlik modeli

| Dosya | Boyut (yaklaşık) | Açıklama |
|-------|------------------|----------|
| `dpt_large_384.pt` | ~1.3 GB | MiDaS DPT_Large state_dict |

Dosyayı proje kökündeki `raw_models/dpt_large_384.pt` konumuna koyun.
`pip install -r requirements.txt` ile `timm` kurulu olmalıdır (DPT mimarisi için gerekli).

Yükleme sırası (`src/models/depth_estimation.py`):

1. Yerel `dpt_large_384.pt` (state_dict veya TorchScript)
2. PyTorch Hub (`intel-isl/MiDaS`) — internet gerekir
3. Basit CNN fallback (~424K parametre)

Dosya yoksa veya `timm` eksikse uygulama fallback moduna düşer; UI'da
"Basit/Fallback" görünür.
