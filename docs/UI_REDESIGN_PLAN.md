# ARTPS — Arayüz Yeniden Tasarım Planı (Mars / Uzay Teması)

> Hedef: Mevcut **varsayılan Streamlit** arayüzünü; Mars/uzay temalı, mühendislik sınıfı,
> uluslararası sempozyum sunumlarına uygun, "mission control" estetiğinde bir bilim
> paneline dönüştürmek. **Davranış/algoritmalar korunur, yalnızca sunum katmanı değişir.**

İlgili analiz: depo baştan sona incelendi. Ana uygulama kök `app.py` (2242 satır).
Arayüzde hiç tema/CSS yok, görsel kimlik tamamen emoji. `.pth` model dosyaları ve
`mars_images/` örnekleri depoda yok → demo sırasında graceful degradation şart.

### Onaylanan kararlar (kullanıcı)
- **Görseller:** Önce internetten **en ilgili mevcut görseller** (Mars yüzeyi, rover) kullanılır.
  Telif güvenliği için kaynak **NASA/JPL public domain** (Perseverance/Curiosity ham görüntüleri,
  Mars manzaraları). Uygun görsel bulunamazsa `GenerateImage` ile üretilir.
- **Logo:** ARTPS'e özel hazır logo internette bulunmadığından **üretilecek** (SVG/PNG wordmark + ikon).
- **Eski dosyalar:** `ARTPS/` ve diğer eski kopyalar **SİLİNMEZ**; `archive/` (veya `_legacy/`)
  altına taşınarak depolanır. Aktif geliştirme kök `app.py` üzerinde.

### Görsel yön (referans ekran görüntüsü hk.)
Kullanıcı Lovable tarzı bir hero ekran görüntüsü paylaştı (üst nav + DOI rozeti + büyük başlık +
turuncu vurgu kelime + CTA'lar + Mars ufku). **Bu yapı birebir hedef değil — daha iyisi yapılacak.**
Yön: aynı kalitede bir **hero/landing bandı** + hemen altında gerçek **mission-control analiz paneli**.
"Daha iyi" = katmanlı Mars arka planı (derinlik/parallax hissi), cam efektli (glassmorphism)
durum panelleri, canlı telemetri şeridi (cihaz CUDA/CPU, yüklü model sayısı, sürüm/DOI),
gradient turuncu vurgular ve HUD ızgara dokusu.

---

## 0. Kapsam ve İlkeler

- **Tek doğruluk kaynağı:** Tüm değişiklikler **kök `app.py`** üzerinde. `ARTPS/app.py` eski
  kopyası bu çalışmaya dahil edilmez (son adımda senkron/temizlik kararı verilir).
- **Davranış değişmez:** Model yükleme, `analyze_mars_image()`, curiosity formülü, slider
  mantığı aynı kalır. Sadece render/stil/yerleşim ve boş-durum davranışı dokunulur.
- **Tema tek noktadan:** Renk/tipografi/CSS tek bir `src/ui/theme.py` + `assets/style.css`
  üzerinden gelir; kod içine dağılmış inline stil yazılmaz.
- **Geri dönülebilirlik:** Her faz bağımsız test edilebilir; `streamlit run app.py` ile
  görsel doğrulama yapılır.

---

## 1. Tasarım Dili (Design System)

**Renk paleti — "Deep Space + Mars Regolith":**

| Rol | Hex | Kullanım |
|---|---|---|
| Arka plan (uzay) | `#0B0E14` | app background |
| Panel/kart | `#141A24` | container, kart |
| Kenarlık/grid | `#243044` | border, ayraç, HUD çizgileri |
| Mars turuncu (vurgu) | `#E2725B` → `#C1440E` | primary, vurgu, butonlar |
| Accent metal/teknik | `#7DD3FC` (buz mavisi) | linkler, ikincil veri |
| Başarı/uyarı/hata | `#34D399` / `#FBBF24` / `#F87171` | durum rozetleri |
| Metin (ana/ikincil) | `#E6EAF2` / `#9AA7BD` | tipografi |

**Tipografi:**
- Başlık: `Orbitron` veya `Space Grotesk` (uzay/teknik karakter).
- Gövde/veri: `Inter` veya `IBM Plex Sans` (okunabilir, mühendislik).
- Metrik/sayılar: `IBM Plex Mono` (telemetri görünümü).
- Fontlar `assets/fonts/` altına gömülür (sempozyumda çevrimdışı çalışsın diye CDN'e bağımlı kalınmaz; CDN fallback bırakılır).

**Bileşen dili:** kenarlıklı koyu kartlar, ince grid çizgileri, yumuşak gölge, köşe
radyusu 10–14px, "HUD" tarzı metrik kutuları, turuncu vurgulu birincil aksiyonlar.

---

## 2. Faz Faz Uygulama Planı

### Faz 1 — Temel altyapı (theme config + asset iskeleti)
- `.streamlit/config.toml` oluştur: `base="dark"`, `primaryColor`, `backgroundColor`,
  `secondaryBackgroundColor`, `textColor`, `font` paletten atanır.
- `assets/` klasörü kur: `assets/style.css`, `assets/fonts/`, `assets/img/` (logo + demo
  görseller), `assets/logo_artps.svg`.
- `src/ui/theme.py` ekle: `inject_theme()` (CSS dosyasını okuyup tek `st.markdown(...,
  unsafe_allow_html=True)` ile enjekte eder), `metric_card()`, `section_header()`,
  `status_badge()`, `empty_state()` yardımcıları.
- `app.py` `main()` başında `inject_theme()` çağrısı (mevcut `st.set_page_config` hemen sonrası).
- **Doğrulama:** uygulama koyu temada açılıyor, hata yok.

### Faz 2 — Global CSS tasarım sistemi (`assets/style.css`)
- Uygulama arka planı, ana konteyner genişliği/padding.
- Sidebar stilizasyonu (koyu, ayraçlı "Kontrol Paneli / Mission Control" görünümü).
- `st.tabs` stilizasyonu: aktif sekme turuncu alt-çizgi, ikon hizası.
- `st.metric` → telemetri kartı görünümü (mono font, etiket küçük caps, delta renkleri).
- `st.button[type=primary]` → Mars turuncu gradient, hover/active durumları.
- `st.container(border=True)` ve tablo/expander stilleri.
- **Doğrulama:** her sekme tutarlı koyu tema; kontrast WCAG-AA seviyesinde okunur.

### Faz 3 — Hero / Landing bandı + marka kimliği (referanstan daha iyi)
Referans ekran görüntüsünün kalitesini aşan bir açılış bandı (custom HTML/CSS, tek helper):
- **Üst nav (sticky):** sol `assets/logo_artps.svg` + "ARTPS / TARGET PRIORITIZATION" wordmark;
  orta gezinme (Overview · Analyze · Parameters → sekmelere/anchor'lara kaydırma); sağ "GitHub" rozeti.
- **Durum rozeti pill:** `● ARTPS v1.3.0 · DOI 10.13140/RG.2.2.12215.18088` (mono font, canlı renk noktası).
- **Hero başlık:** büyük, kalın; "Autonomous target prioritization for **planetary rovers**." —
  vurgu kelime Mars turuncu gradient. TR alt satır da eklenir.
- **Alt başlık:** hibrit AI özeti (AE + PaDiM/PatchCore + ViT derinlik + curiosity head).
- **CTA'lar:** birincil turuncu "Demoyu Başlat" (analiz sekmesine kaydırır) + ikincil çerçeveli "Kaynak".
- **Arka plan:** NASA Mars ufku görseli (`assets/img/hero_mars.jpg`) üzerine koyu gradient + ince HUD ızgara;
  hafif derinlik/parallax hissi. Çevrimdışı çalışsın diye görsel gömülür.
- **Telemetri şeridi:** cihaz (CUDA/CPU), yüklü model sayısı, aktif derinlik modeli — canlı `models` dict'inden.
- `st.title("🚀 ...")` ve düz `st.markdown` başlığı bu band ile değiştirilir.
- Sidebar başlığını "Mission Control" temalı hale getir; mevcut slider/expander'lar **korunur**.
- **Doğrulama:** açılış ekranı sempozyum/landing kalitesinde, referanstan daha zengin.

### Faz 4 — Grafik teması birliği (Matplotlib + Plotly)
- `src/ui/plotting.py`: tek bir koyu Matplotlib stili (`rcParams`: koyu zemin, açık eksen,
  grid `#243044`, font ailesi) + ortak `apply_dark_style(ax/fig)`.
- Anomali haritaları için tutarlı colormap dili (ör. `inferno`/`magma`); derinlik için
  `turbo` korunabilir ama koyu zemine oturtulur; colorbar/etiketler okunur renkte.
- Plotly pie chart (Sistem Durumu) → koyu `template`, marka renk dizisi.
- **Doğrulama:** tüm figürler koyu temayla görsel olarak tutarlı, beyaz kenar yok.

### Faz 5 — Sekme bazlı yerleşim iyileştirmesi
Mevcut 5 sekme korunur, içerik bordered kart/grid'lere yerleştirilir:
- **📸 Görüntü Analizi:** yükleme + iyileştirme kontrolleri kartta; sonuçlar "telemetri
  paneli" (curiosity, anomali, sınıf, MSE) + anomali haritası + tanılama paneli.
- **🔍 Derinlik Analizi:** 2D/3D görseller kartta, 14 metrik HUD grid'inde.
- **📊 Sistem Durumu:** model envanteri kartları + Plotly pie (koyu tema).
- **🎯 Demo Veriler:** görsel grid stilize; klasör yoksa şık boş-durum.
- **ℹ️ Hakkında:** sempozyum dokümanı görünümü (formül, metrik, künye, DOI/lisans).
- **Doğrulama:** her sekme bağımsız, taşma/bozulma yok.

### Faz 6 — Graceful degradation & demo varlıkları (sempozyum güvenliği)
- Model `.pth` yoksa: uygulama `st.error` ile **durmak yerine** "Demo/Tanıtım Modu" boş-durum
  kartı gösterir; arayüz tema/yerleşimiyle gezilebilir kalır (`load_models()` dönüşü `None`
  olduğunda `main()` erken-return yerine demo modu).
- `mars_images/` yoksa: Demo sekmesi `assets/img/` içindeki paketlenmiş örnek Mars
  görselleriyle çalışır.
- **Doğrulama:** model dosyaları/sample klasörü olmadan da uygulama sunulabilir durumda açılır.

### Faz 7 — Çift dil (opsiyonel, sempozyum için önerilir)
- Sidebar'da TR/EN seçici; başlık/sekme/etiketlerde `i18n` sözlüğü (`src/ui/i18n.py`).
- Kapsam küçük tutulur (statik etiketler); model çıktıları aynı kalır.

### Faz 8 — Arşivleme & dokümantasyon
- `ARTPS/` eski kopyası ve diğer eski/yedek dosyalar **silinmez** → `archive/` (veya `_legacy/`)
  altına taşınır; depo kökü sadeleşir, geçmiş korunur.
- `README.md` çalıştırma talimatını kök `app.py` ile güncelle; yeni arayüz ekran görüntüleri ekle.
- `requirements.txt`: yeni görsel bağımlılık eklenmediğinden değişiklik beklenmez (teyit edilir).

---

## 3. Oluşturulacak / Değişecek Dosyalar

| Dosya | İşlem | Amaç |
|---|---|---|
| `.streamlit/config.toml` | yeni | Koyu uzay teması (renk/font) |
| `assets/style.css` | yeni | Global CSS tasarım sistemi |
| `assets/logo_artps.svg`, `assets/img/*`, `assets/fonts/*` | yeni | Marka + demo + font (NASA public-domain görseller + üretilen logo) |
| `archive/` | yeni | `ARTPS/` + eski dosyaların taşınacağı arşiv (silinmez) |
| `src/ui/theme.py` | yeni | CSS enjeksiyon + bileşen helper'ları |
| `src/ui/plotting.py` | yeni | Matplotlib/Plotly koyu tema |
| `src/ui/i18n.py` | yeni (ops.) | TR/EN sözlüğü |
| `app.py` | düzenle | `inject_theme()`, hero başlık, helper kullanımı, graceful degradation |
| `README.md` | düzenle | Çalıştırma + görseller |

---

## 4. Riskler ve Önlemler

- **Streamlit CSS kırılganlığı:** sürüm güncellemesinde class adları değişebilir → seçicileri
  mümkün olduğunca yapısal/`data-testid` üzerinden, minimum sayıda tut.
- **Model/görsel eksikliği:** Faz 6 ile demo modu garanti altına alınır (sempozyum kritik).
- **Tek dosya büyüklüğü (2242 satır):** stil/grafik mantığı `src/ui/` altına taşınarak
  `app.py` şişirilmez; davranış kodu yerinde bırakılır.
- **Çevrimdışı sunum:** fontlar/varlıklar gömülür, CDN'e zorunlu bağımlılık olmaz.

---

## 5. Önerilen Sıra ve Onay Noktaları

1. Faz 1–2 (tema + global CSS) → **ilk görsel onay** (en yüksek etki).
2. Faz 3–4 (hero + grafik teması) → **ikinci görsel onay**.
3. Faz 5 (sekme yerleşimi) → sekme sekme onay.
4. Faz 6 (demo modu) → sunum provası.
5. Faz 7–8 (i18n + temizlik) → kapanış.

> Notlar:
> - Görsel kaynağı: önce NASA/JPL public-domain (Mars ufku, rover); bulunamazsa `GenerateImage`.
> - `ARTPS/` ve eski dosyalar silinmez → `archive/` altına taşınır.
