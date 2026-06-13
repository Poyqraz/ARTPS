"""Turkish UI strings."""

MESSAGES: dict[str, str] = {
    # Core
    "sidebar.language": "Dil",
    "sidebar.credits": "🛰️ Yapım: [Poyraz BAYDEMİR](https://github.com/Poyqraz) · [ResearchGate DOI](http://dx.doi.org/10.13140/RG.2.2.12215.18088)",
    "sidebar.license": "📄 Lisans: [MIT License](https://github.com/Poyqraz/ARTPS/blob/main/LICENSE)",
    "sidebar.control_panel": "🎛️ Kontrol Paneli",
    "sidebar.models_loading": "🤖 Hibrit modeller yükleniyor...",
    "sidebar.models_loaded_prefix": "Modeller yüklendi:\n",
    "sidebar.model.autoencoder": "✅ Autoencoder",
    "sidebar.model.classifier": "✅ Hibrit Sınıflandırıcı",
    "sidebar.model.depth": "✅ Derinlik Tahmini ({model_type}) - {quality}",
    "sidebar.model.padim": "✅ PaDiM (Anomali Füzyon)",
    "sidebar.model.patchcore": "✅ PatchCore (Anomali Füzyon)",
    "sidebar.depth_active": "Aktif Derinlik Modeli: {model_type} — Parametre: {param_count:,} — {quality}",
    "sidebar.params_settings": "📊 Parametre Ayarları",

    # load_models messages
    "models.device": "🖥️ Kullanılan cihaz: {device}",
    "models.autoencoder_missing": "❌ Autoencoder model bulunamadı: {path}",
    "models.classifier_missing": "⚠️ Sınıflandırıcı model bulunamadı, sadece anomali tespiti kullanılacak",
    "models.padim_stats_missing": "⚠️ PaDiM istatistikleri bulunamadı: results/padim_stats.pth. Sadece AE tabanlı anomali kullanılacak",
    "models.padim_load_failed": "⚠️ PaDiM yüklenemedi: {error}",
    "models.patchcore_missing": "ℹ️ PatchCore bellek bankası bulunamadı (tools/prepare_patchcore_bank.py ile üretebilirsiniz)",
    "models.patchcore_load_failed": "⚠️ PatchCore yüklenemedi: {error}",
    "models.dpt_success": "✅ DPT_Large modeli başarıyla yüklendi (yüksek doğruluk) - {params:,} parametre",
    "models.dpt_fallback": "⚠️ DPT_Large modeli yüklenemedi, basit model kullanılıyor - {params:,} parametre",
    "models.dpt_hub_fallback": "ℹ️ PyTorch Hub bağlantı sorunu nedeniyle fallback model aktif",
    "models.depth_load_failed": "❌ Derinlik tahmin modülü yüklenemedi: {error}",
    "models.curiosity_loaded": "🧭 Curiosity ağırlıkları otomatik yüklendi (results/curiosity_weights.json)",
    "models.curiosity_load_failed": "Curiosity ağırlıkları yüklenemedi: {error}",
    "models.quality.high": "Yüksek Doğruluk",
    "models.quality.simple": "Basit Model",

    # Hero
    "hero.brand_tag": "Hedef Önceliklendirme",
    "hero.badge": "ARTPS {version} · YAYINLANDI · DOI {doi}",
    "hero.title_html": "Otonom bilimsel hedef önceliklendirme — <span class=\"accent\">gezgin robotlar</span> için.",
    "hero.subtitle": "Mars yüzeyinde bir sonraki bilimsel hedefi otonom seçin",
    "hero.body_html": (
        "ARTPS; bir autoencoder, iki anomali dedektörü (PaDiM + PatchCore), "
        "bir Vision Transformer derinlik modeli ve öğrenilebilir bir "
        "<b style=\"color:#E2725B\">İlginçlik (Curiosity) Puanı</b> başlığını birleştirerek "
        "bir rover'ın Mars yüzeyinde bir sonraki <i>hangi hedefi</i> inceleyeceğine karar verir."
    ),
    "telemetry.device": "CIHAZ",
    "telemetry.active_models": "AKTIF MODEL",
    "telemetry.depth": "DERINLIK",

    # Demo mode
    "demo.title": "Tanıtım Modu — Modeller yüklenemedi",
    "demo.message": (
        "Eğitilmiş model dosyaları (<code>results/*.pth</code>) bulunamadı. "
        "Arayüz ve tasarım gezilebilir; analiz için model dosyalarını "
        "<code>results/</code> klasörüne ekleyin."
    ),

    # Params sliders
    "params.alpha.label": "α (Alfa) - Bilinen Değer Ağırlığı",
    "params.alpha.help": "Curiosity skorunda sınıflandırıcının tahmin ettiği 'bilinen değer' katkısı. Yüksek olduğunda bilinen bilimsel açıdan değerli sınıflara benzer görüntüler daha çok öne çıkar.",
    "params.beta.label": "β (Beta) - Anomali Ağırlığı",
    "params.beta.help": "Curiosity skorunda AE tabanlı anomali MSE katkısı. Yüksek olduğunda beklenmedik/düzensiz yapılar daha çok öne çıkar.",
    "params.w_combined.label": "w_combined (Birleşik Anomali)",
    "params.w_combined.help": "Birleşik anomali haritasının ortalama yoğunluğunun curiosity skoruna katkısı. AE farkı, derinlik kenarı ve doku bileşenlerinden oluşur.",
    "params.w_dvar.label": "w_depth_variance",
    "params.w_dvar.help": "Derinlik varyansının (3B yapı çeşitliliği) curiosity skoruna katkısı. Yüksek varyans, daha karmaşık jeomorfoloji anlamına gelebilir.",
    "params.w_rough.label": "w_roughness",
    "params.w_rough.help": "Pürüzlülük (gradyan ve laplace değişkenliği) katkısı. Küçük taş/kum çizgileri gibi ince detayları öne çıkarabilir.",
    "params.anomaly_threshold.label": "Anomali Eşiği",
    "params.anomaly_threshold.help": "AE MSE için karar eşiği. Bu eşik üstü değerler tek başına 'anormal' kabul edilebilir.",
    "params.ref_mse.label": "Curiosity Referans MSE",
    "params.ref_mse.help": "Curiosity normalizasyonu için AE MSE referansı. Yaklaşık olarak 2×ref MSE → 1.0 skora sıkıştırılır.",

    # Policy expander
    "params.policy.expander": "🛡️ Operasyonel Seçim Politikası (Clustering + Buffer)",
    "params.policy.enable": "Aktif et (önerilen hedef seti üret)",
    "params.policy.enable_help": "Latent-space clustering ile farklı şekil tiplerinden hedef seçer ve similarity nedeniyle bastırılan yüksek değerli hedefleri Priority Buffer'a alır.",
    "params.policy.budget": "Hedef bütçesi (B)",
    "params.policy.method": "Kümeleme yöntemi",
    "params.policy.k": "K (KMeans)",
    "params.policy.eps": "eps (DBSCAN)",
    "params.policy.min_samples": "min_samples (DBSCAN)",
    "params.policy.lambda_penalty": "λ (Soft Penalty)",
    "params.policy.tau_high": "Buffer τ_high (ham skor)",
    "params.policy.tau_delta": "Buffer τ_Δ (düşüş)",
    "params.policy.history_m": "History uzunluğu (m)",
    "params.policy.history_m_help": "0 ise geçmiş çeşitlilik baskısı kapatılır.",
    "params.policy.crop_margin": "Crop margin",
    "params.policy.crop_margin_help": "Latent çıkarımı için kutuya eklenecek bağlam payı.",

    # Detection expander
    "params.detection.expander": "🔧 Tespit Ayarları (Gelişmiş)",
    "params.detection.unified_threshold": "Birleşik Anomali Eşiği",
    "params.detection.hyst_high": "Histerezis High (%)",
    "params.detection.hyst_low": "Histerezis Low (%)",
    "params.detection.nms_iou": "NMS IoU",
    "params.detection.top_k": "Top-K Kutu",
    "params.detection.min_area": "Min Kutu Alanı (%)",
    "params.detection.min_area_help": "Görüntü alanına göre",
    "params.detection.weights_header": "⚖️ Ağırlıklar",
    "params.detection.w_recon": "w_recon (fark)",
    "params.detection.w_depth": "w_depthEdge (∇depth)",
    "params.detection.w_texture": "w_texture (gölge+kenar)",
    "params.detection.w_lap": "w_lap (Δ depth)",
    "params.detection.edge_reinf": "edge reinforce",
    "params.detection.w_detail": "w_detail (ince detay)",
    "params.detection.w_detail_help": "Küçük taş/kum çizgilerini vurgulayan çok ölçekli detay bileşeni",
    "params.detection.w_padim": "w_padim (PaDiM füzyon)",
    "params.detection.w_padim_help": "PaDiM anomali haritasının birleşik haritaya katkısı",
    "params.detection.w_patchcore": "w_patchcore (PatchCore füzyon)",
    "params.detection.w_patchcore_help": "PatchCore anomali haritasının birleşik haritaya katkısı",
    "params.detection.merge_header": "🔗 Kutu Birleştirme",
    "params.detection.merge_iou": "Birleştirme IoU",
    "params.detection.merge_tol": "Merkez Yakınlık (diagonal oranı)",
    "params.detection.merge_caption": "Yakın küçük kutuları birleşik hedefe toplar; uzak alandaki küçük detaylar için daha düşük IoU ile koruma sağlar.",
    "params.detection.shadow_header": "🌑 Gölge Bastırma (Saha Ayarı)",
    "params.detection.alpha_shad": "Gölge Bastırma Gücü",
    "params.detection.alpha_shad_help": "Koyu + düşük kenarlı bölgeleri bastırma",
    "params.detection.beta_illum": "Aydınlatma-Kenar Azaltımı",
    "params.detection.beta_illum_help": "Görüntü kenarı yüksek ama derinlik kenarı düşükse etkisini azaltır",
    "params.detection.shadow_cut": "Gölge Eleme Eşiği",
    "params.detection.shadow_cut_help": "Saf gölge bölgeleri eleme için alt sınır",
    "params.detection.img_edge_min": "Min Görüntü Kenarı",
    "params.detection.depth_edge_min": "Min Derinlik Kenarı",
    "params.detection.spec_gamma": "Speküler Bastırma Gücü",
    "params.detection.spec_gamma_help": "Yüksek parlaklık + düşük satürasyon bölgeleri bastırma",
    "params.detection.spec_cut": "Speküler Eleme Eşiği",
    "params.detection.spec_lowvar_gamma": "Düşük Varyans Azaltımı",
    "params.detection.spec_lowvar_help": "Düşük doku (düşük varyans) speküler noktalara ek azaltım uygular",
    "params.detection.spec_var_thresh": "Düşük Varyans Eşiği",
    "params.detection.focus_header": "🎯 Odak Görselleri",
    "params.detection.focus_h": "Odak Karo Yüksekliği",
    "params.detection.focus_overlay": "Isı + Orijinal karışımını göster (overlay)",
    "params.detection.focus_sharpen": "Odak Keskinleştirme (unsharp)",
    "params.detection.focus_hide_empty_depth": "Derinlik karosu yoksa gizle",
    "params.detection.focus_interp": "Yeniden örnekleme",
    "params.detection.focus_caption": "Hız için analizden hemen sonra odak karoları önceden üretilir.",

    # Curiosity weights expander
    "params.curiosity.expander": "🧭 Curiosity Ağırlıkları (Opsiyonel)",
    "params.curiosity.use_loaded": "Dosyadan yüklenen ağırlıkları kullan",
    "params.curiosity.weights_path": "Ağırlık dosyası (JSON)",
    "params.curiosity.load_btn": "Yükle",
    "params.curiosity.reset_btn": "Varsayılanlara dön",
    "params.curiosity.loaded_ok": "Ağırlıklar yüklendi",
    "params.curiosity.load_error": "Yükleme hatası: {error}",
    "params.curiosity.defaults_active": "Varsayılan ağırlıklar aktif",
    "params.curiosity.active_caption": "Aktif: known={known:.3f}, anomaly={anomaly:.3f}, combined={combined:.3f}, dvar={dvar:.3f}, rough={rough:.3f}",

    # Tabs and sections
    "tabs.image_analysis": "📸 Görüntü Analizi",
    "tabs.depth": "🔍 Derinlik Analizi",
    "tabs.system": "📊 Sistem Durumu",
    "tabs.demo": "🎯 Demo Veriler",
    "tabs.about": "ℹ️ Hakkında",
    "section.image_analysis": "📸 Mars Görüntüsü Hibrit Analizi",
    "section.depth": "🔍 Derinlik Analizi",
    "section.system": "📊 Sistem Durumu",
    "section.demo": "🎯 Demo Veriler",
    "section.about": "ℹ️ ARTPS Hibrit Sistem Hakkında",

    # Tab1 image analysis
    "analysis.upload_label": "Mars görüntüsü yükleyin (JPG, PNG)",
    "analysis.enhance_header": "🧹 Otomatik Görüntü İyileştirme",
    "analysis.opt_upscale": "Upscale",
    "analysis.opt_upscale_help": "Düşük çözünürlüklü görselleri akıllı büyütme",
    "analysis.opt_denoise": "Denoise",
    "analysis.opt_denoise_help": "Yüksek gürültülü görüntülerde renkli gürültü giderme",
    "analysis.opt_clahe": "Kontrast (CLAHE)",
    "analysis.opt_gamma": "Pozlama (Gamma)",
    "analysis.opt_sharp": "Keskinleştirme",
    "analysis.enhance_btn": "✨ Görüntüyü Otomatik İyileştir",
    "analysis.steps_applied": "Uygulanan adımlar: {steps}",
    "analysis.before": "Önce",
    "analysis.after": "Sonra",
    "analysis.original_header": "📷 Orijinal Görüntü",
    "analysis.original_caption": "Yüklenen Mars görüntüsü",
    "analysis.analyze_btn": "🔍 Hibrit Analiz Et",
    "analysis.spinner": "Hibrit analiz yapılıyor (Anomali + Derinlik + Dinamik Değer)...",
    "analysis.reconstructed_header": "🔄 Yeniden Oluşturulan Görüntü",
    "analysis.anomaly_caption": "Anomali Skoru: {score:.6f}",
    "analysis.results_header": "📊 Hibrit Analiz Sonuçları",
    "analysis.metric.anomaly_mse": "Anomali Skoru (MSE)",
    "analysis.metric.combined": "Birleşik Anomali",
    "analysis.metric.combined_na": "N/A",
    "analysis.metric.anomaly_status": "Anomali Durumu",
    "analysis.status.anomaly": "🚨 Anormal",
    "analysis.status.normal": "✅ Normal",
    "analysis.metric.known_value": "Bilinen Değer",
    "analysis.metric.curiosity": "İlginçlik Puanı",
    "analysis.metric.predicted_class": "Tahmin Edilen Sınıf",
    "analysis.diff_header": "🔍 Fark ve Birleşik Anomali Haritası",
    "plot.original": "Orijinal",
    "plot.reconstructed": "Yeniden Oluşturulan",
    "plot.difference": "Fark",
    "plot.combined_overlay": "Birleşik Anomali (bindirme)",
    "plot.difference_anomaly": "Fark (Anomali)",
    "analysis.diag_header": "🔎 Tespit Tanılama Paneli",
    "analysis.diag_help": (
        "- **sc**: Birleşik anomali skoru\n"
        "- **e**: Kenar yoğunluğu göstergesi\n"
        "- **s**: Gölge/karanlık etkisi (azaltım)\n"
        "- **sp**: Parlama (speküler) etkisi (azaltım)\n"
        "- **lv**: Düşük doku/varians etkisi (azaltım)"
    ),
    "analysis.quick_select": "Hızlı Seçim",
    "analysis.quick_all": "Tümü",
    "analysis.detections_caption": "Birleşik Anomali Tespitleri",
    "analysis.detections_none_suffix": " — tespit bulunamadı",
    "analysis.detections_small_objects": " (küçük cisimler dahil edilir)",
    "analysis.no_detections": "Tespit bulunamadı veya tanılama verisi yok.",
    "analysis.focus_tile": "Odak: #{idx}",
    "analysis.recommendations_header": "💡 Hibrit Öneriler",
    "analysis.reco.high": "🎯 **YÜKSEK ÖNCELİK**: Bu hedef hem anormal hem de yüksek bilimsel değere sahip!",
    "analysis.reco.medium": "🔍 **ORTA ÖNCELİK**: Bu hedef anormal ama bilimsel değeri orta seviyede.",
    "analysis.reco.low_known": "📋 **DÜŞÜK ÖNCELİK**: Bu hedef normal ama bilinen değerli hedeflere benziyor.",
    "analysis.reco.low_normal": "📋 **DÜŞÜK ÖNCELİK**: Bu hedef normal Mars yüzeyi görünüyor.",
    "analysis.anomaly_calc_error": "❌ Anomali hesaplama hatası: {error}",
    "analysis.known_value_error": "⚠️ Bilinen değer hesaplama hatası: {error}",

    # analysis classes
    "analysis.class.negligible": "Değersiz",
    "analysis.class.low": "Düşük",
    "analysis.class.medium": "Orta",
    "analysis.class.medium_high": "Orta-Yüksek",
    "analysis.class.high": "Yüksek",
    "analysis.class.unknown": "Bilinmiyor",

    # categories
    "category.rocky": "Kayalık",
    "category.boulder": "Boulder",
    "category.flat_terrain": "Düz Arazi",
    "category.hills_or_ridge": "Tepe/Sırt",
    "category.dusty": "Tozlu",
    "category.rover": "Rover",

    # Tab2 depth
    "depth.map_header": "🌊 Derinlik Haritası ({model_type}) - {quality}",
    "depth.resolution": "Çözünürlük",
    "depth.resolution_help": "Giriş görüntüsünün analizde kullanılacak çözünürlüğü",
    "depth.apply_enhancement": "Geliştirme Uygula (kontrast + keskinleştirme)",
    "depth.raw_compare": "Ham çıktıyla karşılaştır",
    "depth.raw_compare_help": "Geliştirme kapalı (ham) ve açık çıktıları yan yana göster",
    "depth.original_caption": "Orijinal Görüntü",
    "depth.map_title": "Geliştirilmiş Derinlik Haritası",
    "depth.colorbar_label": "Derinlik (0=Yakın, 1=Uzak)",
    "depth.raw_output": "Ham DPT Çıkışı",
    "depth.enhanced_output": "Geliştirme Uygulandı",
    "depth.enhancement_off": "Geliştirme Kapalı",
    "depth.summary": "📊 **Derinlik Analizi ({model_type})**: {width}x{height} çözünürlük, Kontrast: {contrast:.3f}, Ortalama Derinlik: {mean:.3f}, Süre: {ms:.1f} ms",
    "depth.tuning_expander": "🔧 Derinlik İnce Ayar (Gelişmiş)",
    "depth.apply_tuning_btn": "Uygula (Derinlik İyileştirmeyi Güncelle)",
    "depth.tuning_applied": "İnce ayar parametreleri güncellendi. 'Derinlik Analizi' bölümünü tekrar çalıştırın.",
    "depth.viz_options": "🎨 Derinlik Görselleştirme Seçenekleri",
    "depth.colormap": "Colormap Seçin:",
    "depth.features_header": "📊 Geliştirilmiş Derinlik Özellikleri",
    "depth.metric.mean": "🌊 Ortalama Derinlik",
    "depth.metric.std": "📏 Derinlik Std",
    "depth.metric.variance": "📊 Derinlik Varyansı",
    "depth.metric.min": "⬇️ Min Derinlik",
    "depth.metric.max": "⬆️ Max Derinlik",
    "depth.metric.median": "📈 Derinlik Medyan",
    "depth.metric.complexity": "🏔️ Yüzey Karmaşıklığı",
    "depth.metric.grad_mean": "🌊 Gradient Ortalama",
    "depth.metric.grad_std": "📐 Gradient Std",
    "depth.metric.skewness": "📊 Skewness",
    "depth.metric.kurtosis": "📈 Kurtosis",
    "depth.metric.p75_p25": "🎯 P75-P25",
    "depth.metadata_header": "📋 Derinlik Metadata",
    "depth.distribution_header": "📊 Derinlik Dağılımı",
    "depth.quality_header": "🎯 Derinlik Kalitesi Değerlendirmesi",
    "depth.contrast.high": "✅ **Yüksek Kontrast**: {value:.3f}",
    "depth.contrast.medium": "⚠️ **Orta Kontrast**: {value:.3f}",
    "depth.contrast.low": "❌ **Düşük Kontrast**: {value:.3f}",
    "depth.range.wide": "✅ **Geniş Derinlik Aralığı**: {value:.3f}",
    "depth.range.medium": "⚠️ **Orta Derinlik Aralığı**: {value:.3f}",
    "depth.range.narrow": "❌ **Dar Derinlik Aralığı**: {value:.3f}",
    "depth.surface.smooth": "✅ **Yumuşak Yüzey**: {value:.3f}",
    "depth.surface.medium": "⚠️ **Orta Yüzey**: {value:.3f}",
    "depth.surface.rough": "❌ **Karmaşık Yüzey**: {value:.3f}",
    "depth.analysis_error": "❌ Derinlik analizi hatası: {error}",
    "depth.upload_first": "📸 Derinlik analizi için önce bir görüntü yükleyin.",
    "depth.load_failed": "Derinlik tahmin modülü kullanılamıyor.",
    "depth.plot.histogram_title": "Derinlik Histogramı",
    "depth.plot.depth_value": "Derinlik Değeri",
    "depth.plot.frequency": "Frekans",
    "depth.plot.surface_3d": "3B Derinlik Yüzeyi (Örnek)",
    "depth.plot.contour_2d": "2B Derinlik Konturu (Yedek)",
    "depth.plot.axis_x": "X",
    "depth.plot.axis_y": "Y",
    "depth.plot.axis_z": "Derinlik",

    # Tab3 system
    "system.model_info": "🤖 Hibrit Model Bilgileri",
    "system.ae_params": "Autoencoder Parametreleri",
    "system.ae_size": "Autoencoder Boyutu",
    "system.latent_size": "Latent Boyutu",
    "system.clf_params": "Sınıflandırıcı Parametreleri",
    "system.clf_size": "Sınıflandırıcı Boyutu",
    "system.class_count": "Sınıf Sayısı",
    "system.training_data": "📈 Eğitim Verisi",
    "system.total_images": "Toplam Görüntü",
    "system.category_count": "Kategori Sayısı",
    "system.pie_title": "Kategori Dağılımı",

    # Tab4 demo
    "system.test_images": "📸 Test Görüntüleri",
    "demo.analyze_btn": "🔍 {category} Hibrit Analiz",
    "demo.spinner": "{category} hibrit analiz ediliyor...",
    "demo.anomaly_result": "Anomali: {score:.6f}",
    "demo.known_result": "Bilinen Değer: {score:.3f}",
    "demo.curiosity_metric": "İlginçlik Puanı",

    # about.markdown
    "about.markdown": """
## 🚀 ARTPS - Otonom Bilimsel Keşif Sistemi (Hibrit)

**ARTPS (Autonomous Rover Target Prioritization System)**, Mars rover'larının
Dünya'dan komut beklemeden bilimsel olarak ilginç hedefleri tespit etmesini
sağlayan **hibrit yapay zeka sistemidir**.

**🛰️ Yapım:** [Poyraz BAYDEMİR](https://github.com/Poyqraz) · [ResearchGate DOI](http://dx.doi.org/10.13140/RG.2.2.12215.18088)
**📄 Lisans:** [MIT License](https://github.com/Poyqraz/ARTPS/blob/main/LICENSE)

### 🎯 Sistem Amacı
- Mars yüzeyinde bilimsel olarak değerli hedefleri otonom olarak tespit etmek
- **Derinlik algısı** ile 3D analiz yapmak
- **Dinamik "Bilinen Değer"** puanı hesaplamak
- Hedefleri öncelik sırasına göre sıralamak
- Rover'ın verimliliğini artırmak

### 🔬 Hibrit Teknik Özellikler (Güncel)
- **Convolutional Autoencoder**: Anomali tespiti (optimize 17M param.)
- **Derinlik Geliştirilmiş Sınıflandırıcı**: Dinamik değer (RGB latent + 14 derinlik öz.)
- **DPT_Large Derinlik Tahmini**: Yüksek doğruluk (CUDA hızlandırmalı)
- **PaDiM (Patch Distribution Modeling)**: Görüntü tabanlı anomaliyi AE+Derinlik ile füzyon
- **Çok Ölçekli İnce Detay**: Laplacian(3,5) + DoG ile küçük taş/kum çizgisi vurgusu
- **Uzak Alan Hassasiyeti**: Yakınlık karışımı ve derinliğe koşullu alan eşiği
- **Curiosity Verileri**: ~2,575 görüntü (train/valid)
- **Odağa Yumuşak Maske**: Seçili hedef çevresinde Gauss geçişli vurgulama

### 📊 Gelişmiş İlginçlik Puanı
```
İlginçlik Puanı = α × Dinamik Bilinen Değer + β × Anomali Skoru
```

- **α (Alfa)**: Dinamik bilinen değer ağırlığı (0-1)
- **β (Beta)**: Anomali/keşif ağırlığı (0-1)
- **Dinamik Bilinen Değer**: Kategori bazlı otomatik etiketleme (0-1)

### 🌊 Derinlik Analizi (Güncel)
- **DPT_Large**: Yüksek doğruluklu monocular depth, rehberli iyileştirme ve filtreleme
- **14 Derinlik Özelliği**: Ortalama, std, min, max, yüzey karmaşıklığı, gradient vb.
- **Uzak/ Yakın Denge**: Uzak alanlarda küçük detayları korumak için eşik uyarlama
- **3D/2D Görselleştirme**: Turbo colormap, 3D yüzey, histogram ve istatistikler

### 🎮 Hibrit Kullanım
1. Mars görüntüsü yükleyin
2. Parametreleri ayarlayın (α, β)
3. "Hibrit Analiz Et" butonuna basın
4. Anomali + Derinlik + Dinamik Değer sonuçlarını inceleyin

### 🔍 Gelişmiş Anomali Tespiti
- **Düşük MSE**: Normal Mars yüzeyi
- **Yüksek MSE**: Anormal/ilginç hedef
- **Derinlik Entegrasyonu**: 3D anomali tespiti
- **Dinamik Sınıflandırma**: Otomatik kategori belirleme

### 📈 Hibrit Model Performansı
- **Anomali Tespiti**: %95+ doğruluk
- **Sınıflandırma**: %74 doğruluk
- **Derinlik Tahmini**: DPT_Large (Yüksek Doğruluk) + Fallback
- **Gerçek Zamanlı**: <1 saniye analiz süresi

### 🚀 Gelecek Geliştirmeler
- Perseverance verileri entegrasyonu
- Gelişmiş segmentasyon algoritmaları
- Stereo vision entegrasyonu
- Gerçek zamanlı rover entegrasyonu
- Çoklu rover desteği
- Uzay istasyonu entegrasyonu
""",
}
