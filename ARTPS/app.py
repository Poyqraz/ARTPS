"""
ARTPS - Otonom Bilimsel Keşif Sistemi
Streamlit Web Arayüzü (Hibrit Model - Derinlik + Dinamik Değer)
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting için gerekli
from PIL import Image
import os
from pathlib import Path
from src.models.optimized_autoencoder import OptimizedAutoencoder
from src.models.depth_enhanced_classifier import DepthEnhancedClassifier
from src.models.anomaly import PaDiM, PaDiMConfig, PatchCore, PatchCoreConfig

# Transformers'ın TensorFlow'u içe aktarmasını engelle (NumPy 2.x ile çakışmaları azaltır)
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.models.depth_estimation import MiDaSDepthEstimator
from src.core import CuriosityScorer, CuriosityWeights
from src.utils.image_enhancement import enhance_image_auto
import plotly.express as px
import plotly.graph_objects as go
import cv2
import time
import json

# Matplotlib font ayarları - emoji uyarılarını önlemek için
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Font ayarları - sadece mevcut fontları kullan
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['figure.max_open_warning'] = 0

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="ARTPS - Otonom Bilimsel Keşif Sistemi",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sahiplik/katkı rozeti (sidebar üstü)
st.sidebar.caption("🛰️ Yapım: [Poyraz BAYDEMİR](https://github.com/Poyqraz) · [ResearchGate DOI](http://dx.doi.org/10.13140/RG.2.2.12215.18088)")
st.sidebar.caption("📄 Lisans: [MIT License](https://github.com/Poyqraz/ARTPS/blob/main/LICENSE)")

@st.cache_resource
def load_models():
    """Eğitilen modelleri yükle (cache'li) - GPU Optimizasyonu"""
    
    # GPU kontrolü
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.info(f"🖥️ Kullanılan cihaz: {device}")
    
    models = {}
    
    # 1. Autoencoder modeli
    autoencoder_path = "results/optimized_autoencoder_curiosity_extended.pth"
    if os.path.exists(autoencoder_path):
        autoencoder = OptimizedAutoencoder(input_channels=3, latent_dim=1024)
        checkpoint = torch.load(autoencoder_path, map_location=device, weights_only=True)
        autoencoder.load_state_dict(checkpoint['model_state_dict'])
        autoencoder.to(device)  # GPU'ya taşı
        autoencoder.eval()
        models['autoencoder'] = autoencoder
        models['device'] = device
    else:
        st.error(f"❌ Autoencoder model bulunamadı: {autoencoder_path}")
        return None
    
    # 2. Derinlik geliştirilmiş sınıflandırıcı
    classifier_path = "results/depth_enhanced_classifier.pth"
    if os.path.exists(classifier_path):
        classifier = DepthEnhancedClassifier(num_classes=5, rgb_features=1024, depth_features=14)
        checkpoint = torch.load(classifier_path, map_location=device, weights_only=True)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier.to(device)  # GPU'ya taşı
        classifier.eval()
        models['classifier'] = classifier
    else:
        st.warning("⚠️ Sınıflandırıcı model bulunamadı, sadece anomali tespiti kullanılacak")
    
    # 2. PaDiM anomali modeli (opsiyonel)
    try:
        padim_stats = "results/padim_stats.pth"
        padim = PaDiM(PaDiMConfig(image_size=256))
        if Path(padim_stats).exists():
            padim.load(padim_stats)
            models['padim'] = padim
        else:
            st.warning("⚠️ PaDiM istatistikleri bulunamadı: results/padim_stats.pth. Sadece AE tabanlı anomali kullanılacak")
    except Exception as e:
        st.warning(f"⚠️ PaDiM yüklenemedi: {e}")

    # 2b. PatchCore (opsiyonel)
    try:
        patchcore_bank = "results/patchcore_bank.pth"
        if Path(patchcore_bank).exists():
            pcore = PatchCore(PatchCoreConfig(image_size=256))
            pcore.load(patchcore_bank)
            models['patchcore'] = pcore
        else:
            st.info("ℹ️ PatchCore bellek bankası bulunamadı (tools/prepare_patchcore_bank.py ile üretebilirsiniz)")
    except Exception as e:
        st.warning(f"⚠️ PatchCore yüklenemedi: {e}")

    # 3. Derinlik tahmin modülü (gerçek durumu kontrol et)
    try:
        depth_estimator = MiDaSDepthEstimator(model_type="DPT_Large", device=device)
        
        # Gerçek model durumunu kontrol et
        model_params = sum(p.numel() for p in depth_estimator.model.parameters())
        is_real_dpt = model_params > 100_000_000  # DPT_Large ~345M parametre
        
        if is_real_dpt:
            st.success(f"✅ DPT_Large modeli başarıyla yüklendi (yüksek doğruluk) - {model_params:,} parametre")
        else:
            st.warning(f"⚠️ DPT_Large modeli yüklenemedi, basit model kullanılıyor - {model_params:,} parametre")
            st.info("ℹ️ PyTorch Hub bağlantı sorunu nedeniyle fallback model aktif")
        
        models['depth_estimator'] = depth_estimator
        models['depth_model_info'] = {
            'is_real_dpt': is_real_dpt,
            'param_count': model_params,
            'model_type': depth_estimator.model_type
        }
        
    except Exception as e:
        st.error(f"❌ Derinlik tahmin modülü yüklenemedi: {e}")
    
    # 4. Curiosity skorlayıcı (UI ağırlıklarını daha sonra güncelleyeceğiz)
    models['curiosity_scorer'] = CuriosityScorer(CuriosityWeights())
    # Otomatik: varsa öğrenilmiş ağırlıkları yükle (bozmadan, sessiz fallback)
    try:
        wpath = Path("results/curiosity_weights.json")
        if wpath.exists():
            with open(wpath, 'r', encoding='utf-8') as f:
                wdata = json.load(f)
            models['curiosity_scorer'] = CuriosityScorer(CuriosityWeights(**wdata))
            st.info("🧭 Curiosity ağırlıkları otomatik yüklendi (results/curiosity_weights.json)")
    except Exception as e:
        st.warning(f"Curiosity ağırlıkları yüklenemedi: {e}")
    return models

def calculate_anomaly_score(autoencoder, image, device):
    """Görüntü için anomali skoru hesapla - GPU Optimizasyonu"""
    
    try:
        # Görüntüyü işle
        image = image.resize((128, 128), Image.LANCZOS)
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Tensor'a çevir ve GPU'ya taşı
        input_tensor = torch.from_numpy(image_array).float()
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Model tahmini (AMP ile hızlandırma)
        with torch.no_grad():
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    reconstructed, latent = autoencoder(input_tensor)
            else:
                reconstructed, latent = autoencoder(input_tensor)
        
        # CPU'ya geri taşı ve numpy'a çevir
        reconstructed = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
        latent = latent.squeeze().cpu().numpy()
        
        # MSE hesapla (anomali skoru)
        mse = np.mean((image_array - reconstructed) ** 2)
        
        return mse, image_array, reconstructed, latent
        
    except Exception as e:
        st.error(f"❌ Anomali hesaplama hatası: {e}")
        return None, None, None, None

def _normalize_map(values: np.ndarray) -> np.ndarray:
    """Harita/yoğunluk matrisini yüzde 2-98 aralığına göre normalize eder (0-1)."""
    arr = values.astype(np.float32)
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    norm = (arr - lo) / (hi - lo)
    return np.clip(norm, 0.0, 1.0)

def _unsharp_image(img: np.ndarray, amount: float = 0.6, radius: float = 2.0) -> np.ndarray:
    """Basit unsharp mask ile keskinleştirme (uint8 RGB bekler)."""
    try:
        blur = cv2.GaussianBlur(img, (0, 0), radius)
        sharp = cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)
        return sharp
    except Exception:
        return img

def _safe_imshow(ax, img: np.ndarray, **kwargs) -> None:
    """Matplotlib için güvenli görüntü çizimi: dtype ve aralığı normalize eder.

    - uint8/int32 gibi tiplere karşı dayanıklı
    - float ise [0,1] aralığına sıkıştırır
    """
    try:
        arr = img
        if isinstance(arr, np.ndarray):
            if arr.dtype not in (np.uint8, np.float32, np.float64, np.int16):
                arr = arr.astype(np.float32)
            if arr.dtype in (np.float32, np.float64):
                # Büyük olasılıkla [0,1] olmalı
                if arr.max() > 1.0:
                    arr = np.clip(arr / 255.0, 0.0, 1.0)
                else:
                    arr = np.clip(arr, 0.0, 1.0)
            elif arr.dtype == np.uint8:
                # Matplotlib direkt destekler
                pass
            elif arr.dtype == np.int16:
                # Kısa tip: normalize et
                arr = np.clip(arr.astype(np.float32) / 255.0, 0.0, 1.0)
        ax.imshow(arr, **kwargs)
    except Exception:
        # Son çare: gri göster
        ax.imshow(np.zeros((10, 10), dtype=np.float32), cmap='gray')

def _auto_enhance_focus(img_rgb: np.ndarray, scale: float, interp_code: int, amount: float) -> np.ndarray:
    """Odak kırpım için otomatik kalite artırma: CLAHE + hafif bilateral + (isteğe bağlı) upsample + unsharp.

    img_rgb: uint8 RGB
    scale: 1.0, 1.5, 2.0 ...
    interp_code: cv2.INTER_*
    amount: unsharp miktarı
    """
    try:
        # Kontrast: LAB'ta CLAHE (L kanalı)
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        L, A, B = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        L2 = clahe.apply(L)
        lab2 = cv2.merge([L2, A, B])
        img_rgb = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
        # Gürültü azaltım: hafif bilateral (detayı koru)
        img_rgb = cv2.bilateralFilter(img_rgb, d=3, sigmaColor=25, sigmaSpace=25)
        # İsteğe bağlı upsample (keskinleştirme ÖNDEN değil, SONRADAN uygulanır)
        if float(scale) > 1.0:
            ih, iw = img_rgb.shape[:2]
            img_rgb = cv2.resize(img_rgb, (int(iw * scale), int(ih * scale)), interpolation=interp_code)
        # Keskinleştirme (son adım)
        img_rgb = _unsharp_image(img_rgb, amount=max(0.0, float(amount)), radius=1.6)
    except Exception:
        pass
    return img_rgb

def _precompute_focus_tiles(results: dict, detections: list) -> list:
    """Seçim gecikmesini azaltmak için odak karolarını önceden üretir."""
    try:
        tiles = []
        comb_map = results.get('combined_anomaly_map')
        if comb_map is None or len(detections) == 0:
            return tiles
        H, W = comb_map.shape[:2]
        base = (results['original'] * 255).astype(np.uint8)
        if base.shape[:2] != (H, W):
            base = cv2.resize(base, (W, H), interpolation=cv2.INTER_LINEAR)
        if base.ndim == 2:
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
        heat_full = (plt.cm.inferno(comb_map)[..., :3] * 255).astype(np.uint8)
        depth_full = results.get('depth_map_full')

        # Odak ayarları
        h_target = int(globals().get('focus_h', 300))
        overlay_mode = bool(globals().get('focus_overlay', True))
        sharpen = bool(globals().get('focus_sharpen', True))
        hide_empty_depth = bool(globals().get('focus_hide_empty_depth', True))
        interp_name = str(globals().get('focus_interp', 'INTER_LANCZOS4'))
        interp = getattr(cv2, interp_name, cv2.INTER_LANCZOS4)

        for det in detections:
            try:
                x, y, w, h = det['x'], det['y'], det['w'], det['h']
                pad = int(0.15 * max(w, h))
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(W, x + w + pad)
                y2 = min(H, y + h + pad)
                if (y2 - y1) <= 5 or (x2 - x1) <= 5:
                    tiles.append(None)
                    continue
                # Orijinal kırpım
                raw_crop = cv2.cvtColor(base[y1:y2, x1:x2].copy(), cv2.COLOR_BGR2RGB)
                raw_crop = _auto_enhance_focus(raw_crop, scale=1.0, interp_code=interp, amount=0.6)
                # Isı kapağı
                heat_u8 = heat_full[y1:y2, x1:x2].copy()
                if overlay_mode:
                    if heat_u8.shape[:2] != raw_crop.shape[:2]:
                        heat_u8 = cv2.resize(heat_u8, (raw_crop.shape[1], raw_crop.shape[0]), interpolation=interp)
                    heat_crop = cv2.addWeighted(raw_crop, 0.25, heat_u8, 0.75, 0)
                else:
                    heat_crop = heat_u8
                # Derinlik kenarı karo
                if depth_full is not None and depth_full.shape[:2] == comb_map.shape[:2]:
                    dpatch = depth_full[y1:y2, x1:x2].astype(np.float32)
                    gx = cv2.Sobel(dpatch, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(dpatch, cv2.CV_32F, 0, 1, ksize=3)
                    mag = np.sqrt(gx * gx + gy * gy)
                    mag = (mag - mag.min()) / (mag.ptp() + 1e-6)
                    depth_edge_crop = (plt.cm.cividis(mag)[..., :3] * 255).astype(np.uint8)
                else:
                    depth_edge_crop = None
                # Keskinleştirme
                if sharpen:
                    heat_crop = _unsharp_image(heat_crop, 0.6, 2)
                    if depth_edge_crop is not None:
                        depth_edge_crop = _unsharp_image(depth_edge_crop, 0.6, 2)
                # H yüksekliğine yeniden örnekle
                def _resize_h(img):
                    ih, iw = img.shape[:2]
                    nw = int(iw * (h_target / max(1, ih)))
                    return cv2.resize(img, (nw, h_target), interpolation=interp)
                # Tespit kutusunu işaretle
                try:
                    cv2.rectangle(raw_crop, (max(0, pad), max(0, pad)), (max(0, pad) + w, max(0, pad) + h), (240, 220, 0), 1)
                except Exception:
                    pass
                parts = [_resize_h(raw_crop), _resize_h(heat_crop)]
                if depth_edge_crop is not None or not hide_empty_depth:
                    if depth_edge_crop is None:
                        depth_edge_crop = np.zeros_like(heat_crop)
                    parts.append(_resize_h(depth_edge_crop))
                tile = np.concatenate(parts, axis=1)
                tiles.append(tile)
            except Exception:
                tiles.append(None)
        return tiles
    except Exception:
        return []

def compute_combined_anomaly_map(
    original_rgb: np.ndarray,
    reconstructed_rgb: np.ndarray,
    depth_map: np.ndarray,
    *,
    hyst_high_pct: int = 97,
    hyst_low_pct: int = 92,
    nms_iou: float = 0.35,
    top_k: int = 25,
    w_recon: float = 0.50,
    w_depth: float = 0.30,
    w_texture: float = 0.20,
    edge_reinforce: float = 0.35,
):
    """Rekonstrüksiyon farkı + derinlik süreksizliği + gölge/kenar farkındalığı birleşik haritası.

    Ek olarak kenar rehberli yeniden keskinleştirme ve mühendislik odaklı kutulama uygular.

    Döndürür: (combined_map[H,W] in 0..1, detections[list of dict])
    """
    # Hedef çözünürlüğü derinlik haritası boyutu
    H, W = depth_map.shape[:2]
    orig = cv2.resize(original_rgb.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    recon = cv2.resize(reconstructed_rgb.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)

    # Rekonstrüksiyon farkı (MSE kanal başına)
    recon_diff = ((orig - recon) ** 2).mean(axis=2)
    recon_diff_n = _normalize_map(recon_diff)

    # Görüntü gri, kenar/kontrast ve gölge göstergesi
    img_u8 = (orig * 255.0).astype(np.uint8)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    Hc, Sc, Vc = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    grad_mag_n = _normalize_map(grad_mag)
    shadow_n = _normalize_map(1.0 - gray)  # koyu bölgeler yüksek

    # Derinlik süreksizliği ve yakınlık ağırlığı
    depth = depth_map.astype(np.float32)
    depth_n_for_region = _normalize_map(depth)
    dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    depth_edge = np.sqrt(dx ** 2 + dy ** 2)
    depth_edge_n = _normalize_map(depth_edge)
    proximity_w = _normalize_map(1.0 - depth)  # yakın bölgeler yüksek ağırlık

    # Derinlik Laplacian (çöküntü/çıkıntı vurgusu)
    depth_lap = cv2.Laplacian(depth, cv2.CV_32F, ksize=3)
    depth_lap_n = _normalize_map(np.abs(depth_lap))

    # Birleşik skor (ayarlanabilir ağırlıklar)
    # Not: Gölge bölgeleri sahte anomaliye yol açabildiğinden, texture_term
    # doğrudan gölgeyi yükseltmek yerine kenar ağırlıklı tutulur.
    texture_term = 0.35 * shadow_n + 0.65 * grad_mag_n
    # Laplacian katkısı UI'dan gelebilir; yoksa 0.08 varsay
    w_lap = float(globals().get('w_lap', 0.08))
    # İnce detay vurgusu (küçük taş, kum hatları için): çok ölçekli Laplacian + DoG
    try:
        lap3 = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        lap5 = cv2.Laplacian(gray, cv2.CV_32F, ksize=5)
        dog = cv2.GaussianBlur(gray, (0, 0), 0.8) - cv2.GaussianBlur(gray, (0, 0), 1.6)
        fine_detail = _normalize_map(np.abs(lap3) + 0.6 * np.abs(lap5) + 0.8 * np.abs(dog))
    except Exception:
        fine_detail = np.zeros_like(recon_diff_n)

    w_detail = float(globals().get('w_detail', 0.12))
    raw_combined = (
        w_recon * recon_diff_n
        + w_depth * depth_edge_n
        + w_texture * texture_term
        + w_lap * depth_lap_n
        + w_detail * fine_detail
    )
    # Yakınlık ağırlığı: uzak alanları tamamen bastırmamak için karışım uygula
    proximity_mix = 0.65 * proximity_w + 0.35 * (1.0 - proximity_w)
    combined = np.clip(raw_combined * (0.5 + 0.5 * proximity_mix), 0.0, 1.0)

    # Gölge bastırma: (koyu) AND (düşük görüntü gradyanı) AND (düşük derinlik kenarı)
    # ve aydınlatma-kenar etkisi azaltımı: görüntü kenarı yüksek ama derinlik kenarı düşükse etkisini düşür.
    try:
        illumination_edge = np.clip(grad_mag_n - depth_edge_n, 0.0, 1.0)
        shadow_like = np.clip(shadow_n * (1.0 - grad_mag_n) * (1.0 - depth_edge_n), 0.0, 1.0)
        shadow_like = cv2.GaussianBlur(shadow_like, (5, 5), 0)
        # Speküler/parlak nokta maskesi: yüksek V, düşük S ve düşük kenar
        spec_mask = np.clip(Vc * (1.0 - Sc) * (1.0 - grad_mag_n) * (1.0 - depth_edge_n), 0.0, 1.0)
        spec_mask = cv2.GaussianBlur(spec_mask, (3, 3), 0)
        # Düşük doku (varyans) haritası: küçük pencere varyansı
        gray_f32 = gray.astype(np.float32)
        k = 5
        mean = cv2.boxFilter(gray_f32, ddepth=-1, ksize=(k, k), normalize=True)
        mean_sq = cv2.boxFilter(gray_f32 * gray_f32, ddepth=-1, ksize=(k, k), normalize=True)
        variance = np.clip(mean_sq - mean * mean, 0.0, 1.0)
        var_norm = variance / max(variance.max(), 1e-6)

        # Saha ayarlı katsayılar
        alpha_shad = float(globals().get('alpha_shad', 0.65))
        beta_illum = float(globals().get('beta_illum', 0.25))
        spec_gamma = float(globals().get('spec_gamma', 0.35))
        spec_lowvar_gamma = float(globals().get('spec_lowvar_gamma', 0.35))
        spec_var_thresh = float(globals().get('spec_var_thresh', 0.005))
        # Düşük varyans bölgeleri için ek azaltım (speküler düz alanlar)
        lowvar_mask = (var_norm < spec_var_thresh).astype(np.float32)
        lowvar_mask = cv2.GaussianBlur(lowvar_mask, (3, 3), 0)
        combined = np.clip(
            combined * (1.0 - alpha_shad * shadow_like)
            - beta_illum * illumination_edge
            - spec_gamma * spec_mask
            - spec_lowvar_gamma * lowvar_mask,
            0.0,
            1.0,
        )
    except Exception:
        pass

    # Kenar rehberli yeniden keskinleştirme (overlay ve kutu netliği için)
    try:
        guide_u8 = (orig * 255.0).astype(np.uint8)
        guide_gray = cv2.cvtColor(guide_u8, cv2.COLOR_RGB2GRAY)
        if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'jointBilateralFilter'):
            joint = cv2.ximgproc.jointBilateralFilter(guide_gray, (combined * 255).astype(np.uint8), d=9, sigmaColor=25, sigmaSpace=25)
            combined = joint.astype(np.float32) / 255.0
        else:
            combined = cv2.bilateralFilter((combined * 255).astype(np.uint8), d=9, sigmaColor=25, sigmaSpace=25).astype(np.float32) / 255.0
        # Guided filter ile hizalama (varsa)
        if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'guidedFilter'):
            gf = cv2.ximgproc.guidedFilter(guide_u8, (combined * 255).astype(np.uint8), radius=8, eps=1e-2)
            combined = gf.astype(np.float32) / 255.0
        # Unsharp mask + kenar vurgusu
        edges = cv2.Canny(guide_gray, 50, 150).astype(np.float32) / 255.0
        combined = np.clip(combined + edge_reinforce * (edges * (combined - cv2.GaussianBlur(combined, (0, 0), 1.0))), 0.0, 1.0)
    except Exception:
        combined = cv2.GaussianBlur(combined, (3, 3), 0.0)

    # Histerezis eşikleme ile aday bölgeler (seed-grow): daha sağlam tespit
    high_th = float(np.percentile(combined, hyst_high_pct))
    low_th = float(np.percentile(combined, hyst_low_pct))
    high_mask = (combined >= high_th).astype(np.uint8)
    low_mask = (combined >= low_th).astype(np.uint8)

    # Seed'leri düşük eşik alanında genişlet (yaklaşık morfolojik rekonstrüksiyon)
    kernel = np.ones((3, 3), np.uint8)
    prev = np.zeros_like(high_mask)
    seeds = (high_mask * 255).astype(np.uint8)
    low = (low_mask * 255).astype(np.uint8)
    for _ in range(10):
        dil = cv2.dilate(seeds, kernel, iterations=1)
        seeds = cv2.bitwise_and(dil, low)
        if np.array_equal(seeds, prev):
            break
        prev = seeds.copy()
    mask = seeds
    # Temizleme ve doldurma
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    area_min_pct = float(globals().get('min_area_pct', 0.10)) / 100.0
    area_min = max(1.0, area_min_pct * H * W)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Bölge sınırlarını güvenli kırp
        y1c, y2c = max(0, y), min(H, y + h)
        x1c, x2c = max(0, x), min(W, x + w)
        # Uzak alanlar için daha küçük eşik: derinlik yüksekse alan eşiğini düşür
        region_depth_mean = float(np.mean(depth_n_for_region[y1c:y2c, x1c:x2c])) if (y2c > y1c and x2c > x1c) else 0.0
        local_area_min = area_min * (0.35 + 0.65 * (1.0 - region_depth_mean))
        if cv2.contourArea(cnt) < local_area_min:
            continue
        # Döndürülmüş dikdörtgen (daha sıkı kutulama)
        rect = cv2.minAreaRect(cnt)
        box_pts = cv2.boxPoints(rect)
        box_pts = np.int0(box_pts)
        # Bölge sınırlarını güvenli kırp
        y1, y2 = y1c, y2c
        x1, x2 = x1c, x2c
        region = combined[y1:y2, x1:x2]
        region_edges = grad_mag_n[y1:y2, x1:x2]
        region_shadow = shadow_like[y1:y2, x1:x2] if 'shadow_like' in locals() else None
        region_illum = illumination_edge[y1:y2, x1:x2] if 'illumination_edge' in locals() else None
        region_spec = spec_mask[y1:y2, x1:x2] if 'spec_mask' in locals() else None
        region_prox = proximity_w[y1:y2, x1:x2]
        # Yakınlık ortalaması
        prox_mean = float(np.mean(region_prox)) if region_prox.size else 0.0
        # Bölge skorları
        comb_mean = float(np.mean(region)) if region.size else 0.0
        edge_mean = float(np.mean(region_edges)) if region_edges.size else 0.0
        # Gölge ve aydınlatma-kenarı azaltımları
        shadow_pen = float(np.mean(region_shadow)) if (region_shadow is not None and region_shadow.size) else 0.0
        illum_pen = float(np.mean(region_illum)) if (region_illum is not None and region_illum.size) else 0.0
        spec_pen = float(np.mean(region_spec)) if (region_spec is not None and region_spec.size) else 0.0
        lowvar_pen = float(np.mean(lowvar_mask[y1:y2, x1:x2])) if 'lowvar_mask' in locals() else 0.0
        # Uzak alanlar için küçük ayrıntıları daha iyi puanlamak adına fine_detail katkısını ekle
        fine_local = float(np.mean(fine_detail[y1:y2, x1:x2])) if (y2 > y1 and x2 > x1) else 0.0
        score = 0.5 * comb_mean + 0.25 * edge_mean + 0.2 * prox_mean + 0.05 * fine_local - 0.35 * shadow_pen - 0.20 * illum_pen - 0.30 * spec_pen - 0.25 * lowvar_pen
        score = float(max(0.0, score))

        # Saf gölge veya speküler bölgeleri ele: saha ayarlı eşikler
        sh_cut = float(globals().get('shadow_cut', 0.45))
        im_edge_min = float(globals().get('img_edge_min', 0.10))
        dp_edge_min = float(globals().get('depth_edge_min', 0.08))
        sp_cut = float(globals().get('spec_cut', 0.50))
        if shadow_pen > sh_cut and edge_mean < im_edge_min and float(np.mean(depth_edge_n[y1:y2, x1:x2])) < dp_edge_min:
            continue
        if spec_pen > sp_cut and edge_mean < im_edge_min and float(np.mean(depth_edge_n[y1:y2, x1:x2])) < dp_edge_min:
            continue
        # Çok düşük doku + (düşük kenar) bölgeleri de ele (tek piksel parlamaları)
        if lowvar_pen > 0.6 and edge_mean < im_edge_min:
            continue
        detections.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "score": float(score),
            "poly": box_pts.tolist(),
            # açıklayıcı metrikler (debug/ince ayar için)
            "comb_mean": float(comb_mean),
            "edge_mean": float(edge_mean),
            "prox_mean": float(prox_mean),
            "shadow_pen": float(shadow_pen),
            "illum_pen": float(illum_pen),
            "spec_pen": float(spec_pen),
            "lowvar_pen": float(lowvar_pen) if 'lowvar_pen' in locals() else 0.0,
        })

    # Non-Maximum Suppression (IoU tabanlı) ile kutuları rafine et
    def _iou(a, b):
        ax1, ay1, aw, ah = a[0], a[1], a[2], a[3]
        bx1, by1, bw, bh = b[0], b[1], b[2], b[3]
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter + 1e-6
        return inter / union

    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []
    for det in detections:
        if all(_iou((det['x'], det['y'], det['w'], det['h']), (k['x'], k['y'], k['w'], k['h'])) < nms_iou for k in kept):
            kept.append(det)
    detections = kept[:max(1, int(top_k))]

    # Yakın kutuları birleştir (merkez yakın ve IoU düşükse tek kutu yap)
    try:
        miou = float(globals().get('merge_iou', 0.15))
        mtol = float(globals().get('merge_tol', 0.5))
        merged = []
        used = [False] * len(detections)
        diag = float(np.hypot(W, H))
        for i, a in enumerate(detections):
            if used[i]:
                continue
            axc = a['x'] + a['w'] / 2.0
            ayc = a['y'] + a['h'] / 2.0
            group = [i]
            for j, b in enumerate(detections[i + 1:], start=i + 1):
                if used[j]:
                    continue
                iou_ab = _iou((a['x'], a['y'], a['w'], a['h']), (b['x'], b['y'], b['w'], b['h']))
                bxc = b['x'] + b['w'] / 2.0
                byc = b['y'] + b['h'] / 2.0
                center_dist = np.hypot(axc - bxc, ayc - byc)
                if iou_ab < miou and center_dist < mtol * diag * 0.02:
                    group.append(j)
                    used[j] = True
            # Grupları tek kutuya birleştir
            xs = [detections[g]['x'] for g in group]
            ys = [detections[g]['y'] for g in group]
            ws = [detections[g]['w'] for g in group]
            hs = [detections[g]['h'] for g in group]
            x1 = int(min(xs))
            y1 = int(min(ys))
            x2 = int(max(xs[k] + ws[k] for k in range(len(xs))))
            y2 = int(max(ys[k] + hs[k] for k in range(len(ys))))
            region = combined[y1:y2, x1:x2]
            merged.append({
                'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1,
                'score': float(region.mean()) if region.size else a['score'],
                'poly': None
            })
            used[i] = True
        detections = merged
    except Exception:
        pass

    # Ufuk maskesi: derin ve düşük gradyan alanları (genelde üst kısım)
    try:
        horizon_mask = ((depth > 0.8) & (depth_edge_n < 0.05)).astype(np.uint8)
        # Ufuk bilgisi raporlama için; kombinasyondan çıkarmıyoruz ama metrik olabilir
    except Exception:
        horizon_mask = None

    return combined.astype(np.float32), detections

def calculate_known_value_score(classifier, depth_estimator, image_array, latent_features, device):
    """Dinamik bilinen değer skoru hesapla - GPU Optimizasyonu"""
    
    try:
        # Derinlik tahmini
        depth_map, depth_metadata = depth_estimator.estimate_depth(image_array)
        
        # Derinlik özelliklerini çıkar
        depth_features = depth_estimator.extract_depth_features(depth_map)
        # Eğitimde kullanılan 14 özellik dizilimi (sabit sıra)
        depth_feature_keys = [
            'depth_mean', 'depth_std', 'depth_min', 'depth_max',
            'depth_median', 'depth_percentile_25', 'depth_percentile_75',
            'depth_variance', 'depth_skewness', 'depth_kurtosis',
            'surface_complexity', 'depth_gradient_mean', 'depth_gradient_std', 'depth_gradient_max',
        ]
        depth_vec = [float(depth_features.get(k, 0.0)) for k in depth_feature_keys]
        depth_features_tensor = torch.tensor(depth_vec, dtype=torch.float32).unsqueeze(0).to(device)
        
        # RGB latent features
        rgb_features_tensor = torch.tensor(latent_features, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Sınıflandırma tahmini (AMP)
        with torch.no_grad():
            if device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    predictions = classifier(rgb_features_tensor, depth_features_tensor)
            else:
                predictions = classifier(rgb_features_tensor, depth_features_tensor)
            predicted_class = torch.argmax(predictions, dim=1).item()
            confidence = torch.max(predictions).item()
        
        # Sınıf değerlerini normalize et (0-1 arası)
        value_score = predicted_class / 4.0  # 0-4 arası sınıfları 0-1 arasına çevir
        
        return value_score, confidence, predicted_class, depth_map, depth_features
        
    except Exception as e:
        st.warning(f"⚠️ Bilinen değer hesaplama hatası: {e}")
        return 0.5, 0.0, 2, None, {}  # Fallback değerler

def analyze_mars_image(models, image):
    """Mars görüntüsünü kapsamlı analiz et - GPU Optimizasyonu"""
    
    # Son analiz sonuçlarını yeniden çalıştırmada kaybetmemek için session_state'ten çek
    results = st.session_state.get("results", {})
    device = models.get('device', torch.device('cpu'))
    
    # 1. Anomali skoru hesapla
    mse, original, reconstructed, latent = calculate_anomaly_score(models['autoencoder'], image, device)
    results['anomaly_score'] = mse
    results['original'] = original
    results['reconstructed'] = reconstructed
    results['latent'] = latent
    
    # 2. Bilinen değer skoru hesapla (hibrit model varsa)
    if 'classifier' in models and 'depth_estimator' in models:
        value_score, confidence, predicted_class, depth_map, depth_features = calculate_known_value_score(
            models['classifier'], models['depth_estimator'], original, latent, device
        )
        results['known_value_score'] = value_score
        results['confidence'] = confidence
        results['predicted_class'] = predicted_class
        results['depth_map'] = depth_map
        results['depth_features'] = depth_features
    else:
        # Fallback: Sabit değer
        results['known_value_score'] = 0.5
        results['confidence'] = 0.0
        results['predicted_class'] = 2
        results['depth_map'] = None
        results['depth_features'] = {}
    
    # 3. Derinlik mevcutsa, görüntü + derinlik tabanlı birleşik anomali haritası üret
    try:
        depth_map_for_fusion = None
        if 'depth_estimator' in models:
            depth_input_res = 768
            image_for_depth = np.array(image.resize((depth_input_res, depth_input_res), Image.LANCZOS), dtype=np.float32) / 255.0
            try:
                depth_map_for_fusion, _ = models['depth_estimator'].estimate_depth(
                    image_for_depth,
                    apply_enhancement=True,
                    high_detail=True,
                    tta_flips=True,
                    use_fgs=True,
                    use_wmf=True,
                )
            except Exception:
                depth_map_for_fusion = None

        # Derinlik başarısız olursa, gradient tabanlı sentetik derinlik üret (fallback)
        if depth_map_for_fusion is None:
            img_u8 = (results['original'] * 255.0).astype(np.uint8)
            gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            grad = np.sqrt(sx * sx + sy * sy)
            depth_map_for_fusion = _normalize_map(1.0 - grad)  # kenar alanları uzak, düz alanlar yakın
            results['depth_map_full'] = depth_map_for_fusion
        else:
            results['depth_map_full'] = depth_map_for_fusion

        # Birleşik anomali haritası hesapla (her durumda)
        # UI'dan ayarlar mevcutsa kullan; yoksa varsayılanlar
        cfg_hh = int(globals().get('hyst_high', 97))
        cfg_hl = int(globals().get('hyst_low', 92))
        cfg_nms = float(globals().get('nms_iou', 0.35))
        cfg_topk = int(globals().get('top_k', 25))
        cfg_wr = float(globals().get('w_recon', 0.50))
        cfg_wd = float(globals().get('w_depth', 0.30))
        cfg_wt = float(globals().get('w_texture', 0.20))
        cfg_er = float(globals().get('edge_reinf', 0.35))

        combined_map, detections = compute_combined_anomaly_map(
            results['original'], results['reconstructed'], depth_map_for_fusion,
            hyst_high_pct=cfg_hh, hyst_low_pct=cfg_hl, nms_iou=cfg_nms, top_k=cfg_topk,
            w_recon=cfg_wr, w_depth=cfg_wd, w_texture=cfg_wt, edge_reinforce=cfg_er
        )
        # PaDiM/PatchCore mevcutsa, haritaları yumuşak birleştir
        try:
            base_u8 = (results['original'] * 255).astype(np.uint8)
            if 'padim' in models:
                padim_map = models['padim'].predict_anomaly_map(base_u8)
                padim_w = float(globals().get('w_padim', 0.30))
                combined_map = np.clip((1.0 - padim_w) * combined_map + padim_w * padim_map, 0.0, 1.0)
            if 'patchcore' in models:
                pcore_map = models['patchcore'].predict_anomaly_map(base_u8)
                pcore_w = float(globals().get('w_patchcore', 0.25))
                combined_map = np.clip((1.0 - pcore_w) * combined_map + pcore_w * pcore_map, 0.0, 1.0)
        except Exception:
            pass
        results['combined_anomaly_map'] = combined_map
        results['combined_anomaly_score'] = float(combined_map.mean())
        results['detections'] = detections
        # Odak karo önbelleği: hızlı seçim gecikmesini azalt
        try:
            results['focus_tiles'] = _precompute_focus_tiles(results, detections)
        except Exception:
            results['focus_tiles'] = []
    except Exception:
        # Son çare: yalnızca fark haritasına dayalı basit tespit
        diff_only = ((results['original'] - results['reconstructed']) ** 2).mean(axis=2)
        diff_only = _normalize_map(diff_only)
        results['combined_anomaly_map'] = diff_only
        results['combined_anomaly_score'] = float(diff_only.mean())
        # Basit eşik + kontur
        th = float(np.percentile(diff_only, 97))
        mask = (diff_only >= th).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        H, W = diff_only.shape[:2]
        area_min = 0.001 * H * W
        for c in contours:
            if cv2.contourArea(c) < area_min:
                continue
            x, y, w, h = cv2.boundingRect(c)
            detections.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "score": float(diff_only[y:y+h, x:x+w].mean())})
        results['detections'] = detections
        try:
            results['focus_tiles'] = _precompute_focus_tiles(results, detections)
        except Exception:
            results['focus_tiles'] = []
    
    # 4. Curiosity skoru: tek yerden, seçilebilir bileşenlerle hesapla
    try:
        scorer = models.get('curiosity_scorer')
        if scorer is not None:
            # UI'dan ağırlıkları çek (globals, sidebar içinde set edildi)
            cw = CuriosityWeights(
                w_known=float(globals().get('alpha', 0.4)),
                w_anomaly=float(globals().get('beta', 0.6)),
                w_combined=float(globals().get('w_combined', 0.0)),
                w_depth_variance=float(globals().get('w_dvar', 0.0)),
                w_roughness=float(globals().get('w_rough', 0.0)),
            )
            models['curiosity_scorer'] = CuriosityScorer(cw)
            score, breakdown = models['curiosity_scorer'].compute(
                known_value_score=results.get('known_value_score'),
                anomaly_mse=results.get('anomaly_score'),
                combined_anomaly_score=results.get('combined_anomaly_score'),
                depth_features=results.get('depth_features'),
                reference_mse=float(globals().get('ref_mse', 0.003)),
            )
            results['curiosity_score'] = float(score)
            results['curiosity_breakdown'] = breakdown
    except Exception:
        pass

    return results

def main():
    """Ana uygulama"""
    
    # Başlık
    st.title("🚀 ARTPS - Otonom Bilimsel Keşif Sistemi")
    st.markdown("**Mars Rover'ları için Hibrit AI Sistemi (Derinlik + Dinamik Değer)**")
    
    # Sidebar
    st.sidebar.header("🎛️ Kontrol Paneli")
    
    # Modelleri yükleme
    with st.spinner("🤖 Hibrit modeller yükleniyor..."):
        models = load_models()
    
    if models is None:
        st.error("❌ Modeller yüklenemedi! Lütfen model dosyalarının varlığını kontrol edin.")
        return
    
    # Model durumu
    model_status = []
    if 'autoencoder' in models:
        model_status.append("✅ Autoencoder")
    if 'classifier' in models:
        model_status.append("✅ Hibrit Sınıflandırıcı")
    if 'depth_estimator' in models:
        depth_model_info = models['depth_model_info']
        model_status.append(f"✅ Derinlik Tahmini ({depth_model_info['model_type']}) - {'Yüksek Doğruluk' if depth_model_info['is_real_dpt'] else 'Basit Model'}")
    if 'padim' in models:
        model_status.append("✅ PaDiM (Anomali Füzyon)")
    if 'patchcore' in models:
        model_status.append("✅ PatchCore (Anomali Füzyon)")
    # Derinlik modeli durumunu detaylı göster
    if 'depth_model_info' in models:
        info = models['depth_model_info']
        st.sidebar.info(
            f"Aktif Derinlik Modeli: {info.get('model_type','?')} — Parametre: {info.get('param_count',0):,} — "
            + ("Yüksek Doğruluk" if info.get('is_real_dpt') else "Basit/Fallback")
        )
    
    st.sidebar.success(f"Modeller yüklendi:\n" + "\n".join(model_status))
    
    # Parametre ayarları
    st.sidebar.subheader("📊 Parametre Ayarları")
    
    alpha = st.sidebar.slider(
        "α (Alfa) - Bilinen Değer Ağırlığı", 0.0, 1.0, 0.4, 0.1,
        help="Curiosity skorunda sınıflandırıcının tahmin ettiği 'bilinen değer' katkısı. Yüksek olduğunda bilinen bilimsel açıdan değerli sınıflara benzer görüntüler daha çok öne çıkar."
    )
    beta = st.sidebar.slider(
        "β (Beta) - Anomali Ağırlığı", 0.0, 1.0, 0.6, 0.1,
        help="Curiosity skorunda AE tabanlı anomali MSE katkısı. Yüksek olduğunda beklenmedik/düzensiz yapılar daha çok öne çıkar."
    )
    w_combined = st.sidebar.slider(
        "w_combined (Birleşik Anomali)", 0.0, 1.0, 0.0, 0.05,
        help="Birleşik anomali haritasının ortalama yoğunluğunun curiosity skoruna katkısı. AE farkı, derinlik kenarı ve doku bileşenlerinden oluşur."
    )
    w_dvar = st.sidebar.slider(
        "w_depth_variance", 0.0, 1.0, 0.0, 0.05,
        help="Derinlik varyansının (3B yapı çeşitliliği) curiosity skoruna katkısı. Yüksek varyans, daha karmaşık jeomorfoloji anlamına gelebilir."
    )
    w_rough = st.sidebar.slider(
        "w_roughness", 0.0, 1.0, 0.0, 0.05,
        help="Pürüzlülük (gradyan ve laplace değişkenliği) katkısı. Küçük taş/kum çizgileri gibi ince detayları öne çıkarabilir."
    )
    
    anomaly_threshold = st.sidebar.slider(
        "Anomali Eşiği",
        min_value=0.0,
        max_value=0.01,
        value=0.003,
        step=0.0001,
        help="AE MSE için karar eşiği. Bu eşik üstü değerler tek başına 'anormal' kabul edilebilir."
    )
    ref_mse = st.sidebar.slider(
        "Curiosity Referans MSE",
        min_value=0.0005,
        max_value=0.02,
        value=0.003,
        step=0.0001,
        help="Curiosity normalizasyonu için AE MSE referansı. Yaklaşık olarak 2×ref MSE → 1.0 skora sıkıştırılır."
    )

    # Ağırlıkları global değişkenlere atayarak analiz fonksiyonuna geçiriyoruz
    globals()['alpha'] = alpha
    globals()['beta'] = beta
    globals()['w_combined'] = w_combined
    globals()['w_dvar'] = w_dvar
    globals()['w_rough'] = w_rough
    globals()['anomaly_threshold'] = anomaly_threshold
    globals()['ref_mse'] = ref_mse

    with st.sidebar.expander("🔧 Tespit Ayarları (Gelişmiş)", expanded=False):
        unified_threshold = st.slider("Birleşik Anomali Eşiği", 0.0, 1.0, 0.60, 0.01)
        col_adv1, col_adv2 = st.columns(2)
        with col_adv1:
            hyst_high = st.slider("Histerezis High (%)", 90, 99, 96, 1)
        with col_adv2:
            hyst_low = st.slider("Histerezis Low (%)", 85, 98, 90, 1)
        nms_iou = st.slider("NMS IoU", 0.10, 0.70, 0.25, 0.01)
        top_k = st.number_input("Top-K Kutu", min_value=5, max_value=100, value=25, step=1)
        min_area_pct = st.slider("Min Kutu Alanı (%)", 0.01, 2.00, 0.10, 0.01, help="Görüntü alanına göre")
        with st.expander("Ağırlıklar", expanded=False):
            w_recon = st.slider("w_recon (fark)", 0.0, 1.0, 0.50, 0.05)
            w_depth = st.slider("w_depthEdge (∇depth)", 0.0, 1.0, 0.30, 0.05)
            w_texture = st.slider("w_texture (gölge+kenar)", 0.0, 1.0, 0.20, 0.05)
            w_lap = st.slider("w_lap (Δ depth)", 0.0, 0.5, 0.08, 0.01)
            edge_reinf = st.slider("edge reinforce", 0.0, 1.0, 0.40, 0.05)
            w_detail = st.slider("w_detail (ince detay)", 0.0, 0.5, 0.12, 0.01, help="Küçük taş/kum çizgilerini vurgulayan çok ölçekli detay bileşeni")
            w_padim = st.slider("w_padim (PaDiM füzyon)", 0.0, 1.0, 0.30, 0.05, help="PaDiM anomali haritasının birleşik haritaya katkısı")
            w_patchcore = st.slider("w_patchcore (PatchCore füzyon)", 0.0, 1.0, 0.25, 0.05, help="PatchCore anomali haritasının birleşik haritaya katkısı")
        with st.expander("Kutu Birleştirme", expanded=False):
            merge_iou = st.slider("Birleştirme IoU", 0.0, 0.8, 0.15, 0.01)
            merge_tol = st.slider("Merkez Yakınlık (diagonal oranı)", 0.1, 1.5, 0.5, 0.05)
            st.caption("Yakın küçük kutuları birleşik hedefe toplar; uzak alandaki küçük detaylar için daha düşük IoU ile koruma sağlar.")
        with st.expander("Gölge Bastırma (Saha Ayarı)", expanded=False):
            alpha_shad = st.slider("Gölge Bastırma Gücü", 0.0, 1.0, 0.65, 0.05, help="Koyu + düşük kenarlı bölgeleri bastırma")
            beta_illum = st.slider("Aydınlatma-Kenar Azaltımı", 0.0, 1.0, 0.25, 0.05, help="Görüntü kenarı yüksek ama derinlik kenarı düşükse etkisini azaltır")
            shadow_cut = st.slider("Gölge Eleme Eşiği", 0.0, 1.0, 0.45, 0.05, help="Saf gölge bölgeleri eleme için alt sınır")
            img_edge_min = st.slider("Min Görüntü Kenarı", 0.0, 0.5, 0.10, 0.01)
            depth_edge_min = st.slider("Min Derinlik Kenarı", 0.0, 0.5, 0.08, 0.01)
            spec_gamma = st.slider("Speküler Bastırma Gücü", 0.0, 1.0, 0.35, 0.05, help="Yüksek parlaklık + düşük satürasyon bölgeleri bastırma")
            spec_cut = st.slider("Speküler Eleme Eşiği", 0.0, 1.0, 0.50, 0.05)
            spec_lowvar_gamma = st.slider("Düşük Varyans Azaltımı", 0.0, 1.0, 0.35, 0.05, help="Düşük doku (düşük varyans) speküler noktalara ek azaltım uygular")
            spec_var_thresh = st.slider("Düşük Varyans Eşiği", 0.0005, 0.02, 0.005, 0.0005)

        with st.expander("Odak Görselleri", expanded=False):
            focus_h = st.slider("Odak Karo Yüksekliği", 160, 480, 300, 10)
            focus_overlay = st.checkbox("Isı + Orijinal karışımını göster (overlay)", value=True)
            focus_sharpen = st.checkbox("Odak Keskinleştirme (unsharp)", value=True)
            focus_hide_empty_depth = st.checkbox("Derinlik karosu yoksa gizle", value=True)
            focus_interp = st.selectbox("Yeniden örnekleme", ["INTER_LANCZOS4", "INTER_CUBIC", "INTER_AREA"], index=0,
                help="Büyütmede LANCZOS4/CUBIC daha okunur sonuç verir; küçültmede AREA tercih edilir")
            st.caption("Hız için analizden hemen sonra odak karoları önceden üretilir.")

    # Curiosity ağırlıkları yönetimi (bozmadan opsiyonel)
    with st.sidebar.expander("🧭 Curiosity Ağırlıkları (Opsiyonel)", expanded=False):
        use_loaded = st.checkbox("Dosyadan yüklenen ağırlıkları kullan", value=False)
        weights_path = st.text_input("Ağırlık dosyası (JSON)", value="results/curiosity_weights.json")
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            if st.button("Yükle"):
                try:
                    with open(weights_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    models['curiosity_scorer'] = CuriosityScorer(CuriosityWeights(**data))
                    st.success("Ağırlıklar yüklendi")
                except Exception as e:
                    st.error(f"Yükleme hatası: {e}")
        with col_w2:
            if st.button("Varsayılanlara dön"):
                models['curiosity_scorer'] = CuriosityScorer(CuriosityWeights())
                st.info("Varsayılan ağırlıklar aktif")
        # Görüntüleme
        try:
            w = models['curiosity_scorer'].weights
            st.caption(f"Aktif: known={w.w_known:.3f}, anomaly={w.w_anomaly:.3f}, combined={w.w_combined:.3f}, dvar={w.w_depth_variance:.3f}, rough={w.w_roughness:.3f}")
        except Exception:
            pass
        globals()['use_loaded_weights'] = bool(use_loaded)
    
    # Ana içerik
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📸 Görüntü Analizi", "🔍 Derinlik Analizi", "📊 Sistem Durumu", "🎯 Demo Veriler", "ℹ️ Hakkında"])
    
    with tab1:
        st.header("📸 Mars Görüntüsü Hibrit Analizi")
        
        # Dosya yükleme
        uploaded_file = st.file_uploader(
            "Mars görüntüsü yükleyin (JPG, PNG)",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file is not None:
            # Görüntüyü yükle
            image = Image.open(uploaded_file).convert('RGB')

            # Otomatik görüntü iyileştirme seçenekleri
            st.subheader("🧹 Otomatik Görüntü İyileştirme")
            enh_cols = st.columns(5)
            with enh_cols[0]:
                opt_upscale = st.checkbox("Upscale", value=True, help="Düşük çözünürlüklü görselleri akıllı büyütme")
            with enh_cols[1]:
                opt_denoise = st.checkbox("Denoise", value=True, help="Yüksek gürültülü görüntülerde renkli gürültü giderme")
            with enh_cols[2]:
                opt_clahe = st.checkbox("Kontrast (CLAHE)", value=True)
            with enh_cols[3]:
                opt_gamma = st.checkbox("Pozlama (Gamma)", value=True)
            with enh_cols[4]:
                opt_sharp = st.checkbox("Keskinleştirme", value=True)

            # İyileştirme uygula butonu
            if st.button("✨ Görüntüyü Otomatik İyileştir"):
                cfg = dict(
                    enable_upscale=opt_upscale,
                    enable_denoise=opt_denoise,
                    enable_clahe=opt_clahe,
                    enable_gamma=opt_gamma,
                    enable_sharpen=opt_sharp,
                    target_long_side=1024,
                    denoise_h=5,
                    clahe_clip=2.0,
                    target_mean=128.0,
                    sharpen_strength=0.6,
                    sharpen_radius=2,
                )
                enhanced, before_m, after_m, steps = enhance_image_auto(image, cfg)
                st.success(f"Uygulanan adımlar: {', '.join(steps)}")
                c1, c2 = st.columns(2)
                with c1:
                    st.image(image, caption="Önce", use_container_width=True)
                    st.json({"Önce": before_m})
                with c2:
                    st.image(enhanced, caption="Sonra", use_container_width=True)
                    st.json({"Sonra": after_m})
                # Analizde iyileştirilmiş görüntüyü kullan
                image = enhanced
                st.session_state["enhanced_image_for_analysis"] = image
            
            # İki sütunlu layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Orijinal Görüntü")
                st.image(image, caption="Yüklenen Mars görüntüsü", use_container_width=True)
            
            # Analiz butonu
            clicked = st.button("🔍 Hibrit Analiz Et", type="primary")
            if clicked:
                with st.spinner("Hibrit analiz yapılıyor (Anomali + Derinlik + Dinamik Değer)..."):
                    # Kapsamlı analiz
                    # Varsa iyileştirilmiş görüntüyü kullan
                    image_to_use = st.session_state.get("enhanced_image_for_analysis", image)
                    # Gelişmiş tespit ayarlarını global değişken olarak geçir
                    globals().update({
                        'unified_threshold': unified_threshold,
                        'hyst_high': hyst_high,
                        'hyst_low': hyst_low,
                        'nms_iou': nms_iou,
                        'top_k': top_k,
                        'w_recon': w_recon,
                        'w_depth': w_depth,
                        'w_texture': w_texture,
                        'edge_reinf': edge_reinf,
                        'alpha_shad': alpha_shad,
                        'beta_illum': beta_illum,
                        'shadow_cut': shadow_cut,
                        'img_edge_min': img_edge_min,
                        'depth_edge_min': depth_edge_min,
                        'spec_gamma': spec_gamma,
                        'spec_cut': spec_cut,
                        'spec_lowvar_gamma': spec_lowvar_gamma,
                        'spec_var_thresh': spec_var_thresh,
                        'w_padim': w_padim,
                        'w_patchcore': w_patchcore,
                         'focus_h': focus_h,
                         'focus_overlay': focus_overlay,
                         'focus_sharpen': focus_sharpen,
                         'focus_hide_empty_depth': focus_hide_empty_depth,
                         'focus_interp': focus_interp,
                    })
                    results = analyze_mars_image(models, image_to_use)
                    # Sonuçları yeniden çalıştırmalarda koru
                    st.session_state["results"] = results
                    
                    if results['anomaly_score'] is not None:
                        # Sonuçları göster
                        with col2:
                            st.subheader("🔄 Yeniden Oluşturulan Görüntü")
                            st.image(
                                results['reconstructed'],
                                caption=f"Anomali Skoru: {results['anomaly_score']:.6f}",
                                use_container_width=True,
                            )
                        
                        # Sonuç analizi
                        st.subheader("📊 Hibrit Analiz Sonuçları")
                        
                        # Metrikler
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            st.metric("Anomali Skoru (MSE)", f"{results['anomaly_score']:.6f}")
                        
                        with col2:
                            # Birleşik anomali skoru (derinlik + rekonstrüksiyon)
                            if results.get('combined_anomaly_score') is not None:
                                mse_norm = float(np.clip(results['anomaly_score'] / max(anomaly_threshold, 1e-6), 0.0, 1.0))
                                comb = float(results['combined_anomaly_score'])
                                unified_anomaly = 0.5 * mse_norm + 0.5 * comb
                                st.metric("Birleşik Anomali", f"{unified_anomaly:.3f}")
                                is_anomaly = unified_anomaly > unified_threshold
                            else:
                                is_anomaly = results['anomaly_score'] > anomaly_threshold
                                st.metric("Birleşik Anomali", "N/A")
                            st.metric("Anomali Durumu", "🚨 Anormal" if is_anomaly else "✅ Normal")
                        
                        with col3:
                            st.metric("Bilinen Değer", f"{results['known_value_score']:.3f}")
                        
                        with col4:
                            # İlginçlik puanı (modüler skorlayıcıdan)
                            curiosity_score = results.get('curiosity_score')
                            if curiosity_score is None:
                                curiosity_score = alpha * results['known_value_score'] + beta * results['anomaly_score']
                            st.metric("İlginçlik Puanı", f"{curiosity_score:.6f}")
                        
                        with col5:
                            if 'predicted_class' in results:
                                class_names = {0: "Değersiz", 1: "Düşük", 2: "Orta", 3: "Orta-Yüksek", 4: "Yüksek"}
                                predicted_name = class_names.get(results['predicted_class'], "Bilinmiyor")
                                st.metric("Tahmin Edilen Sınıf", predicted_name)
                        
                        # Fark görüntüsü + birleşik anomali haritası
                        st.subheader("🔍 Fark ve Birleşik Anomali Haritası")
                        diff = np.abs(results['original'] - results['reconstructed'])

                        if results.get('combined_anomaly_map') is not None:
                            comb_map = results['combined_anomaly_map']
                            # Orijinale ısı haritası bindirme (boyutları eşitle)
                            H, W = comb_map.shape[:2]
                            base = (results['original'] * 255).astype(np.uint8)
                            if base.shape[:2] != (H, W):
                                base = cv2.resize(base, (W, H), interpolation=cv2.INTER_LINEAR)
                            if base.ndim == 2:
                                base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
                            heat = (plt.cm.inferno(comb_map)[..., :3] * 255).astype(np.uint8)
                            overlay = cv2.addWeighted(base, 0.6, heat, 0.4, 0)

                            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                            _safe_imshow(axes[0], results['original'])
                            axes[0].set_title("Original")
                            axes[0].axis('off')
                            _safe_imshow(axes[1], results['reconstructed'])
                            axes[1].set_title("Reconstructed")
                            axes[1].axis('off')
                            _safe_imshow(axes[2], diff, cmap='hot')
                            axes[2].set_title("Difference")
                            axes[2].axis('off')
                            _safe_imshow(axes[3], overlay)
                            axes[3].set_title("Combined Anomaly (overlay)")
                            axes[3].axis('off')
                            st.pyplot(fig)

                            # Tespit kutularını göster (tespit olmasa da overlay göster)
                            # Combined anomaly overlay'i hafifçe büyüt, sonra etiketleri çiz
                            detections = results.get('detections') or []
                            # Sağ panelde seçim durum anahtarını hazırla
                            select_key = "diag_selected_idx"
                            if select_key not in st.session_state:
                                st.session_state[select_key] = 0
                            col_vis, col_diag = st.columns([3, 2], gap="large")
                            with col_diag:
                                st.subheader("🔎 Tespit Tanılama Paneli")
                                with st.expander("❓ Metrik Açıklamaları", expanded=False):
                                    st.markdown(
                                        "- **sc**: Birleşik anomali skoru\n"
                                        "- **e**: Kenar yoğunluğu göstergesi\n"
                                        "- **s**: Gölge/karanlık etkisi (azaltım)\n"
                                        "- **sp**: Parlama (speküler) etkisi (azaltım)\n"
                                        "- **lv**: Düşük doku/varians etkisi (azaltım)"
                                    )
                                # Hızlı seçim widget'ını ÖNCE oluştur ki bu turda seçimi kullanabilelim
                                try:
                                    table_rows = []
                                    for i, det in enumerate(detections, start=1):
                                        table_rows.append({
                                            "#": i,
                                            "sc": round(float(det.get('score', 0.0)), 3),
                                            "e": round(float(det.get('edge_mean', 0.0)), 3),
                                            "s": round(float(det.get('shadow_pen', 0.0)), 3),
                                            "sp": round(float(det.get('spec_pen', 0.0)), 3),
                                            "lv": round(float(det.get('lowvar_pen', 0.0)), 3),
                                        })
                                    if len(table_rows) > 0:
                                        st.table(table_rows)
                                        _ = st.radio(
                                            "Hızlı Seçim",
                                            options=[0] + [r["#"] for r in table_rows],
                                            index=([0] + [r["#"] for r in table_rows]).index(st.session_state.get(select_key, 0) if st.session_state.get(select_key, 0) in ([0] + [r["#"] for r in table_rows]) else 0),
                                            format_func=lambda i: ("Tümü" if i == 0 else f"#{i}"),
                                            horizontal=True,
                                            key=select_key,
                                        )
                                except Exception:
                                    pass
                            try:
                                selected_idx = int(st.session_state.get(select_key, 0))
                            except Exception:
                                selected_idx = 0
                            oh0, ow0 = overlay.shape[0], overlay.shape[1]
                            scale_up = 2.5  # istenen büyütme (1.6x)
                            disp = cv2.resize(
                                overlay,
                                (int(round(ow0 * scale_up)), int(round(oh0 * scale_up))),
                                interpolation=cv2.INTER_CUBIC,
                            )
                            disp_base = disp.copy()
                            # Odak modu: seçili anomali varsa arka planı yumuşak maske ile karart ve seçilen bölgeyi ön plana çıkar
                            if selected_idx > 0 and selected_idx <= len(detections):
                                sel = detections[selected_idx - 1]
                                sx, sy, sw, sh = sel['x'], sel['y'], sel['w'], sel['h']
                                sxs, sys = int(round(sx * scale_up)), int(round(sy * scale_up))
                                sws, shs = int(round(sw * scale_up)), int(round(sh * scale_up))
                                dimmed = (disp * 0.20).astype(np.uint8)
                                mask = np.zeros((disp.shape[0], disp.shape[1]), dtype=np.float32)
                                y1 = max(0, sys)
                                y2 = min(disp.shape[0], sys + shs)
                                x1 = max(0, sxs)
                                x2 = min(disp.shape[1], sxs + sws)
                                if y2 > y1 and x2 > x1:
                                    mask[y1:y2, x1:x2] = 1.0
                                    mask = cv2.GaussianBlur(mask, (61, 61), 0)
                                    mask = np.clip(mask, 0.0, 1.0)[..., None]
                                    disp = (disp_base.astype(np.float32) * mask + dimmed.astype(np.float32) * (1.0 - mask)).astype(np.uint8)

                            diag_lines = []
                            for i, det in enumerate(detections):
                                x, y, w, h = det['x'], det['y'], det['w'], det['h']
                                xs, ys = int(round(x * scale_up)), int(round(y * scale_up))
                                ws, hs = int(round(w * scale_up)), int(round(h * scale_up))
                                idx_num = i + 1
                                is_selected = (selected_idx == idx_num)
                                box_color = (255, 0, 0) if is_selected else (0, 255, 0)
                                box_thickness = 2 if is_selected else 2
                                if det.get('poly'):
                                    pts = np.array(det['poly'], dtype=np.float32).reshape((-1, 2))
                                    pts = (pts * scale_up).astype(np.int32).reshape((-1, 1, 2))
                                    cv2.polylines(disp, [pts], isClosed=True, color=box_color, thickness=box_thickness)
                                else:
                                    cv2.rectangle(disp, (xs, ys), (xs + ws, ys + hs), box_color, box_thickness)
                                # Okunur etiket (opak zemin): sadece numara
                                label = f"#{idx_num}"
                                # Etiketleri biraz daha küçült
                                font_scale = 0.26 * scale_up
                                text_thickness = 1
                                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
                                bx1, by1 = xs, max(0, ys - th - 6)
                                bx2, by2 = xs + tw + 6, by1 + th + 4
                                cv2.rectangle(disp, (bx1, by1), (bx2, by2), box_color, -1)
                                cv2.putText(disp, label, (xs + 3, by2 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness, cv2.LINE_AA)
                                # Mini-diagnostic metni bir listede topla (sağ panelde gösterilecek)
                                diag = f"#{idx_num} sc:{det.get('score',0):.2f} e:{det.get('edge_mean',0):.2f} s:{det.get('shadow_pen',0):.2f} sp:{det.get('spec_pen',0):.2f} lv:{det.get('lowvar_pen',0):.2f}"
                                diag_lines.append(diag)

                            # Not: diag_lines ayrı panelde gösterilecektir (görsele eklenmez)
                            # Odaklı görünüm: seçili anomali varsa ana görseli de merkezleyip yakınlaştır
                            disp_to_show = disp
                            if selected_idx > 0 and selected_idx <= len(detections):
                                cx = sxs + sws // 2
                                cy = sys + shs // 2
                                # Hedef kırpma boyutu: seçili kutudan daha geniş bir pencere
                                crop_w = int(min(disp.shape[1], max(int(sws * 2.5), 520)))
                                crop_h = int(min(disp.shape[0], max(int(shs * 2.5), 520)))
                                x1 = max(0, min(disp.shape[1] - crop_w, cx - crop_w // 2))
                                y1 = max(0, min(disp.shape[0] - crop_h, cy - crop_h // 2))
                                x2 = x1 + crop_w
                                y2 = y1 + crop_h
                                if (y2 - y1) > 10 and (x2 - x1) > 10:
                                    disp_to_show = disp[y1:y2, x1:x2]
                            # Gösterimi sabit hedef genişliğe göre yeniden boyutlandır (biraz daha büyük hedef)
                            oh, ow = disp_to_show.shape[0], disp_to_show.shape[1]
                            pref_w = 860
                            scale = min(0.95, max(0.60, float(pref_w) / max(1.0, float(ow))))
                            target_w = max(1, int(round(ow * scale)))
                            target_h = max(1, int(round(oh * scale)))
                            disp_small = cv2.resize(disp_to_show, (target_w, target_h), interpolation=cv2.INTER_AREA)
                            caption = "Birleşik Anomali Tespitleri" + (" — tespit bulunamadı" if len(detections) == 0 else "")
                            # Görsel ve paneli yukarıda oluşturduğumuz kolonlarda göster
                            with col_vis:
                                st.markdown('<div id="anomaly_anchor"></div>', unsafe_allow_html=True)
                                st.image(
                                    disp_small,
                                    caption=f"{caption} (küçük cisimler dahil edilir)",
                                    use_container_width=False,
                                )
                            with col_diag:
                                if diag_lines:
                                    st.code("\n".join(diag_lines), language="text")
                                else:
                                    st.info("Tespit bulunamadı veya tanılama verisi yok.")
                                # Seçili anomali için yakınlaştırılmış odak görüntüsü
                                try:
                                    selected_idx_view = int(st.session_state.get(select_key, 0))
                                except Exception:
                                    selected_idx_view = 0
                                if selected_idx_view > 0 and selected_idx_view <= len(detections):
                                    tiles = results.get('focus_tiles') or []
                                    tile = tiles[selected_idx_view - 1] if (selected_idx_view - 1) < len(tiles) else None
                                    if tile is not None:
                                        st.image(tile, caption=f"Odak: #{selected_idx_view}")
                                # Seçim değiştiğinde otomatik kaydırma
                                _prev = st.session_state.get("_prev_selected_idx", -1)
                                if _prev != selected_idx:
                                    st.session_state["_prev_selected_idx"] = selected_idx
                                    st.markdown("""
                                        <script>
                                        const el = document.getElementById('anomaly_anchor');
                                        if (el) { el.scrollIntoView({behavior: 'smooth', block: 'center'}); }
                                        </script>
                                    """, unsafe_allow_html=True)
                        else:
                            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                            _safe_imshow(axes[0], results['original'])
                            axes[0].set_title("Original")
                            axes[0].axis('off')
                            _safe_imshow(axes[1], results['reconstructed'])
                            axes[1].set_title("Reconstructed")
                            axes[1].axis('off')
                            _safe_imshow(axes[2], diff, cmap='hot')
                            axes[2].set_title("Difference (Anomaly)")
                            axes[2].axis('off')
                            st.pyplot(fig)

            # Eğer sonuç daha önce üretildiyse (ör. seçim değişince rerun), tekrar göster
            persisted = st.session_state.get("results")
            if persisted and not clicked:
                results = persisted
                diff = np.abs(results['original'] - results['reconstructed'])
                if results.get('combined_anomaly_map') is not None:
                    comb_map = results['combined_anomaly_map']
                    H, W = comb_map.shape[:2]
                    base = (results['original'] * 255).astype(np.uint8)
                    if base.shape[:2] != (H, W):
                        base = cv2.resize(base, (W, H), interpolation=cv2.INTER_LINEAR)
                    if base.ndim == 2:
                        base = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
                    heat = (plt.cm.inferno(comb_map)[..., :3] * 255).astype(np.uint8)
                    overlay = cv2.addWeighted(base, 0.6, heat, 0.4, 0)

                    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                    _safe_imshow(axes[0], results['original'])
                    axes[0].set_title("Original")
                    axes[0].axis('off')
                    _safe_imshow(axes[1], results['reconstructed'])
                    axes[1].set_title("Reconstructed")
                    axes[1].axis('off')
                    _safe_imshow(axes[2], diff, cmap='hot')
                    axes[2].set_title("Difference")
                    axes[2].axis('off')
                    _safe_imshow(axes[3], overlay)
                    axes[3].set_title("Combined Anomaly (overlay)")
                    axes[3].axis('off')
                    st.pyplot(fig)

                    detections = results.get('detections') or []
                    select_key = "diag_selected_idx"
                    if select_key not in st.session_state:
                        st.session_state[select_key] = 0
                    col_vis, col_diag = st.columns([3, 2], gap="large")
                    with col_diag:
                        st.subheader("🔎 Tespit Tanılama Paneli")
                        with st.expander("❓ Metrik Açıklamaları", expanded=False):
                            st.markdown(
                                "- **sc**: Birleşik anomali skoru\n"
                                "- **e**: Kenar yoğunluğu göstergesi\n"
                                "- **s**: Gölge/karanlık etkisi (azaltım)\n"
                                "- **sp**: Parlama (speküler) etkisi (azaltım)\n"
                                "- **lv**: Düşük doku/varians etkisi (azaltım)"
                            )
                        # Hızlı seçim bileşeni: aynı turda seçimi yakalamak için ÖNCE oluştur
                        if len(detections) > 0:
                            try:
                                table_rows = []
                                for i, det in enumerate(detections, start=1):
                                    table_rows.append({
                                        "#": i,
                                        "sc": round(float(det.get('score', 0.0)), 3),
                                        "e": round(float(det.get('edge_mean', 0.0)), 3),
                                        "s": round(float(det.get('shadow_pen', 0.0)), 3),
                                        "sp": round(float(det.get('spec_pen', 0.0)), 3),
                                        "lv": round(float(det.get('lowvar_pen', 0.0)), 3),
                                    })
                                st.table(table_rows)
                                _ = st.radio(
                                    "Hızlı Seçim",
                                    options=[0] + [r["#"] for r in table_rows],
                                    index=([0] + [r["#"] for r in table_rows]).index(st.session_state.get(select_key, 0) if st.session_state.get(select_key, 0) in ([0] + [r["#"] for r in table_rows]) else 0),
                                    format_func=lambda i: ("Tümü" if i == 0 else f"#{i}"),
                                    horizontal=True,
                                    key=select_key,
                                )
                            except Exception:
                                pass

                    oh0, ow0 = overlay.shape[0], overlay.shape[1]
                    scale_up = 1.60
                    disp = cv2.resize(overlay, (int(round(ow0 * scale_up)), int(round(oh0 * scale_up))), interpolation=cv2.INTER_CUBIC)
                    disp_base = disp.copy()
                    try:
                        selected_idx = int(st.session_state.get(select_key, 0))
                    except Exception:
                        selected_idx = 0
                    if selected_idx > 0 and selected_idx <= len(detections):
                        sel = detections[selected_idx - 1]
                        sx, sy, sw, sh = sel['x'], sel['y'], sel['w'], sel['h']
                        sxs, sys = int(round(sx * scale_up)), int(round(sy * scale_up))
                        sws, shs = int(round(sw * scale_up)), int(round(sh * scale_up))
                        dimmed = (disp * 0.25).astype(np.uint8)
                        disp = dimmed
                        y1 = max(0, sys); y2 = min(disp.shape[0], sys + shs)
                        x1 = max(0, sxs); x2 = min(disp.shape[1], sxs + sws)
                        if y2 > y1 and x2 > x1:
                            disp[y1:y2, x1:x2] = disp_base[y1:y2, x1:x2]

                    diag_lines = []
                    for i, det in enumerate(detections):
                        x, y, w, h = det['x'], det['y'], det['w'], det['h']
                        xs, ys = int(round(x * scale_up)), int(round(y * scale_up))
                        ws, hs = int(round(w * scale_up)), int(round(h * scale_up))
                        idx_num = i + 1
                        is_selected = (selected_idx == idx_num)
                        box_color = (255, 0, 0) if is_selected else (0, 255, 0)
                        box_thickness = 2 if is_selected else 2
                        if det.get('poly'):
                            pts = np.array(det['poly'], dtype=np.float32).reshape((-1, 2))
                            pts = (pts * scale_up).astype(np.int32).reshape((-1, 1, 2))
                            cv2.polylines(disp, [pts], isClosed=True, color=box_color, thickness=box_thickness)
                        else:
                            cv2.rectangle(disp, (xs, ys), (xs + ws, ys + hs), box_color, box_thickness)
                        label = f"#{idx_num}"
                        font_scale = 0.26 * scale_up
                        text_thickness = 1
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
                        bx1, by1 = xs, max(0, ys - th - 6)
                        bx2, by2 = xs + tw + 6, by1 + th + 4
                        cv2.rectangle(disp, (bx1, by1), (bx2, by2), box_color, -1)
                        cv2.putText(disp, label, (xs + 3, by2 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness, cv2.LINE_AA)
                        diag = f"#{idx_num} sc:{det.get('score',0):.2f} e:{det.get('edge_mean',0):.2f} s:{det.get('shadow_pen',0):.2f} sp:{det.get('spec_pen',0):.2f} lv:{det.get('lowvar_pen',0):.2f}"
                        diag_lines.append(diag)

                    disp_to_show = disp
                    if selected_idx > 0 and selected_idx <= len(detections):
                        cx = sxs + sws // 2; cy = sys + shs // 2
                        crop_w = int(min(disp.shape[1], max(int(sws * 2.5), 520)))
                        crop_h = int(min(disp.shape[0], max(int(shs * 2.5), 520)))
                        x1 = max(0, min(disp.shape[1] - crop_w, cx - crop_w // 2))
                        y1 = max(0, min(disp.shape[0] - crop_h, cy - crop_h // 2))
                        x2 = x1 + crop_w; y2 = y1 + crop_h
                        if (y2 - y1) > 10 and (x2 - x1) > 10:
                            disp_to_show = disp[y1:y2, x1:x2]

                    oh, ow = disp_to_show.shape[0], disp_to_show.shape[1]
                    pref_w = 860
                    scale = min(0.95, max(0.60, float(pref_w) / max(1.0, float(ow))))
                    target_w = max(1, int(round(ow * scale)))
                    target_h = max(1, int(round(oh * scale)))
                    disp_small = cv2.resize(disp_to_show, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    caption = "Birleşik Anomali Tespitleri" + (" — tespit bulunamadı" if len(detections) == 0 else "")
                    with col_vis:
                        st.image(disp_small, caption=f"{caption} (küçük cisimler dahil edilir)", use_container_width=False)
                    with col_diag:
                        if diag_lines:
                            st.code("\n".join(diag_lines), language="text")
                        else:
                            st.info("Tespit bulunamadı veya tanılama verisi yok.")
                        try:
                            selected_idx_view = int(st.session_state.get(select_key, 0))
                        except Exception:
                            selected_idx_view = 0
                        if selected_idx_view > 0 and selected_idx_view <= len(detections):
                            tiles = results.get('focus_tiles') or []
                            tile = tiles[selected_idx_view - 1] if (selected_idx_view - 1) < len(tiles) else None
                            if tile is not None:
                                st.image(tile, caption=f"Odak: #{selected_idx_view}")
                        
                        # is_anomaly'yi (persisted sonuçlar için) yeniden hesapla
                        try:
                            unified_anomaly = results.get('combined_anomaly_score')
                        except Exception:
                            unified_anomaly = None
                        if unified_anomaly is None and results.get('combined_anomaly_map') is not None:
                            try:
                                unified_anomaly = float(np.mean(results['combined_anomaly_map']))
                            except Exception:
                                unified_anomaly = None
                        if unified_anomaly is not None:
                            is_anomaly = bool(unified_anomaly > unified_threshold)
                        else:
                            try:
                                is_anomaly = bool(results.get('anomaly_score', 0.0) > anomaly_threshold)
                            except Exception:
                                is_anomaly = False

                        # Öneriler
                        st.subheader("💡 Hibrit Öneriler")
                        
                        if is_anomaly and results['known_value_score'] > 0.6:
                            st.success("🎯 **YÜKSEK ÖNCELİK**: Bu hedef hem anormal hem de yüksek bilimsel değere sahip!")
                        elif is_anomaly:
                            st.warning("🔍 **ORTA ÖNCELİK**: Bu hedef anormal ama bilimsel değeri orta seviyede.")
                        elif results['known_value_score'] > 0.7:
                            st.info("📋 **DÜŞÜK ÖNCELİK**: Bu hedef normal ama bilinen değerli hedeflere benziyor.")
                        else:
                            st.info("📋 **DÜŞÜK ÖNCELİK**: Bu hedef normal Mars yüzeyi görünüyor.")
    
    with tab2:
        st.header("🔍 Derinlik Analizi")
        
        if uploaded_file is not None and 'depth_estimator' in models:
            depth_model_info = models['depth_model_info']
            st.subheader(f"🌊 Derinlik Haritası ({depth_model_info['model_type']}) - {'Yüksek Doğruluk' if depth_model_info['is_real_dpt'] else 'Basit Model'}")
            
            # Kullanıcı seçenekleri: çözünürlük ve iyileştirme
            col_opts1, col_opts2, col_opts3 = st.columns(3)
            with col_opts1:
                target_resolution = st.selectbox(
                    "Çözünürlük",
                    options=[512, 768, 1024],
                    index=2,
                    help="Giriş görüntüsünün analizde kullanılacak çözünürlüğü"
                )
            with col_opts2:
                apply_enhancement = st.checkbox(
                    "Geliştirme Uygula (kontrast + keskinleştirme)",
                    value=True
                )
            with col_opts3:
                show_raw_compare = st.checkbox(
                    "Ham çıktıyla karşılaştır",
                    value=False,
                    help="Geliştirme kapalı (ham) ve açık çıktıları yan yana göster"
                )

            # Derinlik analizi (yüksek çözünürlük)
            image = Image.open(uploaded_file).convert('RGB')
            # Varsa görüntü iyileştirme sonrası sürümü kullan
            image = st.session_state.get("enhanced_image_for_analysis", image)
            # Seçilen çözünürlükte işle
            image_array = np.array(image.resize((target_resolution, target_resolution), Image.LANCZOS), dtype=np.float32) / 255.0
            
            try:
                # İyileştirme açık/kapalı seçenekleri
                t0 = time.perf_counter()
                depth_map, metadata = models['depth_estimator'].estimate_depth(
                    image_array,
                    apply_enhancement=apply_enhancement,
                    guide_image=np.array(image),
                    high_detail=True,
                    tta_flips=True,
                    use_fgs=True,
                    use_wmf=True,
                )
                t1 = time.perf_counter()
                infer_ms = (t1 - t0) * 1000.0
                
                # Derinlik görselleştirmesi
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption="Orijinal Görüntü", use_container_width=True)
                
                with col2:
                    # Geliştirilmiş derinlik görselleştirmesi
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Daha iyi colormap ve kontrast (turbo daha kontrastlı)
                    im = ax.imshow(depth_map, cmap='turbo', interpolation='bilinear')
                    ax.set_title("Gelistirilmis Derinlik Haritasi", fontsize=14, fontweight='bold')
                    ax.axis('off')
                    
                    # Geliştirilmiş colorbar
                    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
                    cbar.set_label('Derinlik (0=Yakın, 1=Uzak)', fontsize=12)
                    cbar.ax.tick_params(labelsize=10)
                    
                    # Grid ekle
                    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                    
                    plt.tight_layout()
                    st.pyplot(fig)

                # İsteğe bağlı: ham çıktı ile karşılaştırma
                if show_raw_compare:
                    depth_raw, _ = models['depth_estimator'].estimate_depth(
                        image_array, apply_enhancement=False, guide_image=np.array(image), high_detail=True
                    )
                    fig_cmp, (axc1, axc2) = plt.subplots(1, 2, figsize=(14, 6))
                    axc1.imshow(depth_raw, cmap='turbo', interpolation='bilinear')
                    axc1.set_title('Ham DPT Çıkışı')
                    axc1.axis('off')
                    axc2.imshow(depth_map, cmap='turbo', interpolation='bilinear')
                    axc2.set_title('Geliştirme Uygulandı' if apply_enhancement else 'Geliştirme Kapalı')
                    axc2.axis('off')
                    plt.tight_layout()
                    st.pyplot(fig_cmp)
                    
                # Derinlik analizi bilgileri ve süre
                st.info(f"📊 **Derinlik Analizi ({depth_model_info['model_type']})**: {depth_map.shape[1]}x{depth_map.shape[0]} çözünürlük, "
                       f"Kontrast: {depth_map.std():.3f}, Ortalama Derinlik: {depth_map.mean():.3f}, Süre: {infer_ms:.1f} ms")
                
                # İnce ayar paneli
                with st.expander("🔧 Derinlik İnce Ayar (Gelişmiş)", expanded=False):
                    colp1, colp2, colp3 = st.columns(3)
                    with colp1:
                        gf_radius = st.slider("GuidedFilter radius", 2, 32, 8, 1,
                                              help="Guided Filter yarıçapı. Büyük değer: daha geniş, pürüzsüz ancak kenar yumuşaması artabilir.")
                        gf_eps = st.number_input("GuidedFilter eps", min_value=1e-6, max_value=1e-1, value=1e-2, step=1e-3, format="%f",
                                                 help="Guided Filter epsilon. Düşük eps: daha keskin; yüksek eps: daha yumuşak.")
                        jbf_d = st.slider("JointBF d", 1, 21, 9, 1,
                                          help="Joint Bilateral filtre çekirdek çapı. Kenar korumalı yumuşatma için pencere boyutu.")
                    with colp2:
                        jbf_sc = st.slider("JointBF sigmaColor", 1, 100, 25, 1,
                                           help="Renk/yoğunluk duyarlılığı. Yüksek değer: daha fazla yumuşatma, kenar kaçakları artabilir.")
                        jbf_ss = st.slider("JointBF sigmaSpace", 1, 100, 25, 1,
                                           help="Uzamsal duyarlılık. Yüksek değer: daha geniş etkili alan, daha pürüzsüz sonuç.")
                        fgs_lambda = st.slider("FGS lambda", 1.0, 2000.0, 500.0, 1.0,
                                               help="Fast Global Smoother düzgünleştirme gücü. Büyük değer: daha düz, küçük detaylar azalabilir.")
                    with colp3:
                        fgs_sigma = st.slider("FGS sigma_color", 0.1, 5.0, 1.5, 0.1,
                                              help="FGS için renk alanı ölçeği. Kenar hassasiyetini etkiler.")
                        wmf_radius = st.slider("WMF radius", 1, 31, 7, 1,
                                               help="Weighted Median Filter yarıçapı. Gürültüye karşı sağlam, kenarları iyi korur.")
                        wmf_sigma = st.slider("WMF sigma", 1.0, 80.0, 25.5, 0.5,
                                              help="WMF ağırlıklandırma gücü. Büyük değer: daha fazla düzeltme/yumuşatma.")
                    if st.button("Uygula (Derinlik İyileştirmeyi Güncelle)"):
                        models['depth_estimator'].set_refine_params(
                            gf_radius=gf_radius, gf_eps=float(gf_eps), jbf_d=jbf_d,
                            jbf_sigma_color=jbf_sc, jbf_sigma_space=jbf_ss,
                            fgs_lambda=float(fgs_lambda), fgs_sigma_color=float(fgs_sigma),
                            wmf_radius=wmf_radius, wmf_sigma=float(wmf_sigma),
                        )
                        st.success("İnce ayar parametreleri güncellendi. 'Derinlik Analizi' bölümünü tekrar çalıştırın.")

                # Colormap seçenekleri
                st.subheader("🎨 Derinlik Görselleştirme Seçenekleri")
                colormap_option = st.selectbox(
                    "Colormap Seçin:",
                    ["turbo", "plasma", "inferno", "magma", "viridis", "cividis"],
                    index=0,
                    help="Farklı colormap'ler derinlik detaylarını farklı şekilde vurgular"
                )
                
                # Seçilen colormap ile yeniden çiz
                fig2, ax2 = plt.subplots(figsize=(10, 8))
                im2 = ax2.imshow(depth_map, cmap=colormap_option, interpolation='bilinear')
                ax2.set_title(f"Derinlik Haritasi ({colormap_option})", fontsize=14, fontweight='bold')
                ax2.axis('off')
                
                cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, aspect=20)
                cbar2.set_label('Derinlik (0=Yakın, 1=Uzak)', fontsize=12)
                cbar2.ax.tick_params(labelsize=10)
                
                ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                plt.tight_layout()
                st.pyplot(fig2)
                
                # Derinlik özelliklerini çıkar
                depth_features = models['depth_estimator'].extract_depth_features(depth_map)
                
                # Derinlik özellikleri
                st.subheader("📊 Geliştirilmiş Derinlik Özellikleri")
                
                # Özellikleri göster (daha detaylı)
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🌊 Ortalama Derinlik", f"{depth_features.get('depth_mean', 0):.3f}")
                    st.metric("📏 Derinlik Std", f"{depth_features.get('depth_std', 0):.3f}")
                    st.metric("📊 Derinlik Varyansı", f"{depth_features.get('depth_variance', 0):.3f}")
                
                with col2:
                    st.metric("⬇️ Min Derinlik", f"{depth_features.get('depth_min', 0):.3f}")
                    st.metric("⬆️ Max Derinlik", f"{depth_features.get('depth_max', 0):.3f}")
                    st.metric("📈 Derinlik Medyan", f"{depth_features.get('depth_median', 0):.3f}")
                
                with col3:
                    st.metric("🏔️ Yüzey Karmaşıklığı", f"{depth_features.get('surface_complexity', 0):.3f}")
                    st.metric("🌊 Gradient Ortalama", f"{depth_features.get('depth_gradient_mean', 0):.3f}")
                    st.metric("📐 Gradient Std", f"{depth_features.get('depth_gradient_std', 0):.3f}")
                
                with col4:
                    st.metric("📊 Skewness", f"{depth_features.get('depth_skewness', 0):.3f}")
                    st.metric("📈 Kurtosis", f"{depth_features.get('depth_kurtosis', 0):.3f}")
                    st.metric("🎯 P75-P25", f"{depth_features.get('depth_percentile_75', 0) - depth_features.get('depth_percentile_25', 0):.3f}")
                
                # Derinlik metadata ve ek analizler
                st.subheader("📋 Derinlik Metadata")
                st.json(metadata)
                
                # Derinlik histogramı
                st.subheader("📊 Derinlik Dağılımı")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                
                # Histogram
                ax1.hist(depth_map.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                ax1.set_title("Depth Histogram")
                ax1.set_xlabel("Depth Value")
                ax1.set_ylabel("Frequency")
                ax1.grid(True, alpha=0.3)
                
                # 3D yüzey plot (küçük örnek)
                try:
                    sample_size = min(50, depth_map.shape[0], depth_map.shape[1])
                    sample_depth = depth_map[::depth_map.shape[0]//sample_size, ::depth_map.shape[1]//sample_size]
                    y, x = np.mgrid[0:sample_depth.shape[0], 0:sample_depth.shape[1]]
                    
                    surf = ax2.plot_surface(x, y, sample_depth, cmap='viridis', alpha=0.8)
                    ax2.set_title("3D Depth Surface (Sample)")
                    ax2.set_xlabel("X")
                    ax2.set_ylabel("Y")
                    ax2.set_zlabel("Depth")
                except Exception as e:
                    # 3D plot başarısız olursa 2D contour plot göster
                    ax2.contourf(sample_depth, cmap='viridis', levels=20)
                    ax2.set_title("2D Depth Contour (Fallback)")
                    ax2.set_xlabel("X")
                    ax2.set_ylabel("Y")
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Derinlik kalitesi değerlendirmesi
                st.subheader("🎯 Derinlik Kalitesi Değerlendirmesi")
                
                # Kalite metrikleri
                depth_contrast = depth_map.std()
                depth_range = depth_map.max() - depth_map.min()
                depth_smoothness = 1.0 / (1.0 + depth_features.get('surface_complexity', 0))
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if depth_contrast > 0.1:
                        st.success(f"✅ **Yüksek Kontrast**: {depth_contrast:.3f}")
                    elif depth_contrast > 0.05:
                        st.warning(f"⚠️ **Orta Kontrast**: {depth_contrast:.3f}")
                    else:
                        st.error(f"❌ **Düşük Kontrast**: {depth_contrast:.3f}")
                
                with col2:
                    if depth_range > 0.5:
                        st.success(f"✅ **Geniş Derinlik Aralığı**: {depth_range:.3f}")
                    elif depth_range > 0.2:
                        st.warning(f"⚠️ **Orta Derinlik Aralığı**: {depth_range:.3f}")
                    else:
                        st.error(f"❌ **Dar Derinlik Aralığı**: {depth_range:.3f}")
                
                with col3:
                    if depth_smoothness > 0.7:
                        st.success(f"✅ **Yumuşak Yüzey**: {depth_smoothness:.3f}")
                    elif depth_smoothness > 0.4:
                        st.warning(f"⚠️ **Orta Yüzey**: {depth_smoothness:.3f}")
                    else:
                        st.error(f"❌ **Karmaşık Yüzey**: {depth_smoothness:.3f}")
                
            except Exception as e:
                st.error(f"❌ Derinlik analizi hatası: {e}")
        else:
            st.info("📸 Derinlik analizi için önce bir görüntü yükleyin.")
    
    with tab3:
        st.header("📊 Sistem Durumu")
        
        # Model bilgileri
        st.subheader("🤖 Hibrit Model Bilgileri")
        
        if 'autoencoder' in models:
            total_params = sum(p.numel() for p in models['autoencoder'].parameters())
            model_size_mb = total_params * 4 / (1024 * 1024)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Autoencoder Parametreleri", f"{total_params:,}")
            
            with col2:
                st.metric("Autoencoder Boyutu", f"{model_size_mb:.2f} MB")
            
            with col3:
                st.metric("Latent Boyutu", "1024")
        
        if 'classifier' in models:
            classifier_params = sum(p.numel() for p in models['classifier'].parameters())
            classifier_size_mb = classifier_params * 4 / (1024 * 1024)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Sınıflandırıcı Parametreleri", f"{classifier_params:,}")
            
            with col2:
                st.metric("Sınıflandırıcı Boyutu", f"{classifier_size_mb:.2f} MB")
            
            with col3:
                st.metric("Sınıf Sayısı", "5")
        
        # Eğitim verisi analizi
        st.subheader("📈 Eğitim Verisi")
        
        data_dir = Path("mars_images")
        categories = {}
        total_images = 0
        
        if data_dir.exists():
            for split in ['train', 'valid']:
                split_dir = data_dir / split
                if split_dir.exists():
                    for category_dir in split_dir.iterdir():
                        if category_dir.is_dir():
                            category = category_dir.name
                            image_count = len(list(category_dir.glob("*.jpg"))) + len(list(category_dir.glob("*.png")))
                            categories[category] = categories.get(category, 0) + image_count
                            total_images += image_count
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Toplam Görüntü", total_images)
            st.metric("Kategori Sayısı", len(categories))
        
        with col2:
            # Kategori dağılımı grafiği
            if categories:
                fig = px.pie(
                    values=list(categories.values()),
                    names=list(categories.keys()),
                    title="Kategori Dağılımı"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("🎯 Demo Veriler")
        
        # Demo görüntüleri
        st.subheader("📸 Test Görüntüleri")
        
        # Curiosity verilerinden örnekler
        data_dir = Path("mars_images/valid")
        demo_images = []
        
        if data_dir.exists():
            for category_dir in data_dir.iterdir():
                if category_dir.is_dir():
                    image_files = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
                    if image_files:
                        demo_images.append((category_dir.name, str(image_files[0])))
                        if len(demo_images) >= 6:
                            break
        
        if demo_images:
            # Demo görüntülerini göster
            cols = st.columns(3)
            
            for i, (category, img_path) in enumerate(demo_images):
                with cols[i % 3]:
                    image = Image.open(img_path)
                    st.image(image, caption=f"{category}", use_container_width=True)
                    
                    # Hızlı analiz butonu
                    if st.button(f"🔍 {category} Hibrit Analiz", key=f"demo_{i}"):
                        with st.spinner(f"{category} hibrit analiz ediliyor..."):
                            results = analyze_mars_image(models, image)
                            if results['anomaly_score'] is not None:
                                st.success(f"Anomali: {results['anomaly_score']:.6f}")
                                st.success(f"Bilinen Değer: {results['known_value_score']:.3f}")
                                
                                # İlginçlik puanı
                                curiosity_score = alpha * results['known_value_score'] + beta * results['anomaly_score']
                                st.metric("İlginçlik Puanı", f"{curiosity_score:.6f}")
    
    with tab5:
        st.header("ℹ️ ARTPS Hibrit Sistem Hakkında")
        
        st.markdown("""
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
        """)

if __name__ == "__main__":
    main() 