"""
ARTPS - Derinlik Tahmin Modülü
MiDaS tabanlı monocular depth estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from PIL import Image
from typing import Tuple, Optional, Dict, Any, List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MiDaSDepthEstimator:
    """
    MiDaS tabanlı derinlik tahmin modülü
    Tek kamera görüntüsünden derinlik haritası üretir
    """
    
    def __init__(self, model_type: str = "DPT_Large", device: str = "auto", enhancement_enabled: bool = True):
        """
        Args:
            model_type: MiDaS model tipi ("DPT_Large", "DPT_Hybrid", "MiDaS_small")
            device: Cihaz ("auto", "cuda", "cpu")
        """
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == "auto" else device)
        
        # HF boru hattı kullanılıyor mu?
        self.use_hf = False
        self.hf_processor = None
        
        # Geliştirme (kontrast/kenar keskinleştirme) bayrağı
        self.enhancement_enabled = enhancement_enabled
        # Kenar rehberli ince detay iyileştirme
        self.edge_refine_enabled: bool = True
        # Gelişmiş refine parametreleri (varsayılanlar)
        self.refine_params: Dict[str, Any] = {
            'gf_radius': 8,
            'gf_eps': 1e-2,
            'jbf_d': 9,
            'jbf_sigma_color': 25,
            'jbf_sigma_space': 25,
            'fgs_lambda': 500.0,
            'fgs_sigma_color': 1.5,
            'wmf_radius': 7,
            'wmf_sigma': 25.5,
        }

        # MiDaS/DPT modelini yükle
        self.model = self._load_midas_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Transform parametreleri
        self.transform = self._get_transform()
        
        print(f"✅ MiDaS {model_type} modeli yüklendi ({self.device})")

    def set_enhancement_enabled(self, enabled: bool) -> None:
        """Derinlik iyileştirmesini aç/kapat.
        Args:
            enabled: True ise enhancement uygulanır.
        """
        self.enhancement_enabled = bool(enabled)

    def set_edge_refine_enabled(self, enabled: bool) -> None:
        """Kenar rehberli derinlik iyileştirmesini aç/kapat."""
        self.edge_refine_enabled = bool(enabled)

    def set_refine_params(self, **kwargs: Any) -> None:
        """Refine parametrelerini güncelle (guided/joint bilateral/FGS/WMF)."""
        self.refine_params.update(kwargs)
    
    def _load_midas_model(self) -> nn.Module:
        """MiDaS/DPT modelini yükle ve DPT_Large'ı zorla.

        Sıra:
        1) Yerel TorchScript (.pt) → torch.jit.load
        2) Mimari (pretrained=False) + state_dict (yerel)
        3) PyTorch Hub (intel-isl/MiDaS)
        4) Hugging Face (Intel/dpt-large)
        5) Basit CNN fallback
        """
        # 1-3: MiDaS yükleme yolları
        try:
            if self.model_type == "DPT_Large":
                local_path = "raw_models/dpt_large_384.pt"
                if os.path.exists(local_path):
                    file_size = os.path.getsize(local_path)
                    print(f"🔄 {self.model_type} yerel dosya bulundu ({file_size/1024/1024:.1f} MB): {local_path}")
                    if file_size < 800 * 1024 * 1024:
                        print("⚠️ Yerel DPT_Large dosyası beklenenden küçük; yeniden indirilmesi gerekebilir.")
                    # 1) TorchScript
                    try:
                        print("🔍 TorchScript (jit) deneniyor...")
                        scripted = torch.jit.load(local_path, map_location="cpu")
                        print("✅ TorchScript model yüklendi")
                        return scripted
                    except Exception as jit_err:
                        print(f"⚠️ TorchScript başarısız: {jit_err}")
                    # 2) Mimari + state_dict
                    try:
                        model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", pretrained=False, trust_repo=True)
                        weights = torch.load(local_path, map_location="cpu")
                        model.load_state_dict(weights)
                        print("✅ Mimari + yerel ağırlıklar ile yüklendi")
                        return model
                    except Exception as local_err:
                        print(f"⚠️ Yerel ağırlık yükleme başarısız: {local_err}")
                # 3) Hub
                print("🔄 PyTorch Hub üzerinden DPT_Large yükleniyor...")
                model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
                print("✅ DPT_Large (Hub) yüklendi")
                return model
            # Diğer tipler
            print(f"🔄 {self.model_type} Hub'dan yükleniyor...")
            model = torch.hub.load("intel-isl/MiDaS", self.model_type, trust_repo=True)
            print(f"✅ {self.model_type} yüklendi")
            return model
        except Exception as hub_error:
            print(f"⚠️ MiDaS yükleme başarısız: {hub_error}")

        # 4: Hugging Face alternatifi (devre dışı: TF ve NumPy 2.x zincir import sorunları)
        # Tasarım gereği, bu yol kapatıldı. MiDaS Hub ve yerel ağırlıklar önceliklidir.

        # 5: Basit fallback
        print("⚠️ Basit derinlik modeli kullanılıyor")
        return self._create_simple_depth_model()
    
    def _create_simple_depth_model(self) -> nn.Module:
        """Basit derinlik tahmin modeli (fallback)"""
        print("⚠️ Basit derinlik modeli kullanılıyor")
        
        model = nn.Sequential(
            # Encoder
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Decoder
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # Derinlik çıkışı
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        return model
    
    def _get_transform(self) -> Dict[str, Any]:
        """Görüntü transform parametreleri"""
        try:
            # PyTorch Hub'dan transform'ları yükle
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            
            if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
                transform = midas_transforms.dpt_transform
            else:
                transform = midas_transforms.small_transform
                
            return {
                'transform': transform,
                'input_size': (384, 384)  # MiDaS için standart boyut
            }
            
        except Exception as e:
            print(f"⚠️ Transform yükleme hatası, varsayılan kullanılıyor: {e}")
            return {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'input_size': (384, 384)
            }
    
    def preprocess_image(self, image: np.ndarray):
        """
        Görüntüyü model için hazırla (PyTorch Hub transform'ları ile)
        
        Args:
            image: RGB görüntü (H, W, C)
            
        Returns:
            Tensor: Hazırlanmış görüntü tensor'ı
        """
        try:
            # HF yolu: processor kullan
            if self.use_hf and self.hf_processor is not None:
                if isinstance(image, np.ndarray):
                    arr = image
                    if arr.dtype != np.uint8:
                        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
                    if len(arr.shape) == 3 and arr.shape[2] == 3:
                        pil_img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
                    else:
                        pil_img = Image.fromarray(arr)
                elif isinstance(image, Image.Image):
                    pil_img = image
                else:
                    raise ValueError("Desteklenmeyen görüntü formatı")

                encoded = self.hf_processor(images=pil_img, return_tensors="pt")
                return {"pixel_values": encoded["pixel_values"].to(self.device)}

            # PyTorch Hub transform'unu kullan
            if 'transform' in self.transform:
                # MiDaS transform'u BGR uint8 bekler; girdiyi BGR uint8'e çevir
                if isinstance(image, Image.Image):
                    img_rgb = np.array(image.convert('RGB'))
                else:
                    img_rgb = np.array(image) if not isinstance(image, np.ndarray) else image

                # [0,1] float ise [0,255] uint8'e çevir
                if img_rgb.dtype != np.uint8:
                    img_rgb = np.clip(img_rgb * 255.0, 0, 255).astype(np.uint8)

                # RGB -> BGR
                if len(img_rgb.shape) == 3 and img_rgb.shape[2] == 3:
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                else:
                    img_bgr = img_rgb

                input_batch = self.transform['transform'](img_bgr)
                return input_batch.to(self.device)
            
            else:
                # Fallback: Manuel transform
                # PIL Image'e çevir
                if isinstance(image, np.ndarray):
                    # uint8'den float32'ye çevir
                    if image.dtype == np.uint8:
                        image = (image / 255.0).astype(np.float32)
                    image = Image.fromarray((image * 255).astype(np.uint8))
                
                # Boyutlandır
                image = image.resize(self.transform['input_size'], Image.LANCZOS)
                
                # NumPy array'e çevir
                image_array = np.array(image, dtype=np.float32) / 255.0
                
                # Normalize et
                mean = np.array(self.transform['mean']).reshape(1, 1, 3)
                std = np.array(self.transform['std']).reshape(1, 1, 3)
                image_array = (image_array - mean) / std
                
                # Tensor'a çevir ve boyut ekle (float32 olarak)
                tensor = torch.from_numpy(image_array).float().permute(2, 0, 1).unsqueeze(0)
                
                return tensor.to(self.device)
                
        except Exception as e:
            print(f"⚠️ Transform hatası, manuel işlem kullanılıyor: {e}")
            # Manuel fallback
            if isinstance(image, np.ndarray):
                if image.dtype == np.uint8:
                    image = (image / 255.0).astype(np.float32)
                image = Image.fromarray((image * 255).astype(np.uint8))
            
            image = image.resize(self.transform['input_size'], Image.LANCZOS)
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            mean = np.array(self.transform['mean']).reshape(1, 1, 3)
            std = np.array(self.transform['std']).reshape(1, 1, 3)
            image_array = (image_array - mean) / std
            
            tensor = torch.from_numpy(image_array).float().permute(2, 0, 1).unsqueeze(0)
            return tensor.to(self.device)
    
    def estimate_depth(self, image: np.ndarray, apply_enhancement: Optional[bool] = None,
                       guide_image: Optional[np.ndarray] = None, high_detail: bool = False,
                       tta_flips: bool = False, use_fgs: bool = False, use_wmf: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Görüntüden derinlik tahmini yap (PyTorch Hub uyumlu)
        
        Args:
            image: RGB görüntü (H, W, C)
            
        Returns:
            Tuple[ndarray, dict]: Derinlik haritası ve metadata
        """
        try:
            # Derinlik tahmini (opsiyonel flip TTA)
            def _predict_from_img(img: np.ndarray) -> torch.Tensor:
                inp = self.preprocess_image(img)
                with torch.no_grad():
                    if self.use_hf:
                        outputs = self.model(pixel_values=inp["pixel_values"])  # type: ignore
                        pred = outputs.predicted_depth
                    else:
                        pred = self.model(inp)
                if not self.use_hf:
                    if len(pred.shape) == 4:
                        pred = pred.squeeze(0)
                    if len(pred.shape) == 3:
                        pred = pred[0]
                    pred = pred.unsqueeze(0).unsqueeze(0)
                return pred

            prediction = _predict_from_img(image)
            if tta_flips:
                # Yatay flip TTA
                img_flip = np.ascontiguousarray(np.flip(image, axis=1))
                pred_flip = _predict_from_img(img_flip)
                # Geri çevir (genişlik boyunca)
                pred_flip = torch.flip(pred_flip, dims=[-1])
                prediction = 0.5 * (prediction + pred_flip)

            # Orijinal boyuta yeniden boyutlandır
            original_height, original_width = image.shape[:2]

            # Prediction'ın boyutlarını normalize et ve yeniden boyutlandır
            prediction = torch.nn.functional.interpolate(
                prediction,
                size=(original_height, original_width),
                mode="bicubic",
                align_corners=False,
            )
            prediction = prediction.squeeze()

            # NumPy'a çevir
            depth_map = prediction.cpu().numpy()

            # İyileştirme kontrolü
            should_enhance = self.enhancement_enabled if apply_enhancement is None else bool(apply_enhancement)

            if should_enhance:
                # Geliştirilmiş normalizasyon ve post-process
                depth_map = self._enhance_depth_map(depth_map)
                depth_map = self._post_process_depth(depth_map)
            else:
                # Ham DPT çıktısını min-max normalize et
                dmin, dmax = float(depth_map.min()), float(depth_map.max())
                depth_map = (depth_map - dmin) / (dmax - dmin + 1e-8)

            # İnce detay modu: Kenar rehberli iyileştirme (guided/joint bilateral)
            if self.edge_refine_enabled or high_detail:
                try:
                    guide = image if guide_image is None else guide_image
                    depth_map = self._edge_aware_refine(depth_map, guide, strength=0.6 if high_detail else 0.4)
                    # Ek gelişmiş filtreler (opsiyonel)
                    if use_fgs and hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'createFastGlobalSmootherFilter'):
                        depth_map = self._fast_global_smoother(depth_map, guide)
                    if use_wmf and hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'weightedMedianFilter'):
                        depth_map = self._weighted_median_refine(depth_map, guide)
                except Exception:
                    pass
            
            # Metadata (güvenli hesapla)
            try:
                dmin = float(depth_map.min())
                dmax = float(depth_map.max())
                dmean = float(depth_map.mean())
                dstd = float(depth_map.std())
            except Exception:
                dmin, dmax, dmean, dstd = 0.0, 1.0, 0.5, 0.0
            metadata = {
                'model_type': self.model_type,
                'input_size': self.transform.get('input_size', (384, 384)),
                'depth_range': (dmin, dmax),
                'depth_mean': dmean,
                'depth_std': dstd,
                'enhancement_applied': bool(self.enhancement_enabled if apply_enhancement is None else apply_enhancement),
                'edge_refine': bool(self.edge_refine_enabled or high_detail)
            }
            
            return depth_map, metadata
            
        except Exception as e:
            print(f"❌ Derinlik tahmin hatası: {e}")
            # Fallback: Basit derinlik haritası
            fallback_depth = self._create_fallback_depth(image)
            return fallback_depth, {
                'error': str(e),
                'model_type': 'fallback',
                'depth_range': (fallback_depth.min(), fallback_depth.max()),
                'depth_mean': fallback_depth.mean(),
                'depth_std': fallback_depth.std(),
                'enhancement_applied': False
            }
    
    def _enhance_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        """Derinlik haritasını geliştir (kontrast ve detay artırma)"""
        
        # Histogram eşitleme benzeri işlem
        depth_flat = depth_map.flatten()
        
        # Percentile-based normalization (outlier'ları daha az kırp)
        p2, p98 = np.percentile(depth_flat, [2, 98])
        depth_clipped = np.clip(depth_map, p2, p98)
        
        # Normalize et
        depth_normalized = (depth_clipped - p2) / (p98 - p2 + 1e-8)
        
        # Gamma correction (kontrast artırma - daha agresif)
        gamma = 0.85
        depth_enhanced = np.power(depth_normalized, gamma)
        
        # Kenar keskinleştirme
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        depth_sharpened = cv2.filter2D(depth_enhanced, -1, kernel)
        
        # Ağırlıklı ortalama (daha yumuşak keskinleştirme)
        depth_final = 0.8 * depth_enhanced + 0.2 * depth_sharpened
        
        return np.clip(depth_final, 0, 1)
    
    def _post_process_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """Derinlik haritası post-processing"""
        
        # Bilateral filter (kenarları koruyarak gürültü azaltma)
        depth_filtered = cv2.bilateralFilter(
            (depth_map * 255).astype(np.uint8), 
            d=15, sigmaColor=75, sigmaSpace=75
        ).astype(np.float32) / 255.0
        
        # Morphological operations (küçük gürültüleri temizle)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        depth_cleaned = cv2.morphologyEx(
            (depth_filtered * 255).astype(np.uint8), 
            cv2.MORPH_CLOSE, kernel
        ).astype(np.float32) / 255.0
        
        return depth_cleaned

    def _edge_aware_refine(self, depth_map: np.ndarray, guide_rgb: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """Orijinal görüntüyü rehber alarak derinliği kenar-uyumlu şekilde keskinleştirir.

        Önce ortak-bilateral filtre ile kenarları hizalar, ardından guided filter uygular.
        """
        depth = np.clip(depth_map.astype(np.float32), 0.0, 1.0)
        if guide_rgb is None:
            return depth
        if guide_rgb.dtype != np.uint8:
            guide = np.clip(guide_rgb * (255.0 if guide_rgb.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        else:
            guide = guide_rgb

        # Ortak-bilateral: rehber gri ile
        guide_gray = cv2.cvtColor(guide, cv2.COLOR_RGB2GRAY)
        joint = cv2.ximgproc.jointBilateralFilter(guide_gray, (depth * 255).astype(np.uint8), d=9, sigmaColor=25, sigmaSpace=25) if hasattr(cv2, 'ximgproc') else None
        if joint is not None:
            depth_joint = joint.astype(np.float32) / 255.0
        else:
            depth_joint = cv2.bilateralFilter((depth * 255).astype(np.uint8), d=9, sigmaColor=25, sigmaSpace=25).astype(np.float32) / 255.0

        # Guided filter (varsa) ile daha da hizala
        if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'guidedFilter'):
            gf = cv2.ximgproc.guidedFilter(guide.astype(np.uint8), (depth_joint * 255).astype(np.uint8), radius=8, eps=1e-2)
            depth_gf = gf.astype(np.float32) / 255.0
        else:
            depth_gf = depth_joint

        # Kenar vurgusu: Canny ile kenarları çıkar ve lokal kontrast artır
        edges = cv2.Canny(guide_gray, threshold1=50, threshold2=150)
        edges = cv2.GaussianBlur(edges.astype(np.float32), (3, 3), 0) / 255.0
        enhanced = np.clip(depth_gf + strength * (edges * (depth_gf - cv2.GaussianBlur(depth_gf, (0, 0), 1.0))), 0.0, 1.0)
        return enhanced
    
    def _create_fallback_depth(self, image: np.ndarray) -> np.ndarray:
        """Geliştirilmiş basit derinlik haritası (fallback)"""
        # Gri tonlamaya çevir
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Float32'ye çevir
        gray_float = gray.astype(np.float32)
        
        # Gelişmiş gradient hesaplama
        grad_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=5)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Laplacian (kenar tespiti)
        laplacian = cv2.Laplacian(gray_float, cv2.CV_32F, ksize=3)
        
        # Kombine derinlik haritası
        depth_map = (0.7 * gradient_magnitude + 0.3 * np.abs(laplacian))
        
        # Normalize ve geliştir
        depth_map = depth_map / (depth_map.max() + 1e-8)
        depth_map = self._enhance_depth_map(depth_map)
        
        return depth_map
    
    def extract_depth_features(self, depth_map: np.ndarray) -> Dict[str, float]:
        """
        Derinlik haritasından özellikler çıkar
        
        Args:
            depth_map: Derinlik haritası
            
        Returns:
            Dict: Derinlik özellikleri
        """
        features = {}
        
        # Temel istatistikler
        features['depth_mean'] = float(np.mean(depth_map))
        features['depth_std'] = float(np.std(depth_map))
        features['depth_min'] = float(np.min(depth_map))
        features['depth_max'] = float(np.max(depth_map))
        
        # Derinlik dağılımı
        features['depth_median'] = float(np.median(depth_map))
        features['depth_percentile_25'] = float(np.percentile(depth_map, 25))
        features['depth_percentile_75'] = float(np.percentile(depth_map, 75))
        
        # Derinlik varyasyonu
        features['depth_variance'] = float(np.var(depth_map))
        features['depth_skewness'] = float(self._calculate_skewness(depth_map))
        features['depth_kurtosis'] = float(self._calculate_kurtosis(depth_map))
        
        # Yüzey karmaşıklığı
        features['surface_complexity'] = float(self._calculate_surface_complexity(depth_map))
        
        # Derinlik gradyanı
        depth_map_float = depth_map.astype(np.float32)
        grad_x = cv2.Sobel(depth_map_float, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_map_float, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        features['depth_gradient_mean'] = float(np.mean(gradient_magnitude))
        features['depth_gradient_std'] = float(np.std(gradient_magnitude))
        features['depth_gradient_max'] = float(np.max(gradient_magnitude))

        # Yüzey normalleri (yaklaşık): n = (-dz/dx, -dz/dy, 1) / ||.||
        nx = -grad_x
        ny = -grad_y
        nz = np.ones_like(depth_map_float)
        norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
        nx /= norm; ny /= norm; nz /= norm
        features['normal_z_mean'] = float(np.mean(nz))
        features['normal_z_std'] = float(np.std(nz))
        # Normal yön dağılımı (küresel varyans proxy)
        features['normal_dir_var'] = float(np.mean((1.0 - nz) ** 2))

        # Eğrilik (yaklaşık): Laplasyen ve Hessian temelli büyüklük
        dxx = cv2.Sobel(depth_map_float, cv2.CV_32F, 2, 0, ksize=3)
        dyy = cv2.Sobel(depth_map_float, cv2.CV_32F, 0, 2, ksize=3)
        dxy = cv2.Sobel(depth_map_float, cv2.CV_32F, 1, 1, ksize=3)
        curvature_mag = np.sqrt(np.maximum(0.0, dxx**2 + 2.0 * dxy**2 + dyy**2))
        lap = cv2.Laplacian(depth_map_float, cv2.CV_32F, ksize=3)
        features['curvature_mean'] = float(np.mean(curvature_mag))
        features['curvature_std'] = float(np.std(curvature_mag))
        features['curvature_max'] = float(np.max(curvature_mag))
        features['laplace_abs_mean'] = float(np.mean(np.abs(lap)))

        # Planarlık (yaklaşık): Yerel varyans tersine dayalı skor (yüksek = daha düz)
        k = 7
        mean_local = cv2.boxFilter(depth_map_float, ddepth=-1, ksize=(k, k), normalize=True)
        mean_sq_local = cv2.boxFilter(depth_map_float * depth_map_float, ddepth=-1, ksize=(k, k), normalize=True)
        var_local = np.clip(mean_sq_local - mean_local * mean_local, 0.0, None)
        planarity = 1.0 / (1.0 + var_local)
        features['planarity_mean'] = float(np.mean(planarity))
        features['planarity_min'] = float(np.min(planarity))

        # Pürüzlülük (roughness): gradyan std + laplace std birleşik
        features['roughness'] = float(0.5 * np.std(gradient_magnitude) + 0.5 * np.std(lap))
        
        return features
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Çarpıklık hesapla"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Basıklık hesapla"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_surface_complexity(self, depth_map: np.ndarray) -> float:
        """Yüzey karmaşıklığı hesapla"""
        # Float32'ye çevir
        depth_map_float = depth_map.astype(np.float32)
        # Laplacian hesapla
        laplacian = cv2.Laplacian(depth_map_float, cv2.CV_32F)
        return float(np.std(laplacian))
    
    def visualize_depth(self, image: np.ndarray, depth_map: np.ndarray, 
                       save_path: Optional[str] = None) -> None:
        """
        Derinlik haritasını görselleştir
        
        Args:
            image: Orijinal görüntü
            depth_map: Derinlik haritası
            save_path: Kaydetme yolu (opsiyonel)
        """
        fig = plt.figure(figsize=(15, 5))
        
        # Orijinal görüntü
        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(image)
        ax1.set_title('Orijinal Görüntü')
        ax1.axis('off')
        
        # Derinlik haritası
        ax2 = plt.subplot(1, 3, 2)
        im1 = ax2.imshow(depth_map, cmap='plasma')
        ax2.set_title('Derinlik Haritası')
        ax2.axis('off')
        plt.colorbar(im1, ax=ax2)
        
        # 3D yüzey (sadece 2D histogram olarak göster)
        ax3 = plt.subplot(1, 3, 3)
        ax3.hist(depth_map.flatten(), bins=50, alpha=0.7, color='blue')
        ax3.set_title('Derinlik Dağılımı')
        ax3.set_xlabel('Derinlik Değeri')
        ax3.set_ylabel('Frekans')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Derinlik görselleştirmesi kaydedildi: {save_path}")
        
        plt.show()

    def _fast_global_smoother(self, depth_map: np.ndarray, guide_rgb: np.ndarray) -> np.ndarray:
        guide = guide_rgb
        if guide.dtype != np.uint8:
            guide = np.clip(guide * (255.0 if guide.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        src = (np.clip(depth_map, 0.0, 1.0) * 255).astype(np.uint8)
        p = self.refine_params
        try:
            fgs = cv2.ximgproc.createFastGlobalSmootherFilter(guide, float(p['fgs_lambda']), float(p['fgs_sigma_color']))
            out = fgs.filter(src)
            return np.clip(out.astype(np.float32) / 255.0, 0.0, 1.0)
        except Exception:
            return depth_map

    def _weighted_median_refine(self, depth_map: np.ndarray, guide_rgb: np.ndarray) -> np.ndarray:
        guide = guide_rgb
        if guide.dtype != np.uint8:
            guide = np.clip(guide * (255.0 if guide.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        src_u8 = (np.clip(depth_map, 0.0, 1.0) * 255).astype(np.uint8)
        p = self.refine_params
        try:
            out = cv2.ximgproc.weightedMedianFilter(guide, src_u8, int(p['wmf_radius']), sigma=float(p['wmf_sigma']))
            return np.clip(out.astype(np.float32) / 255.0, 0.0, 1.0)
        except Exception:
            return depth_map

def test_depth_estimation():
    """Derinlik tahmin modülünü test et"""
    print("🧪 Derinlik Tahmin Modülü Test Ediliyor...")
    
    # Test görüntüsü oluştur
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Derinlik tahmin modülünü oluştur
    depth_estimator = MiDaSDepthEstimator(model_type="DPT_Large")
    
    # Derinlik tahmini yap
    depth_map, metadata = depth_estimator.estimate_depth(test_image)
    
    print(f"✅ Derinlik tahmini tamamlandı")
    print(f"📊 Metadata: {metadata}")
    
    # Özellikler çıkar
    features = depth_estimator.extract_depth_features(depth_map)
    print(f"🎯 Derinlik özellikleri: {len(features)} özellik")
    
    # Görselleştir
    depth_estimator.visualize_depth(test_image, depth_map, 
                                   save_path="results/depth_estimation_test.png")
    
    return depth_estimator, depth_map, features

if __name__ == "__main__":
    test_depth_estimation() 