"""
ARTPS - Veri İşleme Yardımcı Fonksiyonları

Bu modül, Mars kaya görüntülerinin işlenmesi ve hazırlanması için
yardımcı fonksiyonları içerir.
"""

import os
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
import json


def load_image(image_path: str, target_size: Tuple[int, int] = (128, 128)) -> torch.Tensor:
    """
    Görüntüyü yükle ve tensor'a dönüştür
    
    Args:
        image_path: Görüntü dosya yolu
        target_size: Hedef görüntü boyutu (height, width)
        
    Returns:
        torch.Tensor: Normalize edilmiş görüntü tensor'ı [channels, height, width]
    """
    # Görüntüyü yükle
    image = Image.open(image_path).convert('RGB')
    
    # Boyutu ayarla
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Tensor'a dönüştür
    image = torch.from_numpy(np.array(image)).float()
    image = image.permute(2, 0, 1)  # HWC -> CHW
    image = image / 255.0  # [0, 255] -> [0, 1]
    
    return image


def extract_features(image: torch.Tensor) -> Dict[str, np.ndarray]:
    """
    Görüntüden özellik çıkarımı yap
    
    Args:
        image: Görüntü tensor'ı [channels, height, width]
        
    Returns:
        Dict[str, np.ndarray]: Çıkarılan özellikler
    """
    # Tensor'ı numpy array'e dönüştür
    img_np = image.permute(1, 2, 0).numpy()
    
    features = {}
    
    # Renk özellikleri
    features['mean_color'] = np.mean(img_np, axis=(0, 1))  # RGB ortalama
    features['std_color'] = np.std(img_np, axis=(0, 1))    # RGB standart sapma
    
    # Gri tonlama
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    features['mean_gray'] = np.mean(gray)
    features['std_gray'] = np.std(gray)
    
    # Doku özellikleri (GLCM benzeri basit özellikler)
    # Gradyan hesaplama
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    features['gradient_mean'] = np.mean(gradient_magnitude)
    features['gradient_std'] = np.std(gradient_magnitude)
    
    # Histogram özellikleri
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()  # Normalize et
    
    features['histogram'] = hist
    
    return features


def calculate_similarity(features1: Dict[str, np.ndarray], 
                        features2: Dict[str, np.ndarray]) -> float:
    """
    İki görüntü arasındaki benzerliği hesapla
    
    Args:
        features1: İlk görüntünün özellikleri
        features2: İkinci görüntünün özellikleri
        
    Returns:
        float: Benzerlik skoru (0-1 arası, 1 en benzer)
    """
    # Renk benzerliği
    color_sim = 1.0 - np.linalg.norm(features1['mean_color'] - features2['mean_color'])
    color_sim = max(0, min(1, color_sim))  # [0,1] aralığına sınırla
    
    # Doku benzerliği
    texture_sim = 1.0 - abs(features1['gradient_mean'] - features2['gradient_mean'])
    texture_sim = max(0, min(1, texture_sim))
    
    # Histogram benzerliği (correlation)
    hist_corr = np.corrcoef(features1['histogram'], features2['histogram'])[0, 1]
    hist_sim = max(0, hist_corr) if not np.isnan(hist_corr) else 0
    
    # Ağırlıklı ortalama
    similarity = 0.4 * color_sim + 0.3 * texture_sim + 0.3 * hist_sim
    
    return similarity


def create_data_augmentation() -> transforms.Compose:
    """
    Veri artırma (data augmentation) dönüşümleri oluştur
    
    Returns:
        transforms.Compose: Veri artırma pipeline'ı
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def save_anomaly_results(results: List[Dict], output_path: str):
    """
    Anomali tespit sonuçlarını kaydet
    
    Args:
        results: Anomali tespit sonuçları listesi
        output_path: Çıktı dosya yolu
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_anomaly_results(input_path: str) -> List[Dict]:
    """
    Anomali tespit sonuçlarını yükle
    
    Args:
        input_path: Giriş dosya yolu
        
    Returns:
        List[Dict]: Yüklenen sonuçlar
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def visualize_anomaly_scores(image_paths: List[str], 
                           anomaly_scores: List[float], 
                           save_path: Optional[str] = None,
                           top_k: int = 10):
    """
    Anomali skorlarını görselleştir
    
    Args:
        image_paths: Görüntü dosya yolları
        anomaly_scores: Anomali skorları
        save_path: Kaydetme yolu (opsiyonel)
        top_k: En yüksek k anomali skorunu göster
    """
    # En yüksek anomali skorlarına sahip görüntüleri bul
    sorted_indices = np.argsort(anomaly_scores)[::-1][:top_k]
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, idx in enumerate(sorted_indices):
        if i >= 10:  # Maksimum 10 görüntü göster
            break
            
        # Görüntüyü yükle ve göster
        image = Image.open(image_paths[idx])
        axes[i].imshow(image)
        axes[i].set_title(f'Anomali Skoru: {anomaly_scores[idx]:.4f}')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_sample_data(data_dir: str, num_samples: int = 50, image_size: Tuple[int, int] = (128, 128)):
    """
    Gerçekçi Mars kaya görüntüleri oluştur
    
    Args:
        data_dir: Verilerin kaydedileceği dizin
        num_samples: Oluşturulacak görüntü sayısı
        image_size: Görüntü boyutu (height, width)
    """
    import os
    import numpy as np
    from PIL import Image, ImageDraw, ImageFilter
    
    # Dizini oluştur
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"{num_samples} gerçekçi Mars kaya görüntüsü oluşturuluyor...")
    
    for i in range(num_samples):
        # Mars toprağı rengi paleti (gerçek Mars renkleri)
        mars_colors = [
            (139, 69, 19),   # Kahverengi
            (160, 82, 45),   # Saddle Brown
            (205, 133, 63),  # Peru
            (210, 180, 140), # Tan
            (244, 164, 96),  # Sandy Brown
            (160, 82, 45),   # Saddle Brown
            (139, 69, 19),   # Saddle Brown
            (101, 67, 33),   # Dark Brown
        ]
        
        # Ana arka plan rengi (Mars toprağı)
        bg_color = mars_colors[np.random.randint(0, len(mars_colors))]
        
        # Görüntü oluştur
        img = Image.new('RGB', image_size, bg_color)
        draw = ImageDraw.Draw(img)
        
        # Kaya şekilleri ekle
        num_rocks = np.random.randint(3, 8)
        
        for _ in range(num_rocks):
            # Kaya pozisyonu
            x = np.random.randint(20, image_size[1] - 20)
            y = np.random.randint(20, image_size[0] - 20)
            
            # Kaya boyutu
            size = np.random.randint(15, 40)
            
            # Kaya rengi (biraz farklı ton)
            rock_color = list(bg_color)
            for j in range(3):
                rock_color[j] = max(0, min(255, rock_color[j] + np.random.randint(-30, 30)))
            rock_color = tuple(rock_color)
            
            # Kaya şekli (oval veya düzensiz)
            if np.random.random() > 0.5:
                # Oval kaya
                draw.ellipse([x-size, y-size//2, x+size, y+size//2], 
                           fill=rock_color, outline=(0, 0, 0), width=1)
            else:
                # Düzensiz kaya
                points = []
                for angle in range(0, 360, 30):
                    r = size + np.random.randint(-5, 5)
                    px = x + int(r * np.cos(np.radians(angle)))
                    py = y + int(r * np.sin(np.radians(angle)))
                    points.append((px, py))
                draw.polygon(points, fill=rock_color, outline=(0, 0, 0), width=1)
        
        # Doku ekle (küçük detaylar)
        for _ in range(50):
            x = np.random.randint(0, image_size[1])
            y = np.random.randint(0, image_size[0])
            color = list(bg_color)
            for j in range(3):
                color[j] = max(0, min(255, color[j] + np.random.randint(-20, 20)))
            draw.point((x, y), fill=tuple(color))
        
        # Hafif bulanıklaştır (gerçekçilik için)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Gürültü ekle (gerçek kamera gürültüsü simülasyonu)
        img_array = np.array(img)
        noise = np.random.normal(0, 5, img_array.shape).astype(np.uint8)
        img_array = np.clip(img_array + noise, 0, 255)
        img = Image.fromarray(img_array.astype(np.uint8))
        
        # Kaydet
        filename = f"mars_rock_{i:03d}.jpg"
        filepath = os.path.join(data_dir, filename)
        img.save(filepath, quality=95)
    
    print(f"✅ {num_samples} gerçekçi Mars kaya görüntüsü oluşturuldu: {data_dir}")


def calculate_curiosity_score(exploitation_score: float, 
                            exploration_score: float, 
                            alpha: float = 0.5) -> float:
    """
    İlginçlik puanını hesapla
    
    Args:
        exploitation_score: Bilinen değer puanı (0-1)
        exploration_score: Anomali/keşif puanı (0-1)
        alpha: Ağırlık parametresi (0-1, 0.5 varsayılan)
        
    Returns:
        float: İlginçlik puanı (0-1)
    """
    # İlginçlik puanı = α * Exploitation + (1-α) * Exploration
    curiosity_score = alpha * exploitation_score + (1 - alpha) * exploration_score
    
    return curiosity_score


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Skorları 0-1 aralığına normalize et
    
    Args:
        scores: Normalize edilecek skorlar
        
    Returns:
        List[float]: Normalize edilmiş skorlar
    """
    if not scores:
        return scores
    
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [0.5] * len(scores)  # Tüm skorlar aynıysa 0.5 ver
    
    normalized = [(score - min_score) / (max_score - min_score) for score in scores]
    return normalized 