#!/usr/bin/env python3
"""
Sadece Autoencoder Fark Normalizasyonu
Bu script, görüntüyü yükleyip Autoencoder ile yeniden oluşturarak fark haritasını normalize eder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

class SimpleAutoencoder(nn.Module):
    """Basit Autoencoder modeli"""
    def __init__(self, input_channels=3):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def load_image(image_path, target_size=(256, 256)):
    """Görüntüyü yükle ve ön işle"""
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path
    
    # Boyutlandır
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # PIL -> numpy -> torch
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor, image_np

def compute_ae_diff(image_tensor, model, device='cpu'):
    """Autoencoder fark haritasını hesapla"""
    model.eval()
    with torch.no_grad():
        # Görüntüyü modele gönder
        reconstructed = model(image_tensor.to(device))
        
        # Fark hesapla (MSE)
        diff = F.mse_loss(image_tensor.to(device), reconstructed, reduction='none')
        diff = diff.mean(dim=1, keepdim=True)  # RGB kanallarını birleştir
        
        # CPU'ya taşı ve numpy'a çevir
        diff_np = diff.squeeze().cpu().numpy()
        
        return diff_np, reconstructed.squeeze().permute(1, 2, 0).cpu().numpy()

def normalize_diff(diff_map, method='minmax'):
    """Fark haritasını normalize et"""
    if method == 'minmax':
        # 0-1 aralığına normalize et
        diff_min = diff_map.min()
        diff_max = diff_map.max()
        if diff_max > diff_min:
            normalized = (diff_map - diff_min) / (diff_max - diff_min)
        else:
            normalized = np.zeros_like(diff_map)
    
    elif method == 'zscore':
        # Z-score normalizasyonu
        mean = diff_map.mean()
        std = diff_map.std()
        if std > 0:
            normalized = (diff_map - mean) / std
            # 0-1 aralığına sıkıştır
            normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
        else:
            normalized = np.zeros_like(diff_map)
    
    elif method == 'robust':
        # Robust normalizasyon (outlier'lara dayanıklı)
        q25 = np.percentile(diff_map, 25)
        q75 = np.percentile(diff_map, 75)
        iqr = q75 - q25
        if iqr > 0:
            normalized = (diff_map - q25) / iqr
            normalized = np.clip(normalized, 0, 1)
        else:
            normalized = np.zeros_like(diff_map)
    
    else:
        raise ValueError(f"Bilinmeyen normalizasyon yöntemi: {method}")
    
    return normalized

def visualize_results(original, reconstructed, diff_map, normalized_diff, save_path=None):
    """Sonuçları görselleştir"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Orijinal görüntü
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Orijinal Görüntü')
    axes[0, 0].axis('off')
    
    # Yeniden oluşturulan görüntü
    axes[0, 1].imshow(reconstructed)
    axes[0, 1].set_title('AE Yeniden Oluşturma')
    axes[0, 1].axis('off')
    
    # Ham fark haritası
    im1 = axes[1, 0].imshow(diff_map, cmap='hot')
    axes[1, 0].set_title('Ham Fark Haritası')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0])
    
    # Normalize edilmiş fark haritası
    im2 = axes[1, 1].imshow(normalized_diff, cmap='hot')
    axes[1, 1].set_title('Normalize Fark Haritası')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Görselleştirme kaydedildi: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Autoencoder Fark Normalizasyonu')
    parser.add_argument('--image', type=str, required=True, help='Görüntü dosyası yolu')
    parser.add_argument('--model', type=str, help='Önceden eğitilmiş model yolu (opsiyonel)')
    parser.add_argument('--size', type=int, nargs=2, default=[256, 256], help='Görüntü boyutu (genişlik yükseklik)')
    parser.add_argument('--device', type=str, default='auto', help='Cihaz (cpu, cuda, auto)')
    parser.add_argument('--normalize', type=str, default='minmax', 
                       choices=['minmax', 'zscore', 'robust'], help='Normalizasyon yöntemi')
    parser.add_argument('--save', type=str, help='Görselleştirme kaydetme yolu')
    parser.add_argument('--output', type=str, help='Normalize fark haritası kaydetme yolu (.npy)')
    
    args = parser.parse_args()
    
    # Cihaz seçimi
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Cihaz: {device}")
    print(f"Görüntü boyutu: {args.size}")
    print(f"Normalizasyon: {args.normalize}")
    
    # Görüntüyü yükle
    print(f"Görüntü yükleniyor: {args.image}")
    image_tensor, original_np = load_image(args.image, tuple(args.size))
    
    # Model yükle veya oluştur
    if args.model and Path(args.model).exists():
        print(f"Model yükleniyor: {args.model}")
        model = SimpleAutoencoder()
        model.load_state_dict(torch.load(args.model, map_location=device))
    else:
        print("Yeni model oluşturuluyor (eğitilmemiş)")
        model = SimpleAutoencoder()
    
    model = model.to(device)
    
    # Fark hesapla
    print("Autoencoder fark hesaplanıyor...")
    diff_map, reconstructed = compute_ae_diff(image_tensor, model, device)
    
    # Normalize et
    print("Fark haritası normalize ediliyor...")
    normalized_diff = normalize_diff(diff_map, args.normalize)
    
    # İstatistikler
    print(f"\nFark Haritası İstatistikleri:")
    print(f"Ham - Min: {diff_map.min():.6f}, Max: {diff_map.max():.6f}, Mean: {diff_map.mean():.6f}")
    print(f"Normalize - Min: {normalized_diff.min():.6f}, Max: {normalized_diff.max():.6f}, Mean: {normalized_diff.mean():.6f}")
    
    # Görselleştir
    print("\nGörselleştirme oluşturuluyor...")
    visualize_results(original_np, reconstructed, diff_map, normalized_diff, args.save)
    
    # Normalize fark haritasını kaydet
    if args.output:
        np.save(args.output, normalized_diff)
        print(f"Normalize fark haritası kaydedildi: {args.output}")
    
    print("Tamamlandı!")

if __name__ == "__main__":
    main()
