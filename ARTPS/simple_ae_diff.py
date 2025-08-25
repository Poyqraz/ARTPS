#!/usr/bin/env python3
"""
Sadece AE Fark Görseli Oluşturucu
Bu kod, görüntüyü yükleyip Autoencoder ile yeniden oluşturarak fark haritasını gösterir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Basit encoder-decoder yapısı
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def load_and_process_image(image_path, size=(256, 256)):
    """Görüntüyü yükle ve işle"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize(size, Image.Resampling.LANCZOS)
    
    # PIL -> numpy -> torch tensor
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor, image_np

def compute_ae_difference(image_tensor, model):
    """AE fark haritasını hesapla"""
    model.eval()
    with torch.no_grad():
        # Görüntüyü yeniden oluştur
        reconstructed = model(image_tensor)
        
        # Fark hesapla (MSE)
        diff = F.mse_loss(image_tensor, reconstructed, reduction='none')
        diff = diff.mean(dim=1, keepdim=True)  # RGB kanallarını birleştir
        
        # Numpy'a çevir
        diff_np = diff.squeeze().numpy()
        reconstructed_np = reconstructed.squeeze().permute(1, 2, 0).numpy()
        
        return diff_np, reconstructed_np

def normalize_diff(diff_map):
    """Fark haritasını 0-1 aralığına normalize et"""
    diff_min = diff_map.min()
    diff_max = diff_map.max()
    if diff_max > diff_min:
        normalized = (diff_map - diff_min) / (diff_max - diff_min)
    else:
        normalized = np.zeros_like(diff_map)
    return normalized

def create_ae_diff_visualization(image_path, save_path=None, model_path=None):
    """AE fark görselini oluştur"""
    # Görüntüyü yükle
    image_tensor, original = load_and_process_image(image_path)
    
    # Model yükle veya oluştur
    if model_path and Path(model_path).exists():
        print(f"Eğitilmiş model yükleniyor: {model_path}")
        model = SimpleAutoencoder()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        print("Yeni model oluşturuluyor (eğitilmemiş)")
        model = SimpleAutoencoder()
    
    # Fark hesapla
    diff_map, reconstructed = compute_ae_difference(image_tensor, model)
    
    # Normalize et
    normalized_diff = normalize_diff(diff_map)
    
    # Görselleştir
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Orijinal görüntü
    axes[0].imshow(original)
    axes[0].set_title('Orijinal Görüntü')
    axes[0].axis('off')
    
    # AE yeniden oluşturma
    axes[1].imshow(reconstructed)
    axes[1].set_title('AE Yeniden Oluşturma')
    axes[1].axis('off')
    
    # Normalize fark haritası
    im = axes[2].imshow(normalized_diff, cmap='hot')
    axes[2].set_title('AE Fark (Normalize)')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Görsel kaydedildi: {save_path}")
    
    plt.show()
    
    # İstatistikler
    print(f"Fark haritası - Min: {diff_map.min():.6f}, Max: {diff_map.max():.6f}")
    print(f"Normalize - Min: {normalized_diff.min():.6f}, Max: {normalized_diff.max():.6f}")

if __name__ == "__main__":
    # Kullanım örneği
    image_path = "C:\\Users\\cancor\\Desktop\\Repos\\project_mars\\docs\\docs\\figures\\curiosity_hills_small_objects.jpg"
    
    # Eğitilmiş model yolu
    model_path = "results/working_autoencoder_model.pth"
    
    create_ae_diff_visualization(image_path, "ae_diff_result.png", model_path)
