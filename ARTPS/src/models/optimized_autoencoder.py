"""
ARTPS - Optimize Edilmiş Autoencoder Modeli
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt


class OptimizedAutoencoder(nn.Module):
    """Optimize edilmiş Convolutional Autoencoder modeli"""
    
    def __init__(self, input_channels: int = 3, latent_dim: int = 1024):
        super(OptimizedAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder: 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8
        self.encoder = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Latent space projection (küçültülmüş)
        self.latent_projection = nn.Sequential(
            nn.Flatten(),  # AdaptiveAvgPool2d yerine Flatten
            nn.Linear(128 * 8 * 8, latent_dim),  # 8,192 -> 1,024
            nn.ReLU(inplace=True)
        )
        
        # Decoder: 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
        self.decoder = nn.Sequential(
            # Latent space'den başlayarak
            nn.Linear(latent_dim, 128 * 8 * 8),  # 1,024 -> 8,192
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (128, 8, 8)),  # Reshape
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(16, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        latent = self.latent_projection(encoded)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        return self.latent_projection(encoded)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)


class MarsRockDataset(Dataset):
    """Mars kaya görüntüleri için dataset sınıfı"""
    
    def __init__(self, data_dir: str, transform=None, target_size: Tuple[int, int] = (128, 128)):
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size
        
        # Desteklenen dosya formatları
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Dosya listesini oluştur
        self.image_files = []
        for file in os.listdir(data_dir):
            if any(file.lower().endswith(fmt) for fmt in self.supported_formats):
                self.image_files.append(os.path.join(data_dir, file))
        
        print(f"Dataset oluşturuldu: {len(self.image_files)} görüntü bulundu")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Görüntüyü yükle ve ön işle"""
        
        # Görüntüyü yükle
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Boyutlandır
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        
        # Tensor'a çevir
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).float() / 255.0
            image = image.permute(2, 0, 1)  # HWC -> CHW
        
        return image


class AutoencoderTrainer:
    """Autoencoder eğitimi için trainer sınıfı"""
    
    def __init__(self, model: OptimizedAutoencoder, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer ve loss function
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        print(f"Trainer oluşturuldu. Cihaz: {device}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Tek epoch eğitim"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, images in enumerate(train_loader):
            images = images.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstructed, _ = self.model(images)
            loss = self.criterion(reconstructed, images)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validasyon"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images in val_loader:
                images = images.to(self.device)
                reconstructed, _ = self.model(images)
                loss = self.criterion(reconstructed, images)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def calculate_anomaly_score(self, image: torch.Tensor) -> float:
        """Anomali skoru hesapla"""
        self.model.eval()
        
        with torch.no_grad():
            image = image.unsqueeze(0).to(self.device)  # Batch dimension ekle
            reconstructed, _ = self.model(image)
            
            # MSE loss hesapla
            anomaly_score = self.criterion(reconstructed, image).item()
        
        return anomaly_score
    
    def save_model(self, filepath: str):
        """Modeli kaydet"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"Model kaydedildi: {filepath}")
    
    def load_model(self, filepath: str):
        """Modeli yükle"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model yüklendi: {filepath}")


def visualize_reconstruction(model: OptimizedAutoencoder, 
                           image: torch.Tensor, 
                           save_path: Optional[str] = None):
    """Yeniden oluşturma sonucunu görselleştir"""
    
    model.eval()
    with torch.no_grad():
        # Yeniden oluştur
        reconstructed, _ = model(image.unsqueeze(0))
        reconstructed = reconstructed.squeeze(0)
    
    # Görselleştirme
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Orijinal görüntü
    original_img = image.permute(1, 2, 0).cpu().numpy()
    axes[0].imshow(original_img)
    axes[0].set_title('Orijinal Görüntü')
    axes[0].axis('off')
    
    # Yeniden oluşturulan görüntü
    recon_img = reconstructed.permute(1, 2, 0).cpu().numpy()
    axes[1].imshow(recon_img)
    axes[1].set_title('Yeniden Oluşturulan Görüntü')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Görselleştirme kaydedildi: {save_path}")
    
    plt.show()


def main():
    """Ana fonksiyon - model testi"""
    print("🚀 Optimize Edilmiş Autoencoder Test Başlıyor...")
    
    # Model oluştur
    model = OptimizedAutoencoder(input_channels=3, latent_dim=1024)
    
    # Parametre sayısını kontrol et
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model oluşturuldu. Parametre sayısı: {total_params:,}")
    print(f"Model boyutu: {total_params * 4 / (1024 * 1024):.2f} MB")
    
    # Test görüntüsü oluştur
    test_image = torch.randn(1, 3, 128, 128)
    
    # Forward pass testi
    with torch.no_grad():
        reconstructed, latent = model(test_image)
    
    print(f"Giriş boyutu: {test_image.shape}")
    print(f"Çıkış boyutu: {reconstructed.shape}")
    print(f"Latent boyutu: {latent.shape}")
    
    print("✅ Optimize edilmiş model testi başarılı!")


if __name__ == "__main__":
    main() 