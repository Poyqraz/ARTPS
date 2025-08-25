"""
ARTPS - Basit Convolutional Autoencoder Modeli

Bu modül, Mars kaya görüntülerini analiz etmek için kullanılan
basit ve etkili bir Convolutional Autoencoder modelini içerir.
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


class SimpleAutoencoder(nn.Module):
    """
    Basit Convolutional Autoencoder modeli
    
    Bu model, Mars kaya görüntülerini sıkıştırıp yeniden oluşturur.
    Daha basit ve stabil bir yapı kullanır.
    """
    
    def __init__(self, input_channels: int = 3, latent_dim: int = 16384):
        """
        Args:
            input_channels: Giriş görüntüsünün kanal sayısı (RGB için 3)
            latent_dim: Latent space boyutu (256*8*8 = 16384 olmalı)
        """
        super(SimpleAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder: 128x128 -> 64x64 -> 32x32 -> 16x16
        self.encoder = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Latent space projection
        self.latent_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, latent_dim),
            nn.ReLU(inplace=True)
        )
        
        # Decoder: 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
        self.decoder = nn.Sequential(
            # Latent space'den başlayarak - latent_dim'i 256*8*8'e çıkar
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.ReLU(inplace=True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Çıkışı [0,1] aralığına normalize et
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Giriş görüntüsü tensor'ı [batch_size, channels, height, width]
            
        Returns:
            Tuple[reconstructed, latent]: Yeniden oluşturulan görüntü ve latent representation
        """
        # Encoder
        encoded = self.encoder(x)
        
        # Latent space projection
        latent = self.latent_projection(encoded)
        
        # Decoder
        # Latent'i uygun boyuta reshape et
        batch_size = latent.size(0)
        latent_reshaped = latent.view(batch_size, 256, 8, 8)
        
        # Decode
        reconstructed = self.decoder(latent_reshaped)
        
        return reconstructed, latent
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Sadece encoding yapar"""
        encoded = self.encoder(x)
        latent = self.latent_projection(encoded)
        return latent
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Sadece decoding yapar"""
        batch_size = latent.size(0)
        latent_reshaped = latent.view(batch_size, 256, 8, 8)
        reconstructed = self.decoder(latent_reshaped)
        return reconstructed


class MarsRockDataset(Dataset):
    """
    Mars kaya görüntüleri için özel dataset sınıfı
    """
    
    def __init__(self, data_dir: str, transform=None, target_size: Tuple[int, int] = (128, 128)):
        """
        Args:
            data_dir: Görüntülerin bulunduğu dizin
            transform: Görüntü dönüşümleri
            target_size: Hedef görüntü boyutu (height, width)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size
        
        # Desteklenen görüntü formatları
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        # Görüntü dosyalarını bul
        self.image_files = []
        for file in os.listdir(data_dir):
            if any(file.lower().endswith(ext) for ext in self.image_extensions):
                self.image_files.append(os.path.join(data_dir, file))
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Görüntüyü yükle ve tensor'a dönüştür
        
        Args:
            idx: Görüntü indeksi
            
        Returns:
            torch.Tensor: Normalize edilmiş görüntü tensor'ı [channels, height, width]
        """
        image_path = self.image_files[idx]
        
        # Görüntüyü yükle
        image = Image.open(image_path).convert('RGB')
        
        # Boyutu ayarla
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        
        # Tensor'a dönüştür
        if self.transform:
            image = self.transform(image)
        else:
            # Varsayılan dönüşüm: PIL -> Tensor ve normalize
            image = torch.from_numpy(np.array(image)).float()
            image = image.permute(2, 0, 1)  # HWC -> CHW
            image = image / 255.0  # [0, 255] -> [0, 1]
        
        return image


class AutoencoderTrainer:
    """
    Autoencoder modelini eğitmek için trainer sınıfı
    """
    
    def __init__(self, model: SimpleAutoencoder, device: str = 'cuda'):
        """
        Args:
            model: Eğitilecek autoencoder modeli
            device: Eğitim cihazı ('cuda' veya 'cpu')
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Eğitim geçmişi
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Tek epoch eğitim
        
        Args:
            train_loader: Eğitim veri yükleyicisi
            
        Returns:
            float: Ortalama eğitim kaybı
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, images in enumerate(train_loader):
            images = images.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstructed, _ = self.model(images)
            
            # Loss hesapla
            loss = self.criterion(reconstructed, images)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validasyon
        
        Args:
            val_loader: Validasyon veri yükleyicisi
            
        Returns:
            float: Ortalama validasyon kaybı
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for images in val_loader:
                images = images.to(self.device)
                reconstructed, _ = self.model(images)
                loss = self.criterion(reconstructed, images)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def calculate_anomaly_score(self, image: torch.Tensor) -> float:
        """
        Bir görüntü için anomali puanı hesapla
        
        Args:
            image: Giriş görüntüsü tensor'ı [1, channels, height, width]
            
        Returns:
            float: Anomali puanı (reconstruction error)
        """
        self.model.eval()
        
        with torch.no_grad():
            image = image.to(self.device)
            reconstructed, _ = self.model(image)
            
            # MSE loss hesapla
            mse_loss = self.criterion(reconstructed, image)
            
            # Anomali puanı olarak MSE'yi döndür
            return mse_loss.item()
    
    def save_model(self, filepath: str):
        """Modeli kaydet"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filepath)
    
    def load_model(self, filepath: str):
        """Modeli yükle"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']


def visualize_reconstruction(model: SimpleAutoencoder, 
                           image: torch.Tensor, 
                           save_path: Optional[str] = None):
    """
    Orijinal ve yeniden oluşturulan görüntüyü görselleştir
    
    Args:
        model: Eğitilmiş autoencoder modeli
        image: Giriş görüntüsü
        save_path: Kaydetme yolu (opsiyonel)
    """
    model.eval()
    
    with torch.no_grad():
        reconstructed, _ = model(image.unsqueeze(0))
        reconstructed = reconstructed.squeeze(0)
    
    # Görüntüleri numpy array'e dönüştür
    original = image.permute(1, 2, 0).cpu().numpy()
    reconstructed = reconstructed.permute(1, 2, 0).cpu().numpy()
    
    # Görselleştir
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(original)
    axes[0].set_title('Orijinal Görüntü')
    axes[0].axis('off')
    
    axes[1].imshow(reconstructed)
    axes[1].set_title('Yeniden Oluşturulan Görüntü')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def main():
    """
    Ana fonksiyon - model eğitimi ve test
    """
    # Cihaz kontrolü
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    # Model parametreleri
    input_channels = 3
    latent_dim = 64
    image_size = (128, 128)
    batch_size = 16
    num_epochs = 50
    
    # Model oluştur
    model = SimpleAutoencoder(input_channels=input_channels, latent_dim=latent_dim)
    print(f"Model oluşturuldu. Parametre sayısı: {sum(p.numel() for p in model.parameters()):,}")
    
    # Trainer oluştur
    trainer = AutoencoderTrainer(model, device=device)
    
    # Veri yolu (örnek - gerçek veri yolu ile değiştirilmeli)
    data_dir = "data/mars_rocks"
    
    # Veri seti oluştur (eğer veri varsa)
    if os.path.exists(data_dir):
        dataset = MarsRockDataset(data_dir, target_size=image_size)
        print(f"Veri seti yüklendi. Görüntü sayısı: {len(dataset)}")
        
        # Train/validation split
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # DataLoader'lar
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Eğitim döngüsü
        print("Eğitim başlıyor...")
        for epoch in range(num_epochs):
            train_loss = trainer.train_epoch(train_loader)
            val_loss = trainer.validate(val_loader)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Modeli kaydet
        trainer.save_model("results/simple_autoencoder_model.pth")
        print("Model kaydedildi: results/simple_autoencoder_model.pth")
        
        # Örnek görselleştirme
        if len(dataset) > 0:
            sample_image = dataset[0]
            visualize_reconstruction(model, sample_image, "results/simple_reconstruction_example.png")
    
    else:
        print(f"Veri dizini bulunamadı: {data_dir}")
        print("Model yapısı hazır. Veri eklendikten sonra eğitim yapılabilir.")


if __name__ == "__main__":
    main() 