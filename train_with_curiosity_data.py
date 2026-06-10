"""
Curiosity Verileriyle Model Eğitimi
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2
from src.models.optimized_autoencoder import OptimizedAutoencoder, AutoencoderTrainer
import matplotlib.pyplot as plt
from collections import Counter

class CuriosityDataset(Dataset):
    """Curiosity verileri için dataset"""
    
    def __init__(self, data_dir: str, transform=None, target_size: tuple = (128, 128)):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_size = target_size
        
        # Tüm görüntü dosyalarını topla
        self.image_files = []
        self.categories = []
        
        # Desteklenen formatlar
        supported_formats = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        
        # Train ve valid klasörlerini tara
        for split in ['train', 'valid']:
            split_dir = self.data_dir / split
            if split_dir.exists():
                for category_dir in split_dir.iterdir():
                    if category_dir.is_dir():
                        category = category_dir.name
                        for img_file in category_dir.iterdir():
                            if img_file.suffix.lower() in supported_formats:
                                self.image_files.append(str(img_file))
                                self.categories.append(category)
        
        # Ana dizindeki dosyaları da ekle
        for img_file in self.data_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in supported_formats:
                self.image_files.append(str(img_file))
                self.categories.append('main')
        
        print(f"Dataset oluşturuldu: {len(self.image_files)} görüntü")
        
        # Kategori dağılımını göster
        category_counts = Counter(self.categories)
        print("Kategori dağılımı:")
        for category, count in category_counts.items():
            print(f"  {category}: {count} görüntü")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        # Görüntüyü yükle
        img_path = self.image_files[idx]
        
        try:
            # PIL ile yükle
            image = Image.open(img_path).convert('RGB')
            
            # Boyutlandır
            image = image.resize(self.target_size, Image.LANCZOS)
            
            # NumPy array'e çevir
            image = np.array(image, dtype=np.float32) / 255.0
            
            # Tensor'a çevir
            image = torch.from_numpy(image).float()
            image = image.permute(2, 0, 1)  # HWC -> CHW
            
            if self.transform:
                image = self.transform(image)
            
            return image
            
        except Exception as e:
            print(f"Görüntü yükleme hatası ({img_path}): {e}")
            # Hata durumunda rastgele görüntü döndür
            return torch.randn(3, self.target_size[0], self.target_size[1])

def train_with_curiosity_data():
    """Curiosity verileriyle model eğitimi"""
    
    print("🚀 Curiosity Verileriyle Model Eğitimi Başlıyor...")
    
    # Cihaz kontrolü
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    # Veri dizini
    data_dir = "mars_images"
    
    if not os.path.exists(data_dir):
        print(f"❌ Veri dizini bulunamadı: {data_dir}")
        return
    
    # Dataset oluştur
    try:
        dataset = CuriosityDataset(data_dir)
    except Exception as e:
        print(f"❌ Dataset oluşturma hatası: {e}")
        return
    
    if len(dataset) < 100:
        print("❌ Yeterli veri yok! En az 100 görüntü gerekli.")
        return
    
    # Train/validation/test split
    total_size = len(dataset)
    train_size = int(0.7 * total_size)  # %70 eğitim
    val_size = int(0.15 * total_size)   # %15 validasyon
    test_size = total_size - train_size - val_size  # %15 test
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # DataLoader'lar
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    print(f"Veri seti bölünmesi:")
    print(f"  Eğitim: {len(train_dataset)} görüntü")
    print(f"  Validasyon: {len(val_dataset)} görüntü")
    print(f"  Test: {len(test_dataset)} görüntü")
    
    # Model oluştur
    model = OptimizedAutoencoder(input_channels=3, latent_dim=1024)
    trainer = AutoencoderTrainer(model, device=device)
    
    # Parametre sayısını kontrol et
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parametre sayısı: {total_params:,}")
    print(f"Model boyutu: {total_params * 4 / (1024 * 1024):.2f} MB")
    
    # Eğitim
    num_epochs = 30  # Daha fazla veri ile daha fazla epoch
    print(f"Eğitim başlıyor ({num_epochs} epoch)...")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
        
        # En iyi modeli kaydet
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_model("results/optimized_autoencoder_curiosity_data.pth")
            print(f"  ✅ Yeni en iyi model kaydedildi (Val Loss: {val_loss:.6f})")
        
        # Her 5 epoch'ta bir ara sonuç göster
        if (epoch + 1) % 5 == 0:
            print(f"  📊 Ara sonuç - Epoch {epoch+1}: Train={train_loss:.6f}, Val={val_loss:.6f}")
    
    # Test performansı
    print("\n📊 Test Performansı:")
    test_loss = trainer.validate(test_loader)
    print(f"Test Loss: {test_loss:.6f}")
    
    # Eğitim grafiği
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Curiosity Verileriyle Eğitim Grafiği')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validasyon Loss Detayı')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    # Son 10 epoch'u göster
    if len(train_losses) >= 10:
        plt.plot(train_losses[-10:], label='Train Loss (Son 10)', color='blue')
        plt.plot(val_losses[-10:], label='Val Loss (Son 10)', color='red')
        plt.xlabel('Epoch (Son 10)')
        plt.ylabel('Loss')
        plt.title('Son 10 Epoch Detayı')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/curiosity_training_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Curiosity verileriyle eğitim tamamlandı!")
    print(f"Final Train Loss: {train_losses[-1]:.6f}")
    print(f"Final Val Loss: {val_losses[-1]:.6f}")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"En iyi Val Loss: {best_val_loss:.6f}")
    
    return trainer, train_losses, val_losses, test_loss

def download_additional_curiosity_data():
    """API'den ek Curiosity verileri indir"""
    
    print("\n🔄 API'den Ek Curiosity Verileri İndiriliyor...")
    
    import requests
    import json
    from urllib.parse import urljoin
    
    base_url = "https://mars.nasa.gov/api/v1/raw_image_items/"
    
    # Curiosity verilerini indir
    params = {
        'mission': 'msl',  # Curiosity
        'limit': 100,      # 100 görüntü
        'page': 1
    }
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    try:
        response = session.get(base_url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            items = data.get('items', [])
            
            print(f"API'den {len(items)} Curiosity görüntüsü bulundu")
            
            # İlk 5 görüntüyü göster
            for i, item in enumerate(items[:5]):
                title = item.get('title', 'Unknown')
                url = item.get('url', '')
                print(f"  {i+1}. {title}")
                print(f"     URL: {url}")
            
            return items
        else:
            print(f"API hatası: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"API isteği hatası: {e}")
        return []

if __name__ == "__main__":
    print("🚀 Curiosity Veri Eğitimi Başlıyor...")
    
    # 1. Mevcut verilerle eğitim
    trainer, train_losses, val_losses, test_loss = train_with_curiosity_data()
    
    # 2. Ek verileri indir
    additional_data = download_additional_curiosity_data()
    
    if additional_data:
        print(f"\n📥 {len(additional_data)} ek Curiosity görüntüsü bulundu")
        print("Bu verileri de kullanabilirsiniz.")
    
    print("\n✅ Curiosity eğitimi tamamlandı!")
    print("Sonraki adım: Perseverance verilerini bulmaya devam edin.") 