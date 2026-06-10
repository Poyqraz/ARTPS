"""
Curiosity Verileriyle Model Eğitimi
"""

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from pathlib import Path
from PIL import Image
from src.models.optimized_autoencoder import OptimizedAutoencoder, AutoencoderTrainer
import matplotlib.pyplot as plt
from collections import Counter

class CuriosityDataset(Dataset):
    """Curiosity verileri için dataset"""
    
    def __init__(self, data_dir: str, target_size: tuple = (128, 128)):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        
        # Tüm görüntü dosyalarını topla
        self.image_files = []
        supported_formats = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        
        # Train ve valid klasörlerini tara
        for split in ['train', 'valid']:
            split_dir = self.data_dir / split
            if split_dir.exists():
                for category_dir in split_dir.iterdir():
                    if category_dir.is_dir():
                        for img_file in category_dir.iterdir():
                            if img_file.suffix.lower() in supported_formats:
                                self.image_files.append(str(img_file))
        
        # Ana dizindeki dosyaları da ekle
        for img_file in self.data_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in supported_formats:
                self.image_files.append(str(img_file))
        
        print(f"Dataset oluşturuldu: {len(self.image_files)} görüntü")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_files[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = image.resize(self.target_size, Image.LANCZOS)
            image = np.array(image, dtype=np.float32) / 255.0
            image = torch.from_numpy(image).float()
            image = image.permute(2, 0, 1)  # HWC -> CHW
            return image
        except Exception as e:
            print(f"Görüntü yükleme hatası ({img_path}): {e}")
            return torch.randn(3, self.target_size[0], self.target_size[1])

def train_curiosity_model():
    """Curiosity verileriyle model eğitimi"""
    
    print("🚀 Curiosity Verileriyle Model Eğitimi Başlıyor...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    # Veri dizini
    data_dir = "mars_images"
    
    if not os.path.exists(data_dir):
        print(f"❌ Veri dizini bulunamadı: {data_dir}")
        return
    
    # Dataset oluştur
    dataset = CuriosityDataset(data_dir)
    
    if len(dataset) < 100:
        print("❌ Yeterli veri yok!")
        return
    
    # Train/validation/test split
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
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
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parametre sayısı: {total_params:,}")
    
    # Eğitim
    num_epochs = 20
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
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_model("results/optimized_autoencoder_curiosity_data.pth")
            print(f"  ✅ Yeni en iyi model kaydedildi")
    
    # Test performansı
    test_loss = trainer.validate(test_loader)
    print(f"\nTest Loss: {test_loss:.6f}")
    
    # Eğitim grafiği
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Curiosity Verileriyle Eğitim')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validasyon Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/curiosity_training_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Curiosity eğitimi tamamlandı!")
    print(f"Final Train Loss: {train_losses[-1]:.6f}")
    print(f"Final Val Loss: {val_losses[-1]:.6f}")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"En iyi Val Loss: {best_val_loss:.6f}")
    
    return trainer, train_losses, val_losses, test_loss

if __name__ == "__main__":
    train_curiosity_model() 