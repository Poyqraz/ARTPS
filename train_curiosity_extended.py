"""
Curiosity Modelini Uzun Süreli Eğitim (50+ Epoch)
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
import time

class CuriosityDataset(Dataset):
    def __init__(self, data_dir: str, target_size: tuple = (128, 128)):
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.image_files = []
        supported_formats = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        
        # Train ve valid klasörlerinden görüntüleri topla
        for split in ['train', 'valid']:
            split_dir = self.data_dir / split
            if split_dir.exists():
                for category_dir in split_dir.iterdir():
                    if category_dir.is_dir():
                        for img_file in category_dir.iterdir():
                            if img_file.suffix.lower() in supported_formats:
                                self.image_files.append(str(img_file))
        
        # Ana dizinden de görüntüleri topla
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

def train_curiosity_extended():
    print("🚀 Curiosity Verileriyle Uzun Süreli Model Eğitimi Başlıyor...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    # Veri dizini kontrolü
    data_dir = "mars_images"
    if not os.path.exists(data_dir):
        print(f"❌ Veri dizini bulunamadı: {data_dir}")
        return
    
    # Dataset oluştur
    dataset = CuriosityDataset(data_dir)
    if len(dataset) < 100:
        print("❌ Yeterli veri yok!")
        return
    
    print(f"📊 Toplam görüntü sayısı: {len(dataset)}")
    
    # Veri seti bölünmesi
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # DataLoader'ları oluştur
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    print(f"Veri seti bölünmesi:")
    print(f"  Eğitim: {len(train_dataset)} görüntü")
    print(f"  Validasyon: {len(val_dataset)} görüntü")
    print(f"  Test: {len(test_dataset)} görüntü")
    
    # Model oluştur
    model = OptimizedAutoencoder(input_channels=3, latent_dim=1024)
    trainer = AutoencoderTrainer(model, device=device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parametre sayısı: {total_params:,}")
    
    # Uzun süreli eğitim parametreleri
    num_epochs = 100  # 100 epoch
    print(f"Eğitim başlıyor ({num_epochs} epoch)...")
    print(f"Tahmini süre: {num_epochs * 2:.0f} dakika")
    
    # Eğitim takibi
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10  # Early stopping için
    patience_counter = 0
    
    # Eğitim başlangıç zamanı
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Eğitim
        train_loss = trainer.train_epoch(train_loader)
        
        # Validasyon
        val_loss = trainer.validate(val_loader)
        
        # Sonuçları kaydet
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Epoch süresi
        epoch_time = time.time() - epoch_start
        
        # İlerleme raporu
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] - "
              f"Train: {train_loss:.6f}, Val: {val_loss:.6f}, "
              f"Süre: {epoch_time:.1f}s")
        
        # En iyi model kaydetme
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_model("results/optimized_autoencoder_curiosity_extended.pth")
            print(f"  ✅ Yeni en iyi model kaydedildi (Val Loss: {val_loss:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping kontrolü
        if patience_counter >= patience:
            print(f"  ⚠️ Early stopping: {patience} epoch boyunca iyileşme yok")
            break
        
        # Her 10 epoch'ta bir ara rapor
        if (epoch + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            avg_epoch_time = elapsed_time / (epoch + 1)
            remaining_epochs = num_epochs - (epoch + 1)
            estimated_remaining = remaining_epochs * avg_epoch_time
            
            print(f"\n📊 Ara Rapor (Epoch {epoch+1}):")
            print(f"  Geçen süre: {elapsed_time/60:.1f} dakika")
            print(f"  Ortalama epoch süresi: {avg_epoch_time:.1f} saniye")
            print(f"  Tahmini kalan süre: {estimated_remaining/60:.1f} dakika")
            print(f"  En iyi val loss: {best_val_loss:.6f}")
            print(f"  Patience counter: {patience_counter}/{patience}")
    
    # Test sonuçları
    print(f"\n🧪 Test Sonuçları:")
    test_loss = trainer.validate(test_loader)
    print(f"Test Loss: {test_loss:.6f}")
    
    # Toplam eğitim süresi
    total_time = time.time() - start_time
    print(f"\n⏱️ Toplam eğitim süresi: {total_time/60:.1f} dakika")
    
    # Eğitim eğrilerini çiz
    plt.figure(figsize=(15, 5))
    
    # Train/Val Loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', color='blue', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', color='red', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Eğitim ve Validasyon Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Sadece Validasyon Loss (detaylı)
    plt.subplot(1, 3, 2)
    plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validasyon Loss (Detaylı)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Son 20 epoch
    plt.subplot(1, 3, 3)
    if len(val_losses) >= 20:
        plt.plot(val_losses[-20:], label='Validation Loss', color='red', linewidth=2)
        plt.xlabel('Epoch (Son 20)')
        plt.ylabel('Loss')
        plt.title('Son 20 Epoch Validasyon Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/curiosity_extended_training_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Final rapor
    print(f"\n📋 UZUN SÜRELİ EĞİTİM RAPORU")
    print("=" * 50)
    print(f"Toplam epoch: {len(train_losses)}")
    print(f"Final Train Loss: {train_losses[-1]:.6f}")
    print(f"Final Val Loss: {val_losses[-1]:.6f}")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"En iyi Val Loss: {best_val_loss:.6f}")
    print(f"Toplam eğitim süresi: {total_time/60:.1f} dakika")
    print(f"Ortalama epoch süresi: {total_time/len(train_losses):.1f} saniye")
    
    # Model dosyası bilgisi
    model_path = "results/optimized_autoencoder_curiosity_extended.pth"
    if os.path.exists(model_path):
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Model dosya boyutu: {model_size:.2f} MB")
    
    print(f"\n✅ Uzun süreli eğitim tamamlandı!")
    print(f"🎯 Model artık daha iyi performans gösterecek")
    print(f"📁 Model kaydedildi: {model_path}")
    
    return trainer, train_losses, val_losses, test_loss

if __name__ == "__main__":
    train_curiosity_extended() 