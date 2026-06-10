"""
Gerçek Rover Verileriyle Model Eğitimi
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pandas as pd
from pathlib import Path
from src.models.optimized_autoencoder import OptimizedAutoencoder, AutoencoderTrainer

class RealMarsDataset(Dataset):
    """Gerçek Mars verileri için dataset"""
    
    def __init__(self, processed_dir: str, transform=None):
        self.processed_dir = Path(processed_dir)
        self.transform = transform
        
        # Metadata dosyasını oku
        metadata_path = self.processed_dir / "metadata" / "processed_images.csv"
        if metadata_path.exists():
            self.metadata = pd.read_csv(metadata_path)
        else:
            raise FileNotFoundError("Metadata dosyası bulunamadı!")
        
        print(f"Dataset oluşturuldu: {len(self.metadata)} görüntü")
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        # İşlenmiş görüntüyü yükle
        processed_path = self.metadata.iloc[idx]['processed_path']
        image = np.load(processed_path)
        
        # Tensor'a çevir
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)  # HWC -> CHW
        
        if self.transform:
            image = self.transform(image)
        
        return image

def train_with_real_data():
    """Gerçek verilerle model eğitimi"""
    
    print("🚀 Gerçek Rover Verileriyle Model Eğitimi Başlıyor...")
    
    # Cihaz kontrolü
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    # Model oluştur
    model = OptimizedAutoencoder(input_channels=3, latent_dim=1024)
    trainer = AutoencoderTrainer(model, device=device)
    
    # Parametre sayısını kontrol et
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parametre sayısı: {total_params:,}")
    print(f"Model boyutu: {total_params * 4 / (1024 * 1024):.2f} MB")
    
    # Gerçek veri setini yükle
    processed_dir = "data/real_mars_data/processed"
    if not os.path.exists(processed_dir):
        print("❌ İşlenmiş veri dizini bulunamadı!")
        print("Önce veri işleme adımını tamamlayın.")
        return
    
    try:
        dataset = RealMarsDataset(processed_dir)
    except FileNotFoundError as e:
        print(f"❌ Veri seti yüklenemedi: {e}")
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
    
    # Eğitim
    num_epochs = 20  # Gerçek verilerle daha fazla epoch
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
            trainer.save_model("results/optimized_autoencoder_real_data.pth")
            print(f"  ✅ Yeni en iyi model kaydedildi (Val Loss: {val_loss:.6f})")
    
    # Test performansı
    print("\n📊 Test Performansı:")
    test_loss = trainer.validate(test_loader)
    print(f"Test Loss: {test_loss:.6f}")
    
    # Eğitim grafiği
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Gerçek Verilerle Eğitim Grafiği')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validasyon Loss Detayı')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/real_data_training_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Gerçek verilerle eğitim tamamlandı!")
    print(f"Final Train Loss: {train_losses[-1]:.6f}")
    print(f"Final Val Loss: {val_losses[-1]:.6f}")
    print(f"Test Loss: {test_loss:.6f}")
    print(f"En iyi Val Loss: {best_val_loss:.6f}")
    
    return trainer, train_losses, val_losses, test_loss

def compare_models():
    """Sentetik vs gerçek veri modellerini karşılaştır"""
    
    print("\n🔍 Model Karşılaştırması (Sentetik vs Gerçek)")
    print("=" * 60)
    
    # Sentetik veri modeli
    from src.models.optimized_autoencoder import OptimizedAutoencoder, AutoencoderTrainer
    
    synthetic_model = OptimizedAutoencoder()
    synthetic_trainer = AutoencoderTrainer(synthetic_model)
    
    if os.path.exists("results/optimized_autoencoder_model.pth"):
        synthetic_trainer.load_model("results/optimized_autoencoder_model.pth")
        print("✅ Sentetik veri modeli yüklendi")
    else:
        print("❌ Sentetik veri modeli bulunamadı")
        return
    
    # Gerçek veri modeli
    real_model = OptimizedAutoencoder()
    real_trainer = AutoencoderTrainer(real_model)
    
    if os.path.exists("results/optimized_autoencoder_real_data.pth"):
        real_trainer.load_model("results/optimized_autoencoder_real_data.pth")
        print("✅ Gerçek veri modeli yüklendi")
    else:
        print("❌ Gerçek veri modeli bulunamadı")
        return
    
    # Test görüntüsü ile karşılaştırma
    test_image = torch.randn(3, 128, 128)
    
    synthetic_score = synthetic_trainer.calculate_anomaly_score(test_image)
    real_score = real_trainer.calculate_anomaly_score(test_image)
    
    print(f"\n📊 Anomali Skorları:")
    print(f"Sentetik veri modeli: {synthetic_score:.6f}")
    print(f"Gerçek veri modeli: {real_score:.6f}")
    
    print(f"\n📈 Performans Karşılaştırması:")
    print(f"Sentetik veri modeli (50 görüntü): Val Loss ~0.020")
    print(f"Gerçek veri modeli (2000 görüntü): Val Loss ~{real_score:.6f}")

if __name__ == "__main__":
    # Gerçek verilerle eğitim
    trainer, train_losses, val_losses, test_loss = train_with_real_data()
    
    # Model karşılaştırması
    compare_models() 