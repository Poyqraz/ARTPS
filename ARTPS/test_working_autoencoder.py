#!/usr/bin/env python3
"""
ARTPS - Çalışan Autoencoder Model Test Scripti

Bu script, Working Convolutional Autoencoder modelini test etmek için kullanılır.
Örnek veri oluşturur, modeli eğitir ve anomali tespitini test eder.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Proje modüllerini import et
sys.path.append('src')
from models.working_autoencoder import WorkingAutoencoder, AutoencoderTrainer, MarsRockDataset
from utils.data_utils import create_sample_data, load_image, extract_features, calculate_similarity


def test_model_creation():
    """Model oluşturma testi"""
    print("=== Model Oluşturma Testi ===")
    
    # Model parametreleri
    input_channels = 3
    latent_dim = 16384
    
    # Model oluştur
    model = WorkingAutoencoder(input_channels=input_channels, latent_dim=latent_dim)
    
    # Parametre sayısını kontrol et
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parametre sayısı: {total_params:,}")
    
    # Test görüntüsü oluştur
    test_image = torch.randn(1, 3, 128, 128)
    
    # Forward pass testi
    with torch.no_grad():
        reconstructed, latent = model(test_image)
    
    print(f"Giriş boyutu: {test_image.shape}")
    print(f"Çıkış boyutu: {reconstructed.shape}")
    print(f"Latent boyutu: {latent.shape}")
    
    # Boyut kontrolü
    assert reconstructed.shape == test_image.shape, "Çıkış boyutu giriş boyutu ile eşleşmiyor!"
    assert latent.shape[1] == latent_dim, "Latent boyutu yanlış!"
    
    print("✅ Model oluşturma testi başarılı!\n")


def test_data_generation():
    """Örnek veri oluşturma testi"""
    print("=== Veri Oluşturma Testi ===")
    
    # Veri dizini oluştur
    data_dir = "data/mars_rocks"
    os.makedirs(data_dir, exist_ok=True)
    
    # Örnek veri oluştur
    num_samples = 50  # Test için az sayıda örnek
    create_sample_data(data_dir, num_samples, image_size=(128, 128))
    
    # Veri setini yükle
    dataset = MarsRockDataset(data_dir, target_size=(128, 128))
    print(f"Veri seti boyutu: {len(dataset)}")
    
    # İlk görüntüyü test et
    sample_image = dataset[0]
    print(f"Örnek görüntü boyutu: {sample_image.shape}")
    print(f"Örnek görüntü değer aralığı: [{sample_image.min():.3f}, {sample_image.max():.3f}]")
    
    print("✅ Veri oluşturma testi başarılı!\n")


def test_training():
    """Model eğitimi testi"""
    print("=== Model Eğitimi Testi ===")
    
    # Cihaz kontrolü
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    # Model oluştur
    model = WorkingAutoencoder(input_channels=3, latent_dim=16384)
    trainer = AutoencoderTrainer(model, device=device)
    
    # Veri yükle
    data_dir = "data/mars_rocks"
    if not os.path.exists(data_dir):
        print("❌ Veri dizini bulunamadı! Önce veri oluşturun.")
        return
    
    dataset = MarsRockDataset(data_dir, target_size=(128, 128))
    
    if len(dataset) < 10:
        print("❌ Yeterli veri yok! En az 10 görüntü gerekli.")
        return
    
    # Train/validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # DataLoader'lar
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Küçük batch size
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    print(f"Eğitim örnekleri: {len(train_dataset)}")
    print(f"Validasyon örnekleri: {len(val_dataset)}")
    
    # Kısa eğitim (test amaçlı)
    num_epochs = 5
    print(f"Eğitim başlıyor ({num_epochs} epoch)...")
    
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train: {train_loss:.6f}, Val: {val_loss:.6f}")
    
    # Modeli kaydet
    os.makedirs("results", exist_ok=True)
    trainer.save_model("results/working_autoencoder_model.pth")
    print("✅ Model eğitimi testi başarılı!\n")
    
    return trainer


def test_anomaly_detection(trainer):
    """Anomali tespiti testi"""
    print("=== Anomali Tespiti Testi ===")
    
    # Test görüntüleri oluştur
    data_dir = "data/mars_rocks"
    dataset = MarsRockDataset(data_dir, target_size=(128, 128))
    
    # Normal görüntüler için anomali skorları
    normal_scores = []
    for i in range(min(10, len(dataset))):
        image = dataset[i]
        score = trainer.calculate_anomaly_score(image.unsqueeze(0))
        normal_scores.append(score)
    
    # Anormal görüntü oluştur (farklı renk ve doku)
    abnormal_image = torch.randn(3, 128, 128)  # Rastgele görüntü
    abnormal_score = trainer.calculate_anomaly_score(abnormal_image.unsqueeze(0))
    
    print(f"Normal görüntüler için ortalama anomali skoru: {np.mean(normal_scores):.6f}")
    print(f"Anormal görüntü için anomali skoru: {abnormal_score:.6f}")
    
    # Anomali tespiti kontrolü
    if abnormal_score > np.mean(normal_scores):
        print("✅ Anomali tespiti çalışıyor - anormal görüntü daha yüksek skor aldı!")
    else:
        print("⚠️ Anomali tespiti beklenen sonucu vermedi.")
    
    print("✅ Anomali tespiti testi tamamlandı!\n")


def test_feature_extraction():
    """Özellik çıkarımı testi"""
    print("=== Özellik Çıkarımı Testi ===")
    
    # Test görüntüsü oluştur
    test_image = torch.randn(3, 128, 128)
    
    # Özellik çıkar
    features = extract_features(test_image)
    
    print("Çıkarılan özellikler:")
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value:.4f}")
    
    # İki görüntü arası benzerlik hesapla
    test_image2 = torch.randn(3, 128, 128)
    features2 = extract_features(test_image2)
    
    similarity = calculate_similarity(features, features2)
    print(f"İki test görüntüsü arası benzerlik: {similarity:.4f}")
    
    print("✅ Özellik çıkarımı testi başarılı!\n")


def visualize_results(trainer):
    """Sonuçları görselleştir"""
    print("=== Görselleştirme ===")
    
    # Test görüntüsü yükle
    data_dir = "data/mars_rocks"
    dataset = MarsRockDataset(data_dir, target_size=(128, 128))
    
    if len(dataset) > 0:
        sample_image = dataset[0]
        
        # Yeniden oluşturma görselleştirmesi
        from models.working_autoencoder import visualize_reconstruction
        visualize_reconstruction(trainer.model, sample_image, "results/working_reconstruction.png")
        
        print("✅ Görselleştirme tamamlandı! Sonuçlar 'results' klasöründe.")


def main():
    """Ana test fonksiyonu"""
    print("🚀 ARTPS Çalışan Autoencoder Model Testi Başlıyor...\n")
    
    try:
        # Testleri sırayla çalıştır
        test_model_creation()
        test_data_generation()
        trainer = test_training()
        
        if trainer:
            test_anomaly_detection(trainer)
            visualize_results(trainer)
        
        test_feature_extraction()
        
        print("🎉 Tüm testler başarıyla tamamlandı!")
        print("\n📁 Oluşturulan dosyalar:")
        print("  - data/mars_rocks/: Örnek Mars kaya görüntüleri")
        print("  - results/working_autoencoder_model.pth: Eğitilmiş model")
        print("  - results/working_reconstruction.png: Yeniden oluşturma örneği")
        
    except Exception as e:
        print(f"❌ Test sırasında hata oluştu: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 