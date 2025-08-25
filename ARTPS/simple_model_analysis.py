"""
Basit Model Analizi
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path
from src.models.optimized_autoencoder import OptimizedAutoencoder

def analyze_training_data():
    """Eğitim verilerini analiz et"""
    
    print("🔍 Eğitim Verilerini Analiz Ediliyor...")
    
    data_dir = Path("mars_images")
    
    # Kategori dağılımını analiz et
    categories = {}
    total_images = 0
    
    for split in ['train', 'valid']:
        split_dir = data_dir / split
        if split_dir.exists():
            for category_dir in split_dir.iterdir():
                if category_dir.is_dir():
                    category = category_dir.name
                    image_count = len(list(category_dir.glob("*.jpg"))) + len(list(category_dir.glob("*.png")))
                    categories[category] = categories.get(category, 0) + image_count
                    total_images += image_count
    
    print(f"\n📊 Eğitim Veri Analizi:")
    print(f"Toplam görüntü: {total_images}")
    print(f"Kategori sayısı: {len(categories)}")
    
    print("\nKategori dağılımı:")
    for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_images) * 100
        print(f"  {category}: {count} görüntü ({percentage:.1f}%)")
    
    return categories, total_images

def load_trained_model():
    """Eğitilen modeli yükle"""
    
    print("\n🤖 Eğitilen Model Yükleniyor...")
    
    model_path = "results/optimized_autoencoder_curiosity_data.pth"
    
    if not os.path.exists(model_path):
        print(f"❌ Model dosyası bulunamadı: {model_path}")
        return None
    
    # Model oluştur
    model = OptimizedAutoencoder(input_channels=3, latent_dim=1024)
    
    # Eğitilen ağırlıkları yükle
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model yüklendi: {model_path}")
    print(f"Model parametre sayısı: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def test_simple_reconstruction():
    """Basit yeniden oluşturma testi"""
    
    print("\n🔄 Basit Yeniden Oluşturma Testi...")
    
    model = load_trained_model()
    if model is None:
        return
    
    # Test görüntüsü seç
    data_dir = Path("mars_images/valid")
    test_image_path = None
    
    if data_dir.exists():
        for category_dir in data_dir.iterdir():
            if category_dir.is_dir():
                image_files = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
                if image_files:
                    test_image_path = str(image_files[0])
                    break
    
    if not test_image_path:
        print("❌ Test görüntüsü bulunamadı")
        return
    
    try:
        # Görüntüyü yükle
        image = Image.open(test_image_path).convert('RGB')
        image = image.resize((128, 128), Image.LANCZOS)
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Tensor'a çevir
        input_tensor = torch.from_numpy(image_array).float()
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
        
        # Model tahmini
        with torch.no_grad():
            reconstructed, latent = model(input_tensor)
        
        # Sonuçları görselleştir
        reconstructed = reconstructed.squeeze(0).permute(1, 2, 0).numpy()
        reconstructed = np.clip(reconstructed, 0, 1)
        
        # MSE hesapla
        mse = np.mean((image_array - reconstructed) ** 2)
        
        print(f"Test görüntüsü: {Path(test_image_path).parent.name}")
        print(f"MSE: {mse:.6f}")
        
        # Görselleştir
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image_array)
        axes[0].set_title("Orijinal Görüntü")
        axes[0].axis('off')
        
        axes[1].imshow(reconstructed)
        axes[1].set_title(f"Yeniden Oluşturulan\nMSE: {mse:.6f}")
        axes[1].axis('off')
        
        # Fark görüntüsü
        diff = np.abs(image_array - reconstructed)
        axes[2].imshow(diff, cmap='hot')
        axes[2].set_title("Fark Görüntüsü")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/simple_reconstruction_test.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Yeniden oluşturma testi tamamlandı!")
        
    except Exception as e:
        print(f"❌ Hata: {e}")

def test_anomaly_detection():
    """Basit anomali tespiti testi"""
    
    print("\n🚨 Basit Anomali Tespiti Testi...")
    
    model = load_trained_model()
    if model is None:
        return
    
    # Test durumları
    test_cases = []
    
    # Normal Mars görüntüsü
    data_dir = Path("mars_images/valid")
    if data_dir.exists():
        for category_dir in data_dir.iterdir():
            if category_dir.is_dir():
                image_files = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
                if image_files:
                    test_cases.append(("Normal Mars", str(image_files[0])))
                    break
    
    # Anormal görüntüler
    test_cases.append(("Rastgele Noise", np.random.rand(128, 128, 3).astype(np.float32)))
    test_cases.append(("Siyah Görüntü", np.zeros((128, 128, 3), dtype=np.float32)))
    test_cases.append(("Beyaz Görüntü", np.ones((128, 128, 3), dtype=np.float32)))
    
    results = []
    
    for case_name, img_data in test_cases:
        try:
            if isinstance(img_data, str):
                # Dosyadan yükle
                image = Image.open(img_data).convert('RGB')
                image = image.resize((128, 128), Image.LANCZOS)
                image_array = np.array(image, dtype=np.float32) / 255.0
            else:
                # NumPy array
                image_array = img_data
            
            # Tensor'a çevir
            input_tensor = torch.from_numpy(image_array).float()
            input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
            
            # Model tahmini
            with torch.no_grad():
                reconstructed, _ = model(input_tensor)
            
            # MSE hesapla
            reconstructed = reconstructed.squeeze(0).permute(1, 2, 0).numpy()
            mse = np.mean((image_array - reconstructed) ** 2)
            
            results.append((case_name, mse, image_array, reconstructed))
            print(f"  {case_name}: Anomali Skoru = {mse:.6f}")
            
        except Exception as e:
            print(f"  Hata ({case_name}): {e}")
    
    # Sonuçları görselleştir
    if results:
        fig, axes = plt.subplots(2, len(results), figsize=(4*len(results), 8))
        
        for i, (case_name, mse, original, reconstructed) in enumerate(results):
            # Orijinal
            axes[0, i].imshow(original)
            axes[0, i].set_title(f"{case_name}\nOrijinal")
            axes[0, i].axis('off')
            
            # Yeniden oluşturulan
            axes[1, i].imshow(reconstructed)
            axes[1, i].set_title(f"Yeniden Oluşturulan\nMSE: {mse:.6f}")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/simple_anomaly_test.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    print("✅ Anomali tespiti testi tamamlandı!")

def generate_model_summary():
    """Model özeti oluştur"""
    
    print("\n📋 MODEL ÖZETİ")
    print("=" * 50)
    
    # Eğitim verisi analizi
    categories, total_images = analyze_training_data()
    
    # Model bilgileri
    model = load_trained_model()
    if model:
        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        print(f"\n🤖 Model Bilgileri:")
        print(f"  Parametre sayısı: {total_params:,}")
        print(f"  Model boyutu: {model_size_mb:.2f} MB")
        print(f"  Latent boyutu: 1024")
        print(f"  Giriş boyutu: 128x128x3")
        
        print(f"\n📊 Eğitim Verisi:")
        print(f"  Toplam görüntü: {total_images}")
        print(f"  Kategori sayısı: {len(categories)}")
        
        print(f"\n🎯 Model Ne Öğrendi:")
        print(f"  - Mars yüzey görüntülerinin normal görünümü")
        print(f"  - Farklı Mars yüzey kategorileri:")
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    * {category}: {count} görüntü")
        
        print(f"\n🔍 Model Ne Yapabilir:")
        print(f"  ✅ Normal Mars görüntülerini yeniden oluşturabilir")
        print(f"  ✅ Anormal görüntüleri tespit edebilir (yüksek MSE)")
        print(f"  ✅ Görüntüleri 1024 boyutlu latent space'e sıkıştırabilir")
        
        print(f"\n❌ Model Ne Yapamaz:")
        print(f"  - Perseverance verilerini tanımaz (sadece Curiosity)")
        print(f"  - Yeni Mars bölgelerini tanımaz")
        print(f"  - Dünya görüntülerini anlamaz")

if __name__ == "__main__":
    print("🚀 Basit Model Analizi Başlıyor...")
    
    # 1. Eğitim verisi analizi
    analyze_training_data()
    
    # 2. Basit yeniden oluşturma testi
    test_simple_reconstruction()
    
    # 3. Basit anomali tespiti testi
    test_anomaly_detection()
    
    # 4. Model özeti
    generate_model_summary()
    
    print("\n✅ Basit model analizi tamamlandı!") 