"""
Eğitilen Modeli Analiz Et
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path
from src.models.optimized_autoencoder import OptimizedAutoencoder
from src.utils.data_utils import extract_features
import cv2

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

def test_model_reconstruction():
    """Modelin yeniden oluşturma yeteneğini test et"""
    
    print("\n🔄 Model Yeniden Oluşturma Testi...")
    
    model = load_trained_model()
    if model is None:
        return
    
    # Test görüntüleri seç
    test_images = []
    data_dir = Path("mars_images")
    
    # Farklı kategorilerden örnekler al
    for split in ['valid']:
        split_dir = data_dir / split
        if split_dir.exists():
            for category_dir in split_dir.iterdir():
                if category_dir.is_dir():
                    image_files = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
                    if image_files:
                        test_images.append(str(image_files[0]))
                        if len(test_images) >= 6:  # 6 farklı kategori
                            break
    
    if not test_images:
        print("❌ Test görüntüsü bulunamadı")
        return
    
    # Görüntüleri test et
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    
    for i, img_path in enumerate(test_images):
        try:
            # Orijinal görüntüyü yükle
            image = Image.open(img_path).convert('RGB')
            image = image.resize((128, 128), Image.LANCZOS)
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            # Tensor'a çevir
            input_tensor = torch.from_numpy(image_array).float()
            input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)  # CHW -> BCHW
            
            # Model tahmini
            with torch.no_grad():
                reconstructed, latent = model(input_tensor)
            
            # Sonuçları görselleştir
            reconstructed = reconstructed.squeeze(0).permute(1, 2, 0).numpy()
            reconstructed = np.clip(reconstructed, 0, 1)
            
            # Orijinal görüntü
            axes[0, i].imshow(image_array)
            axes[0, i].set_title(f"Orijinal\n{Path(img_path).parent.name}")
            axes[0, i].axis('off')
            
            # Yeniden oluşturulan görüntü
            axes[1, i].imshow(reconstructed)
            axes[1, i].set_title(f"Yeniden Oluşturulan")
            axes[1, i].axis('off')
            
            # MSE hesapla
            mse = np.mean((image_array - reconstructed) ** 2)
            print(f"  {Path(img_path).parent.name}: MSE = {mse:.6f}")
            
        except Exception as e:
            print(f"  Hata ({img_path}): {e}")
    
    plt.tight_layout()
    plt.savefig('results/model_reconstruction_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Yeniden oluşturma testi tamamlandı!")

def analyze_latent_space():
    """Latent space'i analiz et"""
    
    print("\n🧠 Latent Space Analizi...")
    
    model = load_trained_model()
    if model is None:
        return
    
    # Farklı kategorilerden görüntüler al
    data_dir = Path("mars_images")
    category_latents = {}
    
    for split in ['valid']:
        split_dir = data_dir / split
        if split_dir.exists():
            for category_dir in split_dir.iterdir():
                if category_dir.is_dir():
                    category = category_dir.name
                    image_files = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
                    
                    if image_files and len(image_files) >= 5:
                        latents = []
                        
                        for img_file in image_files[:5]:  # Her kategoriden 5 görüntü
                            try:
                                image = Image.open(img_file).convert('RGB')
                                image = image.resize((128, 128), Image.LANCZOS)
                                image_array = np.array(image, dtype=np.float32) / 255.0
                                
                                input_tensor = torch.from_numpy(image_array).float()
                                input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
                                
                                with torch.no_grad():
                                    _, latent = model(input_tensor)
                                
                                latents.append(latent.squeeze().numpy())
                                
                            except Exception as e:
                                continue
                        
                        if latents:
                            category_latents[category] = np.array(latents)
    
    # Latent space görselleştirmesi
    if len(category_latents) >= 2:
        # PCA ile 2D'ye indir
        from sklearn.decomposition import PCA
        
        all_latents = []
        all_categories = []
        
        for category, latents in category_latents.items():
            all_latents.extend(latents)
            all_categories.extend([category] * len(latents))
        
        all_latents = np.array(all_latents)
        
        # PCA uygula
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(all_latents)
        
        # Görselleştir
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(category_latents)))
        
        for i, category in enumerate(category_latents.keys()):
            mask = np.array(all_categories) == category
            plt.scatter(latents_2d[mask, 0], latents_2d[mask, 1], 
                       label=category, color=colors[i], alpha=0.7, s=50)
        
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('Latent Space - Kategori Dağılımı')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('results/latent_space_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Latent space analizi tamamlandı!")
        print(f"Toplam kategori: {len(category_latents)}")
        print(f"Toplam görüntü: {len(all_latents)}")
        print(f"Latent boyutu: {all_latents.shape[1]}")

def test_anomaly_detection():
    """Anomali tespiti test et"""
    
    print("\n🚨 Anomali Tespiti Testi...")
    
    model = load_trained_model()
    if model is None:
        return
    
    # Test görüntüleri
    test_cases = []
    
    # Normal Mars görüntüleri
    data_dir = Path("mars_images/valid")
    if data_dir.exists():
        for category_dir in data_dir.iterdir():
            if category_dir.is_dir():
                image_files = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
                if image_files:
                    test_cases.append(("Normal", str(image_files[0])))
                    break
    
    # Anormal görüntüler (rastgele noise)
    noise_image = np.random.rand(128, 128, 3).astype(np.float32)
    test_cases.append(("Anormal (Noise)", noise_image))
    
    # Tamamen siyah görüntü
    black_image = np.zeros((128, 128, 3), dtype=np.float32)
    test_cases.append(("Anormal (Siyah)", black_image))
    
    # Tamamen beyaz görüntü
    white_image = np.ones((128, 128, 3), dtype=np.float32)
    test_cases.append(("Anormal (Beyaz)", white_image))
    
    # Test et
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
            
            # MSE hesapla (anomali skoru)
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
        plt.savefig('results/anomaly_detection_test.png', dpi=150, bbox_inches='tight')
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
        print(f"  ✅ Kategori bazlı latent space gruplandırması")
        
        print(f"\n❌ Model Ne Yapamaz:")
        print(f"  - Perseverance verilerini tanımaz (sadece Curiosity)")
        print(f"  - Yeni Mars bölgelerini tanımaz")
        print(f"  - Dünya görüntülerini anlamaz")
        print(f"  - Renkli/kompleks görüntüleri iyi yeniden oluşturamaz")

if __name__ == "__main__":
    print("🚀 Model Analizi Başlıyor...")
    
    # 1. Eğitim verisi analizi
    analyze_training_data()
    
    # 2. Model yeniden oluşturma testi
    test_model_reconstruction()
    
    # 3. Latent space analizi
    analyze_latent_space()
    
    # 4. Anomali tespiti testi
    test_anomaly_detection()
    
    # 5. Model özeti
    generate_model_summary()
    
    print("\n✅ Model analizi tamamlandı!") 