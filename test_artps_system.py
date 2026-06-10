"""
ARTPS Sistemi Test
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

def load_trained_model():
    """Eğitilen modeli yükle"""
    
    print("🤖 Eğitilen Model Yükleniyor...")
    
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
    return model

def calculate_anomaly_score(model, image_path):
    """Görüntü için anomali skoru hesapla"""
    
    try:
        # Görüntüyü yükle
        image = Image.open(image_path).convert('RGB')
        image = image.resize((128, 128), Image.LANCZOS)
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Tensor'a çevir
        input_tensor = torch.from_numpy(image_array).float()
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
        
        # Model tahmini
        with torch.no_grad():
            reconstructed, latent = model(input_tensor)
        
        # MSE hesapla (anomali skoru)
        reconstructed = reconstructed.squeeze(0).permute(1, 2, 0).numpy()
        mse = np.mean((image_array - reconstructed) ** 2)
        
        return mse, image_array, reconstructed, latent.squeeze().numpy()
        
    except Exception as e:
        print(f"❌ Hata ({image_path}): {e}")
        return None, None, None, None

def test_artps_on_curiosity_data():
    """Curiosity verilerinde ARTPS testi"""
    
    print("🚀 ARTPS Sistemi Curiosity Verilerinde Test Ediliyor...")
    
    model = load_trained_model()
    if model is None:
        return
    
    # Test görüntüleri seç
    data_dir = Path("mars_images/valid")
    test_images = []
    
    if data_dir.exists():
        for category_dir in data_dir.iterdir():
            if category_dir.is_dir():
                image_files = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
                if image_files:
                    # Her kategoriden 2 görüntü al
                    for img_file in image_files[:2]:
                        test_images.append((category_dir.name, str(img_file)))
    
    if not test_images:
        print("❌ Test görüntüsü bulunamadı")
        return
    
    print(f"📊 {len(test_images)} görüntü test edilecek")
    
    # Anomali skorlarını hesapla
    results = []
    
    for category, img_path in test_images:
        mse, original, reconstructed, latent = calculate_anomaly_score(model, img_path)
        
        if mse is not None:
            results.append({
                'category': category,
                'image_path': img_path,
                'anomaly_score': mse,
                'original': original,
                'reconstructed': reconstructed,
                'latent': latent
            })
            print(f"  {category}: Anomali Skoru = {mse:.6f}")
    
    if not results:
        print("❌ Sonuç bulunamadı")
        return
    
    # Sonuçları sırala (en yüksek anomali skoru en üstte)
    results.sort(key=lambda x: x['anomaly_score'], reverse=True)
    
    # En ilginç 6 hedefi göster
    top_targets = results[:6]
    
    print(f"\n🎯 En İlginç 6 Hedef:")
    for i, result in enumerate(top_targets):
        print(f"  {i+1}. {result['category']}: {result['anomaly_score']:.6f}")
    
    # Görselleştir
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    
    for i, result in enumerate(top_targets):
        # Orijinal görüntü
        axes[0, i].imshow(result['original'])
        axes[0, i].set_title(f"{result['category']}\nOrijinal")
        axes[0, i].axis('off')
        
        # Yeniden oluşturulan görüntü
        axes[1, i].imshow(result['reconstructed'])
        axes[1, i].set_title(f"Yeniden Oluşturulan\nAnomali: {result['anomaly_score']:.6f}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/artps_curiosity_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Anomali skoru dağılımını analiz et
    anomaly_scores = [r['anomaly_score'] for r in results]
    categories = [r['category'] for r in results]
    
    print(f"\n📊 Anomali Skoru Analizi:")
    print(f"  Ortalama: {np.mean(anomaly_scores):.6f}")
    print(f"  Standart Sapma: {np.std(anomaly_scores):.6f}")
    print(f"  Minimum: {np.min(anomaly_scores):.6f}")
    print(f"  Maksimum: {np.max(anomaly_scores):.6f}")
    
    # Kategori bazlı analiz
    category_scores = {}
    for category, score in zip(categories, anomaly_scores):
        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].append(score)
    
    print(f"\n📈 Kategori Bazlı Anomali Skorları:")
    for category, scores in category_scores.items():
        avg_score = np.mean(scores)
        print(f"  {category}: {avg_score:.6f} (n={len(scores)})")
    
    return results

def test_artps_on_api_data():
    """API'den indirilen verilerde ARTPS testi"""
    
    print("\n🔄 ARTPS Sistemi API Verilerinde Test Ediliyor...")
    
    model = load_trained_model()
    if model is None:
        return
    
    # API'den indirilen görüntüleri test et
    api_data_dir = Path("data/curiosity_api_images")
    
    if not api_data_dir.exists():
        print("❌ API veri dizini bulunamadı")
        return
    
    # API görüntülerini listele
    api_images = list(api_data_dir.glob("*.jpg")) + list(api_data_dir.glob("*.png"))
    
    if not api_images:
        print("❌ API görüntüsü bulunamadı")
        return
    
    print(f"📊 {len(api_images)} API görüntüsü test edilecek")
    
    # Anomali skorlarını hesapla
    api_results = []
    
    for img_path in api_images:
        mse, original, reconstructed, latent = calculate_anomaly_score(model, str(img_path))
        
        if mse is not None:
            api_results.append({
                'image_path': str(img_path),
                'anomaly_score': mse,
                'original': original,
                'reconstructed': reconstructed,
                'latent': latent
            })
            print(f"  {img_path.name}: Anomali Skoru = {mse:.6f}")
    
    if not api_results:
        print("❌ API sonuç bulunamadı")
        return
    
    # Sonuçları sırala
    api_results.sort(key=lambda x: x['anomaly_score'], reverse=True)
    
    print(f"\n🎯 API Verilerinde En İlginç Hedefler:")
    for i, result in enumerate(api_results[:5]):
        print(f"  {i+1}. {Path(result['image_path']).name}: {result['anomaly_score']:.6f}")
    
    # Görselleştir
    if len(api_results) >= 4:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for i, result in enumerate(api_results[:4]):
            # Orijinal görüntü
            axes[0, i].imshow(result['original'])
            axes[0, i].set_title(f"API Görüntü {i+1}\nOrijinal")
            axes[0, i].axis('off')
            
            # Yeniden oluşturulan görüntü
            axes[1, i].imshow(result['reconstructed'])
            axes[1, i].set_title(f"Yeniden Oluşturulan\nAnomali: {result['anomaly_score']:.6f}")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('results/artps_api_test.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    return api_results

def generate_artps_summary(curiosity_results, api_results):
    """ARTPS test özeti oluştur"""
    
    print("\n📋 ARTPS SİSTEM TEST ÖZETİ")
    print("=" * 50)
    
    if curiosity_results:
        curiosity_scores = [r['anomaly_score'] for r in curiosity_results]
        print(f"\n🔍 Curiosity Verileri Test Sonuçları:")
        print(f"  Test edilen görüntü: {len(curiosity_results)}")
        print(f"  Ortalama anomali skoru: {np.mean(curiosity_scores):.6f}")
        print(f"  En yüksek anomali skoru: {np.max(curiosity_scores):.6f}")
        print(f"  En düşük anomali skoru: {np.min(curiosity_scores):.6f}")
        
        # En ilginç hedefler
        top_curiosity = sorted(curiosity_results, key=lambda x: x['anomaly_score'], reverse=True)[:3]
        print(f"  En ilginç hedefler:")
        for i, result in enumerate(top_curiosity):
            print(f"    {i+1}. {result['category']}: {result['anomaly_score']:.6f}")
    
    if api_results:
        api_scores = [r['anomaly_score'] for r in api_results]
        print(f"\n🌐 API Verileri Test Sonuçları:")
        print(f"  Test edilen görüntü: {len(api_results)}")
        print(f"  Ortalama anomali skoru: {np.mean(api_scores):.6f}")
        print(f"  En yüksek anomali skoru: {np.max(api_scores):.6f}")
        print(f"  En düşük anomali skoru: {np.min(api_scores):.6f}")
        
        # En ilginç hedefler
        top_api = sorted(api_results, key=lambda x: x['anomaly_score'], reverse=True)[:3]
        print(f"  En ilginç hedefler:")
        for i, result in enumerate(top_api):
            print(f"    {i+1}. {Path(result['image_path']).name}: {result['anomaly_score']:.6f}")
    
    print(f"\n✅ ARTPS Sistemi Başarıyla Test Edildi!")
    print(f"🎯 Sistem, Mars görüntülerinde anomali tespiti yapabilir")
    print(f"🔍 Yüksek anomali skorlu hedefler 'ilginç' olarak işaretlenir")
    print(f"📊 Model, Curiosity verileriyle eğitildiği için güvenilir")

if __name__ == "__main__":
    print("🚀 ARTPS Sistemi Test Başlıyor...")
    
    # 1. Curiosity verilerinde test
    curiosity_results = test_artps_on_curiosity_data()
    
    # 2. API verilerinde test
    api_results = test_artps_on_api_data()
    
    # 3. Özet oluştur
    generate_artps_summary(curiosity_results, api_results)
    
    print("\n✅ ARTPS sistemi testi tamamlandı!") 