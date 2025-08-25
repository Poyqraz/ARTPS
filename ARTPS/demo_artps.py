"""
ARTPS - Otonom Bilimsel Keşif Sistemi Demo

Bu script, eğitilmiş autoencoder modelini kullanarak:
1. Mars kaya görüntülerinde anomali tespiti yapar
2. İlginçlik puanlarını hesaplar
3. Sonuçları görselleştirir
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Proje modüllerini import et
from src.models.working_autoencoder import WorkingAutoencoder, AutoencoderTrainer, MarsRockDataset
from src.utils.data_utils import extract_features, calculate_similarity, calculate_curiosity_score


def load_trained_model(model_path: str, device: str = 'cuda'):
    """Eğitilmiş modeli yükle"""
    print(f"Model yükleniyor: {model_path}")
    
    # Model oluştur
    model = WorkingAutoencoder(input_channels=3, latent_dim=16384)
    trainer = AutoencoderTrainer(model, device=device)
    
    # Modeli yükle
    trainer.load_model(model_path)
    
    print("✅ Model başarıyla yüklendi!")
    return trainer


def analyze_mars_images(trainer, data_dir: str, num_samples: int = 10):
    """Mars görüntülerini analiz et ve anomali skorlarını hesapla"""
    print(f"\n=== Mars Görüntü Analizi ===")
    print(f"Analiz edilecek görüntü sayısı: {num_samples}")
    
    # Veri setini yükle
    dataset = MarsRockDataset(data_dir, target_size=(128, 128))
    
    if len(dataset) == 0:
        print("❌ Veri seti bulunamadı!")
        return None
    
    # Analiz sonuçları
    results = []
    
    for i in range(min(num_samples, len(dataset))):
        image = dataset[i]
        image_path = dataset.image_files[i]
        
        # Anomali skoru hesapla (Exploration Score)
        anomaly_score = trainer.calculate_anomaly_score(image.unsqueeze(0))
        
        # Özellik çıkar (Exploitation Score için)
        features = extract_features(image)
        
        # Basit bir exploitation score (örnek olarak)
        # Gerçek uygulamada bu, bilinen değerli hedeflerle karşılaştırılacak
        exploitation_score = np.random.uniform(0.1, 0.9)  # Simüle edilmiş
        
        # İlginçlik puanını hesapla
        curiosity_score = calculate_curiosity_score(exploitation_score, anomaly_score)
        
        results.append({
            'image_path': image_path,
            'image': image,
            'anomaly_score': anomaly_score,
            'exploitation_score': exploitation_score,
            'curiosity_score': curiosity_score,
            'features': features
        })
        
        print(f"Görüntü {i+1}: Anomali={anomaly_score:.4f}, "
              f"Bilinen Değer={exploitation_score:.4f}, "
              f"İlginçlik={curiosity_score:.4f}")
    
    return results


def visualize_analysis_results(results, save_path: str = "results/artps_analysis.png"):
    """Analiz sonuçlarını görselleştir"""
    if not results:
        print("❌ Görselleştirilecek sonuç yok!")
        return
    
    print(f"\n=== Sonuç Görselleştirme ===")
    
    # En yüksek ilginçlik puanına sahip görüntüleri seç
    sorted_results = sorted(results, key=lambda x: x['curiosity_score'], reverse=True)
    top_results = sorted_results[:6]  # İlk 6 görüntü
    
    # Görselleştirme
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('ARTPS - En İlginç Mars Kaya Görüntüleri', fontsize=16, fontweight='bold')
    
    for i, result in enumerate(top_results):
        row = i // 3
        col = i % 3
        
        # Görüntüyü göster
        image = result['image'].permute(1, 2, 0).cpu().numpy()
        axes[row, col].imshow(image)
        
        # Başlık ve skorları göster
        title = f"İlginçlik: {result['curiosity_score']:.3f}\n"
        title += f"Anomali: {result['anomaly_score']:.3f}\n"
        title += f"Bilinen Değer: {result['exploitation_score']:.3f}"
        
        axes[row, col].set_title(title, fontsize=10)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Görselleştirme kaydedildi: {save_path}")
    plt.show()


def demonstrate_anomaly_detection(trainer):
    """Anomali tespiti demo'su"""
    print(f"\n=== Anomali Tespiti Demo ===")
    
    # Normal Mars kaya görüntüsü (sentetik)
    normal_image = torch.randn(3, 128, 128) * 0.5 + 0.5  # [0,1] aralığında
    normal_image = torch.clamp(normal_image, 0, 1)
    
    # Anormal görüntü (çok farklı renk ve doku)
    abnormal_image = torch.randn(3, 128, 128) * 2.0  # Çok farklı dağılım
    abnormal_image = torch.clamp(abnormal_image, 0, 1)
    
    # Skorları hesapla
    normal_score = trainer.calculate_anomaly_score(normal_image.unsqueeze(0))
    abnormal_score = trainer.calculate_anomaly_score(abnormal_image.unsqueeze(0))
    
    print(f"Normal görüntü anomali skoru: {normal_score:.6f}")
    print(f"Anormal görüntü anomali skoru: {abnormal_score:.6f}")
    
    if abnormal_score > normal_score:
        print("✅ Anomali tespiti çalışıyor - anormal görüntü daha yüksek skor aldı!")
    else:
        print("⚠️ Anomali tespiti beklenen sonucu vermedi.")
    
    # Görselleştir
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(normal_image.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title(f'Normal Görüntü\nAnomali Skoru: {normal_score:.4f}')
    axes[0].axis('off')
    
    axes[1].imshow(abnormal_image.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title(f'Anormal Görüntü\nAnomali Skoru: {abnormal_score:.4f}')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig("results/anomaly_detection_demo.png", dpi=300, bbox_inches='tight')
    print("✅ Anomali tespiti demo'su kaydedildi: results/anomaly_detection_demo.png")
    plt.show()


def main():
    """Ana demo fonksiyonu"""
    print("🚀 ARTPS - Otonom Bilimsel Keşif Sistemi Demo Başlıyor...\n")
    
    # Cihaz kontrolü
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    # Model yolu
    model_path = "results/working_autoencoder_model.pth"
    data_dir = "data/mars_rocks"
    
    # Model yükle
    if not os.path.exists(model_path):
        print(f"❌ Model dosyası bulunamadı: {model_path}")
        print("Önce 'python test_working_autoencoder.py' komutunu çalıştırın.")
        return
    
    trainer = load_trained_model(model_path, device=device)
    
    # Anomali tespiti demo'su
    demonstrate_anomaly_detection(trainer)
    
    # Mars görüntülerini analiz et
    results = analyze_mars_images(trainer, data_dir, num_samples=15)
    
    if results:
        # Sonuçları görselleştir
        visualize_analysis_results(results)
        
        # İstatistikler
        anomaly_scores = [r['anomaly_score'] for r in results]
        curiosity_scores = [r['curiosity_score'] for r in results]
        
        print(f"\n=== İstatistikler ===")
        print(f"Ortalama anomali skoru: {np.mean(anomaly_scores):.4f} ± {np.std(anomaly_scores):.4f}")
        print(f"Ortalama ilginçlik puanı: {np.mean(curiosity_scores):.4f} ± {np.std(curiosity_scores):.4f}")
        print(f"En yüksek ilginçlik puanı: {max(curiosity_scores):.4f}")
        print(f"En düşük ilginçlik puanı: {min(curiosity_scores):.4f}")
    
    print(f"\n🎉 ARTPS Demo tamamlandı!")
    print(f"📁 Sonuçlar 'results' klasöründe bulunabilir.")


if __name__ == "__main__":
    main() 