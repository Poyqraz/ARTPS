"""
Otomatik Etiketleme Stratejileri - Manuel Etiketleme Olmadan
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import cv2

class AutomaticLabelingStrategy:
    """
    Manuel etiketleme olmadan otomatik etiketleme stratejileri
    """
    
    def __init__(self, autoencoder_model):
        self.autoencoder = autoencoder_model
        self.value_labels = {
            'rocky': 4,        # Yüksek değer - Karmaşık jeoloji
            'boulder': 3,      # Orta-yüksek değer - Büyük kayalar
            'hills_or_ridge': 3, # Orta-yüksek değer - Yapısal özellikler
            'flat_terrain': 1, # Düşük değer - Sıradan yüzey
            'dusty': 1,        # Düşük değer - Tozlu yüzey
            'rover': 0,        # Değersiz - Rover parçaları
        }
    
    def strategy_1_category_based(self, data_dir):
        """
        Strateji 1: Kategori bazlı otomatik etiketleme
        Mevcut klasör yapısını kullanarak otomatik etiketleme
        """
        print("🎯 Strateji 1: Kategori Bazlı Otomatik Etiketleme")
        
        samples = []
        category_stats = {}
        
        for split in ['train', 'valid']:
            split_dir = Path(data_dir) / split
            if split_dir.exists():
                for category_dir in split_dir.iterdir():
                    if category_dir.is_dir():
                        category = category_dir.name
                        if category in self.value_labels:
                            label = self.value_labels[category]
                            image_files = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
                            
                            # Kategori istatistikleri
                            if category not in category_stats:
                                category_stats[category] = {
                                    'count': 0,
                                    'avg_complexity': 0.0,
                                    'avg_texture': 0.0
                                }
                            
                            for img_file in image_files:
                                # Görüntü karmaşıklığını hesapla
                                complexity = self._calculate_image_complexity(str(img_file))
                                texture = self._calculate_texture_score(str(img_file))
                                
                                samples.append({
                                    'image_path': str(img_file),
                                    'category': category,
                                    'value_label': label,
                                    'complexity': complexity,
                                    'texture': texture
                                })
                                
                                category_stats[category]['count'] += 1
                                category_stats[category]['avg_complexity'] += complexity
                                category_stats[category]['avg_texture'] += texture
        
        # Ortalama değerleri hesapla
        for category in category_stats:
            count = category_stats[category]['count']
            if count > 0:
                category_stats[category]['avg_complexity'] /= count
                category_stats[category]['avg_texture'] /= count
        
        print(f"📊 Toplam {len(samples)} örnek etiketlendi")
        print("\n📈 Kategori İstatistikleri:")
        for category, stats in category_stats.items():
            print(f"  {category}: {stats['count']} örnek, "
                  f"Karmaşıklık: {stats['avg_complexity']:.3f}, "
                  f"Doku: {stats['avg_texture']:.3f}")
        
        return samples, category_stats
    
    def strategy_2_complexity_based(self, data_dir):
        """
        Strateji 2: Görüntü karmaşıklığına dayalı otomatik etiketleme
        """
        print("🎯 Strateji 2: Karmaşıklık Bazlı Otomatik Etiketleme")
        
        samples = []
        all_images = []
        
        # Tüm görüntüleri topla
        for split in ['train', 'valid']:
            split_dir = Path(data_dir) / split
            if split_dir.exists():
                for category_dir in split_dir.iterdir():
                    if category_dir.is_dir():
                        image_files = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
                        for img_file in image_files:
                            all_images.append(str(img_file))
        
        # Karmaşıklık skorlarını hesapla
        complexity_scores = []
        for img_path in all_images:
            complexity = self._calculate_image_complexity(img_path)
            complexity_scores.append(complexity)
        
        # Karmaşıklık skorlarını normalize et ve kategorilere ayır
        complexity_scores = np.array(complexity_scores)
        percentiles = np.percentile(complexity_scores, [20, 40, 60, 80])
        
        for i, img_path in enumerate(all_images):
            complexity = complexity_scores[i]
            
            # Karmaşıklığa göre değer etiketi ata
            if complexity <= percentiles[0]:
                value_label = 0  # Çok düşük karmaşıklık
            elif complexity <= percentiles[1]:
                value_label = 1  # Düşük karmaşıklık
            elif complexity <= percentiles[2]:
                value_label = 2  # Orta karmaşıklık
            elif complexity <= percentiles[3]:
                value_label = 3  # Yüksek karmaşıklık
            else:
                value_label = 4  # Çok yüksek karmaşıklık
            
            samples.append({
                'image_path': img_path,
                'complexity': complexity,
                'value_label': value_label
            })
        
        print(f"📊 {len(samples)} örnek karmaşıklık bazlı etiketlendi")
        print(f"🎯 Karmaşıklık percentilleri: {percentiles}")
        
        return samples
    
    def strategy_3_latent_clustering(self, data_dir, n_clusters=5):
        """
        Strateji 3: Latent space clustering ile otomatik etiketleme
        """
        print("🎯 Strateji 3: Latent Space Clustering")
        
        # Tüm görüntülerin latent features'larını çıkar
        latent_features = []
        image_paths = []
        
        for split in ['train', 'valid']:
            split_dir = Path(data_dir) / split
            if split_dir.exists():
                for category_dir in split_dir.iterdir():
                    if category_dir.is_dir():
                        image_files = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
                        for img_file in image_files:
                            try:
                                # Görüntüyü yükle ve latent features çıkar
                                image = Image.open(str(img_file)).convert('RGB')
                                image = image.resize((128, 128), Image.LANCZOS)
                                image_array = np.array(image, dtype=np.float32) / 255.0
                                
                                with torch.no_grad():
                                    input_tensor = torch.from_numpy(image_array).float()
                                    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
                                    _, latent = self.autoencoder(input_tensor)
                                    latent_features.append(latent.squeeze().numpy())
                                    image_paths.append(str(img_file))
                            except Exception as e:
                                print(f"❌ Hata ({img_file}): {e}")
        
        if len(latent_features) == 0:
            print("❌ Latent features çıkarılamadı!")
            return []
        
        # K-means clustering uygula
        latent_features = np.array(latent_features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(latent_features)
        
        # Cluster'ları analiz et ve değer etiketleri ata
        cluster_centers = kmeans.cluster_centers_
        cluster_complexities = []
        
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            cluster_paths = [image_paths[j] for j in cluster_indices]
            
            # Bu cluster'daki görüntülerin ortalama karmaşıklığını hesapla
            avg_complexity = np.mean([self._calculate_image_complexity(path) for path in cluster_paths])
            cluster_complexities.append(avg_complexity)
        
        # Karmaşıklığa göre cluster'ları sırala ve değer etiketleri ata
        sorted_clusters = np.argsort(cluster_complexities)
        cluster_to_value = {sorted_clusters[i]: i for i in range(n_clusters)}
        
        samples = []
        for i, (img_path, cluster_label) in enumerate(zip(image_paths, cluster_labels)):
            value_label = cluster_to_value[cluster_label]
            complexity = self._calculate_image_complexity(img_path)
            
            samples.append({
                'image_path': img_path,
                'cluster': cluster_label,
                'value_label': value_label,
                'complexity': complexity
            })
        
        print(f"📊 {len(samples)} örnek clustering ile etiketlendi")
        print(f"🎯 {n_clusters} cluster oluşturuldu")
        
        # Cluster analizi
        print("\n📈 Cluster Analizi:")
        for i in range(n_clusters):
            cluster_samples = [s for s in samples if s['cluster'] == i]
            avg_complexity = np.mean([s['complexity'] for s in cluster_samples])
            print(f"  Cluster {i}: {len(cluster_samples)} örnek, "
                  f"Ort. Karmaşıklık: {avg_complexity:.3f}, "
                  f"Değer: {cluster_to_value[i]}")
        
        return samples
    
    def _calculate_image_complexity(self, image_path):
        """
        Görüntü karmaşıklığını hesapla
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return 0.0
            
            # Gri tonlamaya çevir
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Gradient hesapla
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Karmaşıklık = gradient büyüklüğünün standart sapması
            complexity = np.std(gradient_magnitude)
            
            return complexity
        except Exception as e:
            return 0.0
    
    def _calculate_texture_score(self, image_path):
        """
        Doku skorunu hesapla
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return 0.0
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # GLCM (Gray Level Co-occurrence Matrix) benzeri özellik
            # Basit olarak komşu pikseller arası farkların varyansı
            diff_h = np.diff(gray, axis=1)
            diff_v = np.diff(gray, axis=0)
            
            texture_score = np.std(diff_h) + np.std(diff_v)
            
            return texture_score
        except Exception as e:
            return 0.0
    
    def visualize_labeling_strategies(self, samples_1, samples_2, samples_3):
        """
        Farklı etiketleme stratejilerini görselleştir
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Strateji 1: Kategori bazlı
        if samples_1:
            categories = [s['category'] for s in samples_1 if 'category' in s]
            values = [s['value_label'] for s in samples_1]
            
            axes[0, 0].hist(values, bins=5, alpha=0.7, color='blue')
            axes[0, 0].set_title('Strateji 1: Kategori Bazlı')
            axes[0, 0].set_xlabel('Bilimsel Değer')
            axes[0, 0].set_ylabel('Örnek Sayısı')
        
        # Strateji 2: Karmaşıklık bazlı
        if samples_2:
            complexities = [s['complexity'] for s in samples_2]
            values = [s['value_label'] for s in samples_2]
            
            axes[0, 1].scatter(complexities, values, alpha=0.6, color='red')
            axes[0, 1].set_title('Strateji 2: Karmaşıklık Bazlı')
            axes[0, 1].set_xlabel('Görüntü Karmaşıklığı')
            axes[0, 1].set_ylabel('Bilimsel Değer')
        
        # Strateji 3: Clustering
        if samples_3:
            clusters = [s['cluster'] for s in samples_3]
            values = [s['value_label'] for s in samples_3]
            
            axes[1, 0].scatter(clusters, values, alpha=0.6, color='green')
            axes[1, 0].set_title('Strateji 3: Clustering')
            axes[1, 0].set_xlabel('Cluster')
            axes[1, 0].set_ylabel('Bilimsel Değer')
        
        # Karşılaştırma
        if samples_1 and samples_2 and samples_3:
            values_1 = [s['value_label'] for s in samples_1]
            values_2 = [s['value_label'] for s in samples_2]
            values_3 = [s['value_label'] for s in samples_3]
            
            axes[1, 1].hist(values_1, bins=5, alpha=0.5, label='Kategori', color='blue')
            axes[1, 1].hist(values_2, bins=5, alpha=0.5, label='Karmaşıklık', color='red')
            axes[1, 1].hist(values_3, bins=5, alpha=0.5, label='Clustering', color='green')
            axes[1, 1].set_title('Strateji Karşılaştırması')
            axes[1, 1].set_xlabel('Bilimsel Değer')
            axes[1, 1].set_ylabel('Örnek Sayısı')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('results/automatic_labeling_strategies.png', dpi=150, bbox_inches='tight')
        plt.show()

def main():
    """
    Otomatik etiketleme stratejilerini test et
    """
    print("🚀 Otomatik Etiketleme Stratejileri Test Ediliyor...")
    
    # Autoencoder modelini yükle
    from src.models.optimized_autoencoder import OptimizedAutoencoder
    
    autoencoder_path = "results/optimized_autoencoder_curiosity_extended.pth"
    if not os.path.exists(autoencoder_path):
        print(f"❌ Autoencoder model bulunamadı: {autoencoder_path}")
        return
    
    autoencoder = OptimizedAutoencoder(input_channels=3, latent_dim=1024)
    checkpoint = torch.load(autoencoder_path, map_location='cpu')
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    autoencoder.eval()
    
    # Etiketleme stratejilerini oluştur
    labeling = AutomaticLabelingStrategy(autoencoder)
    
    # Strateji 1: Kategori bazlı
    print("\n" + "="*50)
    samples_1, category_stats = labeling.strategy_1_category_based("mars_images")
    
    # Strateji 2: Karmaşıklık bazlı
    print("\n" + "="*50)
    samples_2 = labeling.strategy_2_complexity_based("mars_images")
    
    # Strateji 3: Clustering
    print("\n" + "="*50)
    samples_3 = labeling.strategy_3_latent_clustering("mars_images", n_clusters=5)
    
    # Görselleştir
    print("\n" + "="*50)
    labeling.visualize_labeling_strategies(samples_1, samples_2, samples_3)
    
    print("\n✅ Otomatik etiketleme stratejileri tamamlandı!")
    print("📊 Sonuçlar 'results/automatic_labeling_strategies.png' dosyasında")

if __name__ == "__main__":
    main() 