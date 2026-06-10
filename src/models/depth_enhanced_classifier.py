"""
ARTPS - Derinlik Geliştirilmiş Kategori Bazlı Sınıflandırıcı
Derinlik algısı + kategori bazlı otomatik etiketleme
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import cv2

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.depth_estimation import MiDaSDepthEstimator
from src.models.optimized_autoencoder import OptimizedAutoencoder

class DepthEnhancedClassifier(nn.Module):
    """
    Derinlik geliştirilmiş bilimsel değer sınıflandırıcısı
    """
    
    def __init__(self, num_classes=5, rgb_features=1024, depth_features=14):
        super(DepthEnhancedClassifier, self).__init__()
        
        # RGB özellikleri için (autoencoder latent features)
        self.rgb_feature_extractor = nn.Sequential(
            nn.Linear(rgb_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Derinlik özellikleri için
        self.depth_feature_extractor = nn.Sequential(
            nn.Linear(depth_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Birleştirilmiş özellikler için sınıflandırıcı
        combined_features = 256 + 32  # RGB + Depth features
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, rgb_features, depth_features):
        """
        Forward pass
        
        Args:
            rgb_features: Autoencoder latent features
            depth_features: Derinlik özellikleri
            
        Returns:
            Predictions: Sınıflandırma tahminleri
        """
        # RGB özelliklerini işle
        rgb_processed = self.rgb_feature_extractor(rgb_features)
        
        # Derinlik özelliklerini işle
        depth_processed = self.depth_feature_extractor(depth_features)
        
        # Özellikleri birleştir
        combined = torch.cat([rgb_processed, depth_processed], dim=1)
        
        # Sınıflandır
        predictions = self.classifier(combined)
        
        return predictions

class DepthEnhancedDataset(Dataset):
    """
    Derinlik geliştirilmiş veri seti
    """
    
    def __init__(self, data_dir, autoencoder_model, depth_estimator, transform=None):
        self.data_dir = Path(data_dir)
        self.autoencoder = autoencoder_model
        self.depth_estimator = depth_estimator
        self.transform = transform
        self.samples = []
        
        # Kategori bazlı bilimsel değer etiketleri
        self.value_labels = {
            'rocky': 4,        # Yüksek değer - Karmaşık jeoloji
            'boulder': 3,      # Orta-yüksek değer - Büyük kayalar
            'hills_or_ridge': 3, # Orta-yüksek değer - Yapısal özellikler
            'flat_terrain': 1, # Düşük değer - Sıradan yüzey
            'dusty': 1,        # Düşük değer - Tozlu yüzey
            'rover': 0,        # Değersiz - Rover parçaları
        }
        
        self._load_samples()
    
    def _load_samples(self):
        """Veri setini yükle"""
        print("📊 Veri seti yükleniyor...")
        
        for split in ['train', 'valid']:
            split_dir = self.data_dir / split
            if split_dir.exists():
                for category_dir in split_dir.iterdir():
                    if category_dir.is_dir():
                        category = category_dir.name
                        if category in self.value_labels:
                            label = self.value_labels[category]
                            image_files = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
                            
                            for img_file in image_files:
                                self.samples.append({
                                    'image_path': str(img_file),
                                    'category': category,
                                    'value_label': label
                                })
        
        print(f"✅ {len(self.samples)} örnek yüklendi")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Görüntüyü yükle
            image = Image.open(sample['image_path']).convert('RGB')
            image = image.resize((128, 128), Image.LANCZOS)
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            # Autoencoder ile RGB latent features çıkar
            with torch.no_grad():
                input_tensor = torch.from_numpy(image_array).float()
                input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
                _, latent_features = self.autoencoder(input_tensor)
                latent_features = latent_features.squeeze()
            
            # Derinlik tahmini yap
            depth_map, _ = self.depth_estimator.estimate_depth(image_array)
            
            # Derinlik özelliklerini çıkar
            depth_features = self.depth_estimator.extract_depth_features(depth_map)
            depth_features_tensor = torch.tensor(list(depth_features.values()), dtype=torch.float32)
            
            return {
                'rgb_features': latent_features,
                'depth_features': depth_features_tensor,
                'value_label': torch.tensor(sample['value_label'], dtype=torch.long),
                'category': sample['category'],
                'image_path': sample['image_path'],
                'depth_map': depth_map
            }
            
        except Exception as e:
            print(f"❌ Hata ({sample['image_path']}): {e}")
            # Fallback: Sıfır özellikler
            return {
                'rgb_features': torch.zeros(1024),
                'depth_features': torch.zeros(14),
                'value_label': torch.tensor(sample['value_label'], dtype=torch.long),
                'category': sample['category'],
                'image_path': sample['image_path'],
                'depth_map': np.zeros((128, 128))
            }

def create_depth_enhanced_classifier():
    """
    Derinlik geliştirilmiş sınıflandırıcı oluştur ve eğit
    """
    
    print("🚀 Derinlik Geliştirilmiş Sınıflandırıcı Oluşturuluyor...")
    
    # 1. Modelleri yükle
    print("📥 Modeller yükleniyor...")
    
    # Autoencoder
    autoencoder_path = "results/optimized_autoencoder_curiosity_extended.pth"
    if not os.path.exists(autoencoder_path):
        print(f"❌ Autoencoder model bulunamadı: {autoencoder_path}")
        return None
    
    autoencoder = OptimizedAutoencoder(input_channels=3, latent_dim=1024)
    checkpoint = torch.load(autoencoder_path, map_location='cpu')
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    autoencoder.eval()
    
    # Derinlik tahmin modülü
    depth_estimator = MiDaSDepthEstimator(model_type="DPT_Large")
    
    # 2. Veri setini oluştur
    print("📊 Veri seti oluşturuluyor...")
    dataset = DepthEnhancedDataset("mars_images", autoencoder, depth_estimator)
    
    if len(dataset) == 0:
        print("❌ Veri seti boş!")
        return None
    
    # Kategori dağılımını analiz et
    category_counts = {}
    value_distribution = {}
    
    for sample in dataset.samples:
        category = sample['category']
        value = sample['value_label']
        
        category_counts[category] = category_counts.get(category, 0) + 1
        value_distribution[value] = value_distribution.get(value, 0) + 1
    
    print("\n📈 Kategori Dağılımı:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count} örnek")
    
    print("\n🎯 Bilimsel Değer Dağılımı:")
    value_names = {0: "Değersiz", 1: "Düşük", 2: "Orta", 3: "Orta-Yüksek", 4: "Yüksek"}
    for value, count in sorted(value_distribution.items()):
        print(f"  {value_names[value]} ({value}): {count} örnek")
    
    # 3. Veri setini böl
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 4. DataLoader'ları oluştur
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # 5. Model oluştur
    num_classes = 5  # 0-4 arası değer kategorileri
    classifier = DepthEnhancedClassifier(num_classes=num_classes, rgb_features=1024, depth_features=14)
    
    # 6. Eğitim parametreleri
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = classifier.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 7. Eğitim döngüsü
    num_epochs = 30
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\n🎓 Eğitim başlıyor ({num_epochs} epoch)...")
    
    for epoch in range(num_epochs):
        # Eğitim
        classifier.train()
        train_loss = 0.0
        for batch in train_loader:
            rgb_features = batch['rgb_features'].to(device)
            depth_features = batch['depth_features'].to(device)
            labels = batch['value_label'].to(device)
            
            optimizer.zero_grad()
            outputs = classifier(rgb_features, depth_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validasyon
        classifier.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                rgb_features = batch['rgb_features'].to(device)
                depth_features = batch['depth_features'].to(device)
                labels = batch['value_label'].to(device)
                
                outputs = classifier(rgb_features, depth_features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Accuracy: {accuracy:.2f}%")
        
        # En iyi modeli kaydet
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'accuracy': accuracy
            }, "results/depth_enhanced_classifier.pth")
            print(f"  ✅ Yeni en iyi model kaydedildi")
    
    # 8. Eğitim eğrilerini çiz
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Derinlik Geliştirilmiş Sınıflandırıcı Eğitimi')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validasyon Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/depth_enhanced_classifier_training.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Derinlik geliştirilmiş sınıflandırıcı eğitimi tamamlandı!")
    print(f"📁 Model kaydedildi: results/depth_enhanced_classifier.pth")
    print(f"🎯 En iyi validasyon loss: {best_val_loss:.4f}")
    print(f"📊 Final doğruluk: {accuracy:.2f}%")
    
    return classifier

def test_depth_enhanced_classifier():
    """
    Derinlik geliştirilmiş sınıflandırıcıyı test et
    """
    
    print("🧪 Derinlik Geliştirilmiş Sınıflandırıcı Test Ediliyor...")
    
    # Model yükle
    classifier_path = "results/depth_enhanced_classifier.pth"
    if not os.path.exists(classifier_path):
        print(f"❌ Sınıflandırıcı model bulunamadı: {classifier_path}")
        return
    
    # Autoencoder yükle
    autoencoder_path = "results/optimized_autoencoder_curiosity_extended.pth"
    autoencoder = OptimizedAutoencoder(input_channels=3, latent_dim=1024)
    checkpoint = torch.load(autoencoder_path, map_location='cpu')
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    autoencoder.eval()
    
    # Derinlik tahmin modülü
    depth_estimator = MiDaSDepthEstimator(model_type="DPT_Large")
    
    # Sınıflandırıcı yükle
    classifier = DepthEnhancedClassifier(num_classes=5, rgb_features=1024, depth_features=14)
    checkpoint = torch.load(classifier_path, map_location='cpu')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()
    
    # Test veri seti oluştur
    test_dataset = DepthEnhancedDataset("mars_images", autoencoder, depth_estimator)
    
    # Test et
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = classifier.to(device)
    
    all_predictions = []
    all_labels = []
    all_categories = []
    
    with torch.no_grad():
        for i in range(min(50, len(test_dataset))):  # İlk 50 örnek
            sample = test_dataset[i]
            rgb_features = sample['rgb_features'].unsqueeze(0).to(device)
            depth_features = sample['depth_features'].unsqueeze(0).to(device)
            label = sample['value_label']
            category = sample['category']
            
            output = classifier(rgb_features, depth_features)
            _, predicted = torch.max(output, 1)
            
            all_predictions.append(predicted.item())
            all_labels.append(label.item())
            all_categories.append(category)
    
    # Sonuçları analiz et
    print("\n📊 Sınıflandırma Raporu:")
    value_names = {0: "Değersiz", 1: "Düşük", 2: "Orta", 3: "Orta-Yüksek", 4: "Yüksek"}
    
    # Gerçek sınıf sayısını kontrol et
    unique_labels = sorted(set(all_labels))
    unique_predictions = sorted(set(all_predictions))
    
    print(f"Gerçek sınıflar: {unique_labels}")
    print(f"Tahmin edilen sınıflar: {unique_predictions}")
    
    # Sadece mevcut sınıflar için rapor oluştur
    available_classes = sorted(set(all_labels + all_predictions))
    target_names = [value_names.get(i, f"Sınıf_{i}") for i in available_classes]
    
    print(classification_report(all_labels, all_predictions, 
                               labels=available_classes,
                               target_names=target_names))
    
    # Kategori bazlı analiz
    print("\n🎯 Kategori Bazlı Bilimsel Değer Tahminleri:")
    category_predictions = {}
    for i, category in enumerate(all_categories):
        if category not in category_predictions:
            category_predictions[category] = []
        category_predictions[category].append(all_predictions[i])
    
    for category, predictions in category_predictions.items():
        avg_value = np.mean(predictions)
        print(f"  {category}: Ortalama Değer = {avg_value:.2f} ({value_names[int(avg_value)]})")
    
    return classifier

if __name__ == "__main__":
    print("🚀 Derinlik Geliştirilmiş Kategori Bazlı Sınıflandırıcı Projesi")
    print("=" * 60)
    
    # 1. Sınıflandırıcı oluştur ve eğit
    classifier = create_depth_enhanced_classifier()
    
    if classifier:
        # 2. Test et
        test_depth_enhanced_classifier()
    
    print("\n✅ Proje tamamlandı!") 