"""
Bilinen Değer Sınıflandırma Modeli - Yol Haritası
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

class KnownValueClassifier(nn.Module):
    """
    Bilinen Değer Sınıflandırma Modeli
    Mars görüntülerini bilimsel değer kategorilerine sınıflandırır
    """
    
    def __init__(self, num_classes=5, feature_dim=1024):
        super(KnownValueClassifier, self).__init__()
        
        # Autoencoder'dan gelen latent features için
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Sınıflandırma katmanı
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, latent_features):
        features = self.feature_extractor(latent_features)
        predictions = self.classifier(features)
        return predictions

class KnownValueDataset(Dataset):
    """
    Bilinen değer etiketli veri seti
    """
    
    def __init__(self, data_dir, autoencoder_model, transform=None):
        self.data_dir = Path(data_dir)
        self.autoencoder = autoencoder_model
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
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Görüntüyü yükle
        image = Image.open(sample['image_path']).convert('RGB')
        image = image.resize((128, 128), Image.LANCZOS)
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Autoencoder ile latent features çıkar
        with torch.no_grad():
            input_tensor = torch.from_numpy(image_array).float()
            input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
            _, latent_features = self.autoencoder(input_tensor)
            latent_features = latent_features.squeeze()
        
        return {
            'latent_features': latent_features,
            'value_label': torch.tensor(sample['value_label'], dtype=torch.long),
            'category': sample['category'],
            'image_path': sample['image_path']
        }

def create_known_value_classifier():
    """
    Bilinen Değer Sınıflandırma Modeli Oluşturma
    """
    
    print("🚀 Bilinen Değer Sınıflandırma Modeli Oluşturuluyor...")
    
    # 1. Autoencoder modelini yükle
    from src.models.optimized_autoencoder import OptimizedAutoencoder
    
    autoencoder_path = "results/optimized_autoencoder_curiosity_extended.pth"
    if not os.path.exists(autoencoder_path):
        print(f"❌ Autoencoder model bulunamadı: {autoencoder_path}")
        return None
    
    autoencoder = OptimizedAutoencoder(input_channels=3, latent_dim=1024)
    checkpoint = torch.load(autoencoder_path, map_location='cpu')
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    autoencoder.eval()
    
    # 2. Veri setini oluştur
    dataset = KnownValueDataset("mars_images", autoencoder)
    print(f"📊 Veri seti oluşturuldu: {len(dataset)} örnek")
    
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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 5. Model oluştur
    num_classes = 5  # 0-4 arası değer kategorileri
    classifier = KnownValueClassifier(num_classes=num_classes, feature_dim=1024)
    
    # 6. Eğitim parametreleri
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = classifier.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 7. Eğitim döngüsü
    num_epochs = 50
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\n🎓 Eğitim başlıyor ({num_epochs} epoch)...")
    
    for epoch in range(num_epochs):
        # Eğitim
        classifier.train()
        train_loss = 0.0
        for batch in train_loader:
            latent_features = batch['latent_features'].to(device)
            labels = batch['value_label'].to(device)
            
            optimizer.zero_grad()
            outputs = classifier(latent_features)
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
                latent_features = batch['latent_features'].to(device)
                labels = batch['value_label'].to(device)
                
                outputs = classifier(latent_features)
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
            }, "results/known_value_classifier.pth")
            print(f"  ✅ Yeni en iyi model kaydedildi")
    
    # 8. Eğitim eğrilerini çiz
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Bilinen Değer Sınıflandırıcı Eğitimi')
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
    plt.savefig('results/known_value_classifier_training.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Bilinen Değer Sınıflandırıcı eğitimi tamamlandı!")
    print(f"📁 Model kaydedildi: results/known_value_classifier.pth")
    print(f"🎯 En iyi validasyon loss: {best_val_loss:.4f}")
    print(f"📊 Final doğruluk: {accuracy:.2f}%")
    
    return classifier

def test_known_value_classifier():
    """
    Bilinen Değer Sınıflandırıcısını Test Et
    """
    
    print("🧪 Bilinen Değer Sınıflandırıcısı Test Ediliyor...")
    
    # Model yükle
    classifier_path = "results/known_value_classifier.pth"
    if not os.path.exists(classifier_path):
        print(f"❌ Sınıflandırıcı model bulunamadı: {classifier_path}")
        return
    
    # Autoencoder yükle
    from src.models.optimized_autoencoder import OptimizedAutoencoder
    
    autoencoder_path = "results/optimized_autoencoder_curiosity_extended.pth"
    autoencoder = OptimizedAutoencoder(input_channels=3, latent_dim=1024)
    checkpoint = torch.load(autoencoder_path, map_location='cpu')
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    autoencoder.eval()
    
    # Sınıflandırıcı yükle
    classifier = KnownValueClassifier(num_classes=5, feature_dim=1024)
    checkpoint = torch.load(classifier_path, map_location='cpu')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()
    
    # Test veri seti oluştur
    test_dataset = KnownValueDataset("mars_images", autoencoder)
    
    # Test et
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = classifier.to(device)
    
    all_predictions = []
    all_labels = []
    all_categories = []
    
    with torch.no_grad():
        for i in range(min(100, len(test_dataset))):  # İlk 100 örnek
            sample = test_dataset[i]
            latent_features = sample['latent_features'].unsqueeze(0).to(device)
            label = sample['value_label']
            category = sample['category']
            
            output = classifier(latent_features)
            _, predicted = torch.max(output, 1)
            
            all_predictions.append(predicted.item())
            all_labels.append(label.item())
            all_categories.append(category)
    
    # Sonuçları analiz et
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\n📊 Sınıflandırma Raporu:")
    value_names = {0: "Değersiz", 1: "Düşük", 2: "Orta", 3: "Orta-Yüksek", 4: "Yüksek"}
    print(classification_report(all_labels, all_predictions, 
                               target_names=[value_names[i] for i in range(5)]))
    
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
    print("🚀 Bilinen Değer Sınıflandırma Modeli Projesi")
    print("=" * 50)
    
    # 1. Sınıflandırıcı oluştur ve eğit
    classifier = create_known_value_classifier()
    
    if classifier:
        # 2. Test et
        test_known_value_classifier()
    
    print("\n✅ Proje tamamlandı!") 