#!/usr/bin/env python3
"""
Hızlı test scripti
"""

import torch
import sys
import os

# Proje modüllerini import et
sys.path.append('src')

try:
    from models.simple_autoencoder import SimpleAutoencoder
    print("✅ SimpleAutoencoder import edildi")
    
    # Model oluştur
    model = SimpleAutoencoder(input_channels=3, latent_dim=16384)
    print(f"✅ Model oluşturuldu. Parametre sayısı: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test görüntüsü
    test_image = torch.randn(1, 3, 128, 128)
    print(f"✅ Test görüntüsü oluşturuldu: {test_image.shape}")
    
    # Forward pass
    with torch.no_grad():
        reconstructed, latent = model(test_image)
    
    print(f"✅ Forward pass başarılı!")
    print(f"   Giriş: {test_image.shape}")
    print(f"   Çıkış: {reconstructed.shape}")
    print(f"   Latent: {latent.shape}")
    
    print("\n🎉 Tüm testler başarılı!")
    
except Exception as e:
    print(f"❌ Hata: {str(e)}")
    import traceback
    traceback.print_exc() 