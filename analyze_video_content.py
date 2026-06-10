#!/usr/bin/env python3
"""
Video İçerik Analizi Betiği
Video'dan kareler çıkarıp içeriği analiz eder
"""

import os
import subprocess
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def extract_frames(video_path, output_dir, frame_interval=2):
    """Video'dan belirli aralıklarla kareler çıkarır"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # FFmpeg ile kareleri çıkar
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', f'select=not(mod(n\\,{frame_interval}))',
        '-vsync', 'vfr',
        '-q:v', '2',
        os.path.join(output_dir, f'{video_name}_frame_%03d.jpg')
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✅ {video_path} için kareler çıkarıldı: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Hata: {e}")
        return False
    
    return True

def analyze_frame_content(frame_path):
    """Kare içeriğini analiz eder"""
    try:
        img = cv2.imread(frame_path)
        if img is None:
            return None
        
        # Temel bilgiler
        height, width = img.shape[:2]
        
        # Renk analizi
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        avg_hue = np.mean(hsv[:,:,0])
        avg_sat = np.mean(hsv[:,:,1])
        avg_val = np.mean(hsv[:,:,2])
        
        # Kenar tespiti
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        
        return {
            'size': (width, height),
            'avg_hue': avg_hue,
            'avg_sat': avg_sat,
            'avg_val': avg_val,
            'edge_density': edge_density,
            'brightness': np.mean(gray)
        }
    except Exception as e:
        print(f"❌ Kare analizi hatası: {e}")
        return None

def main():
    """Ana fonksiyon"""
    video_dir = r"C:\Users\cancor\Downloads\Video"
    output_dir = "video_frames"
    
    # Video dosyalarını listele
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    
    print("🎬 Video İçerik Analizi Başlıyor...")
    print(f"📁 Bulunan video sayısı: {len(video_files)}")
    
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        print(f"\n🔍 Analiz ediliyor: {video_file}")
        
        # Kareleri çıkar
        if extract_frames(video_path, output_dir):
            # Çıkarılan kareleri analiz et
            frame_files = [f for f in os.listdir(output_dir) if video_file.split('.')[0] in f]
            
            print(f"   📸 Çıkarılan kare sayısı: {len(frame_files)}")
            
            for frame_file in frame_files[:3]:  # İlk 3 kareyi analiz et
                frame_path = os.path.join(output_dir, frame_file)
                analysis = analyze_frame_content(frame_path)
                
                if analysis:
                    print(f"   📊 {frame_file}:")
                    print(f"      Boyut: {analysis['size']}")
                    print(f"      Ortalama Parlaklık: {analysis['brightness']:.1f}")
                    print(f"      Kenar Yoğunluğu: {analysis['edge_density']:.4f}")
                    print(f"      Ortalama Renk Tonu: {analysis['avg_hue']:.1f}°")
    
    print(f"\n✅ Analiz tamamlandı! Kareler '{output_dir}' klasöründe.")

if __name__ == "__main__":
    main()

