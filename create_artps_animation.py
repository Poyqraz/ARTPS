#!/usr/bin/env python3
"""
ARTPS (Otonom Bilimsel Keşif Sistemi) Tanıtım Animasyonu
Stable Video Diffusion ile 20 saniyelik animasyon oluşturma
"""

import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

class ARTPSAnimator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Kullanılan cihaz: {self.device}")
        
        # SVD pipeline'ını yükle
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        
        if self.device == "cuda":
            self.pipe = self.pipe.to(self.device)
            self.pipe.enable_model_cpu_offload()
        
        print("SVD modeli başarıyla yüklendi!")
    
    def load_reference_images(self):
        """Proje çıktılarından referans görselleri yükle"""
        self.images = {}
        
        # Görsel iyileştirme süreci
        if os.path.exists("results/paper_figs/0100ML0004900050102961C00_DXXX_fig.png"):
            self.images['input_enhancement'] = Image.open("results/paper_figs/0100ML0004900050102961C00_DXXX_fig.png")
        
        # Derinlik analizi
        if os.path.exists("results/paper_figs/depth_compare/0100ML0004900050102961C00_DXXX_depth_compare.png"):
            self.images['depth_analysis'] = Image.open("results/paper_figs/depth_compare/0100ML0004900050102961C00_DXXX_depth_compare.png")
        
        # Anomali tespiti
        if os.path.exists("results/detection_overlays/0735MR0031500040403079E01_DXXX_combined.png"):
            self.images['anomaly_detection'] = Image.open("results/detection_overlays/0735MR0031500040403079E01_DXXX_combined.png")
        
        # Hibrit füzyon
        if os.path.exists("results/paper_figs/ae_diff_norm_hills_combined.png"):
            self.images['hybrid_fusion'] = Image.open("results/paper_figs/ae_diff_norm_hills_combined.png")
        
        print(f"Yüklenen görsel sayısı: {len(self.images)}")
        return self.images
    
    def create_animation_sequence(self):
        """20 saniyelik animasyon sekansı oluştur"""
        print("Animasyon sekansı oluşturuluyor...")
        
        # 1. Giriş (3 saniye) - Mars yüzeyi + rover
        print("1. Giriş sekansı...")
        if 'input_enhancement' in self.images:
            video_frames = self.pipe(
                self.images['input_enhancement'],
                num_frames=8,  # 3 saniye için
                fps=3,
                motion_bucket_id=127,
                noise_aug_strength=0.1
            ).frames[0]
            
            # Video olarak kaydet
            export_to_video(video_frames, "artps_intro.mp4", fps=3)
            print("✓ Giriş sekansı kaydedildi: artps_intro.mp4")
        
        # 2. Derinlik analizi (5 saniye)
        print("2. Derinlik analizi sekansı...")
        if 'depth_analysis' in self.images:
            video_frames = self.pipe(
                self.images['depth_analysis'],
                num_frames=15,  # 5 saniye için
                fps=3,
                motion_bucket_id=127,
                noise_aug_strength=0.05
            ).frames[0]
            
            export_to_video(video_frames, "artps_depth.mp4", fps=3)
            print("✓ Derinlik analizi kaydedildi: artps_depth.mp4")
        
        # 3. Anomali tespiti (6 saniye)
        print("3. Anomali tespiti sekansı...")
        if 'anomaly_detection' in self.images:
            video_frames = self.pipe(
                self.images['anomaly_detection'],
                num_frames=18,  # 6 saniye için
                fps=3,
                motion_bucket_id=127,
                noise_aug_strength=0.08
            ).frames[0]
            
            export_to_video(video_frames, "artps_anomaly.mp4", fps=3)
            print("✓ Anomali tespiti kaydedildi: artps_anomaly.mp4")
        
        # 4. Hibrit füzyon (6 saniye)
        print("4. Hibrit füzyon sekansı...")
        if 'hybrid_fusion' in self.images:
            video_frames = self.pipe(
                self.images['hybrid_fusion'],
                num_frames=18,  # 6 saniye için
                fps=3,
                motion_bucket_id=127,
                noise_aug_strength=0.06
            ).frames[0]
            
            export_to_video(video_frames, "artps_fusion.mp4", fps=3)
            print("✓ Hibrit füzyon kaydedildi: artps_fusion.mp4")
    
    def create_matplotlib_animation(self):
        """Matplotlib ile teknik süreç animasyonu"""
        print("Matplotlib animasyonu oluşturuluyor...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_title("ARTPS: Otonom Bilimsel Keşif Sistemi", fontsize=16, fontweight='bold')
        
        # Süreç kutuları
        processes = [
            {"name": "Girdi İyileştirme", "x": 20, "y": 80, "color": "lightblue"},
            {"name": "Derinlik Analizi", "x": 50, "y": 80, "color": "lightgreen"},
            {"name": "Anomali Tespiti", "x": 80, "y": 80, "color": "lightcoral"},
            {"name": "Hibrit Füzyon", "x": 50, "y": 50, "color": "gold"},
            {"name": "İlginçlik Puanı", "x": 50, "y": 20, "color": "purple"}
        ]
        
        boxes = []
        for proc in processes:
            box = patches.Rectangle((proc["x"]-15, proc["y"]-10), 30, 20, 
                                  linewidth=2, edgecolor='black', facecolor=proc["color"])
            ax.add_patch(box)
            ax.text(proc["x"], proc["y"], proc["name"], ha='center', va='center', fontweight='bold')
            boxes.append(box)
        
        # Oklar
        arrows = []
        for i in range(len(processes)-1):
            if i < 3:  # Üst sıra
                arrow = ax.annotate("", xy=(processes[i+1]["x"]-15, processes[i+1]["y"]),
                                  xytext=(processes[i]["x"]+15, processes[i]["y"]),
                                  arrowprops=dict(arrowstyle="->", lw=2, color='red'))
                arrows.append(arrow)
            elif i == 3:  # Orta sıra
                arrow = ax.annotate("", xy=(processes[i+1]["x"], processes[i+1]["y"]+10),
                                  xytext=(processes[i]["x"], processes[i]["y"]-10),
                                  arrowprops=dict(arrowstyle="->", lw=2, color='red'))
                arrows.append(arrow)
        
        def animate(frame):
            # Animasyon efekti
            for i, box in enumerate(boxes):
                if frame >= i * 20:  # Her süreç 20 frame'de aktif olsun
                    box.set_alpha(1.0)
                    box.set_linewidth(3)
                else:
                    box.set_alpha(0.3)
                    box.set_linewidth(1)
            
            return boxes + arrows
        
        anim = FuncAnimation(fig, animate, frames=100, interval=200, blit=False)
        anim.save('artps_process_animation.gif', writer='pillow', fps=5)
        print("✓ Matplotlib animasyonu kaydedildi: artps_process_animation.gif")
        
        plt.show()

def main():
    print("🚀 ARTPS Tanıtım Animasyonu Oluşturucu")
    print("=" * 50)
    
    animator = ARTPSAnimator()
    
    # Referans görselleri yükle
    images = animator.load_reference_images()
    
    if not images:
        print("❌ Referans görsel bulunamadı! Matplotlib animasyonu oluşturuluyor...")
        animator.create_matplotlib_animation()
        return
    
    # SVD animasyonları oluştur
    try:
        animator.create_animation_sequence()
        print("\n✅ Tüm animasyon sekansları başarıyla oluşturuldu!")
        print("\n📁 Oluşturulan dosyalar:")
        print("   - artps_intro.mp4 (3 saniye)")
        print("   - artps_depth.mp4 (5 saniye)")
        print("   - artps_anomaly.mp4 (6 saniye)")
        print("   - artps_fusion.mp4 (6 saniye)")
        print("\n🎬 Toplam süre: 20 saniye")
        
    except Exception as e:
        print(f"❌ SVD animasyon hatası: {e}")
        print("Matplotlib animasyonu oluşturuluyor...")
        animator.create_matplotlib_animation()

if __name__ == "__main__":
    main()

