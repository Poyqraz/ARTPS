#!/usr/bin/env python3
"""
RunwayML Gen-3 Benzeri ARTPS Text-to-Video Animasyonu
Local olarak çalışır, yüksek kaliteli video üretir
"""

import os
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

class RunwayMLGen3ARTPS:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Kullanılan cihaz: {self.device}")
        
        # Model yükleme
        self.load_models()
        
    def load_models(self):
        """RunwayML Gen-3 benzeri modelleri yükle"""
        print("📥 RunwayML Gen-3 benzeri modeller yükleniyor...")
        
        try:
            # Text-to-Video modeli (daha hafif versiyon)
            self.pipe = DiffusionPipeline.from_pretrained(
                "damo-vilab/text-to-video-ms-1.7b",
                torch_dtype=torch.float16,
                variant="fp16"
            )
            
            if self.device == "cuda":
                self.pipe = self.pipe.to(self.device)
                self.pipe.enable_model_cpu_offload()
            
            print("✅ Text-to-Video modeli başarıyla yüklendi!")
            
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            print("🔄 Alternatif model deneniyor...")
            self.load_alternative_model()
    
    def load_alternative_model(self):
        """Alternatif model yükle"""
        try:
            # Daha küçük model
            self.pipe = DiffusionPipeline.from_pretrained(
                "damo-vilab/text-to-video-ms-1.7b",
                torch_dtype=torch.float32,  # CPU için
                variant="fp32"
            )
            
            if self.device == "cuda":
                self.pipe = self.pipe.to(self.device)
            
            print("✅ Alternatif model yüklendi!")
            
        except Exception as e:
            print(f"❌ Alternatif model hatası: {e}")
            print("🎬 Matplotlib animasyonu oluşturuluyor...")
            self.create_matplotlib_animation()
    
    def create_text_to_video(self, prompt, filename, duration=4):
        """Text-to-video oluştur"""
        print(f"🎬 Video oluşturuluyor: {prompt[:50]}...")
        
        try:
            # Video üretimi
            video_frames = self.pipe(
                prompt,
                num_inference_steps=20,
                num_frames=duration * 8,  # 8 FPS
                height=512,
                width=512
            ).frames[0]
            
            # Video olarak kaydet
            export_to_video(video_frames, f"{filename}.mp4", fps=8)
            print(f"✅ Video kaydedildi: {filename}.mp4")
            
            return True
            
        except Exception as e:
            print(f"❌ Video oluşturma hatası: {e}")
            return False
    
    def create_artps_animation_sequence(self):
        """ARTPS animasyon sekanslarını oluştur"""
        print("🎬 ARTPS Animasyon Sekansları Oluşturuluyor...")
        
        # Sekans tanımları (20 saniye toplam)
        sequences = [
            {
                "name": "intro",
                "prompt": "Mars surface with Curiosity rover, red soil, scientific exploration, camera slowly zooming up, high quality, detailed, cinematic lighting",
                "duration": 4
            },
            {
                "name": "image_processing", 
                "prompt": "Digital image processing effects on Mars image, fog removal algorithm working, image becoming clearer, photometric correction, rover and surface separation becoming distinct, scientific visualization",
                "duration": 4
            },
            {
                "name": "depth_analysis",
                "prompt": "Depth map forming on image, 3D topographic data visualization, discontinuity points marked in red, depth analysis algorithm, Mars terrain elevation data, scientific mapping",
                "duration": 4
            },
            {
                "name": "anomaly_detection",
                "prompt": "Anomaly detection algorithm working, suspicious regions framed with green boxes, autoencoder and PADIM models working in parallel, machine learning visualization, scientific discovery",
                "duration": 4
            },
            {
                "name": "hybrid_fusion",
                "prompt": "Multi-component data fusion process, depth texture and anomaly data combining, curiosity score calculation, scientific decision making, autonomous exploration system",
                "duration": 4
            }
        ]
        
        created_videos = []
        
        for seq in sequences:
            print(f"\n🎬 {seq['name'].upper()} sekansı oluşturuluyor...")
            
            if self.create_text_to_video(seq['prompt'], f"artps_{seq['name']}", seq['duration']):
                created_videos.append(f"artps_{seq['name']}.mp4")
            else:
                print(f"⚠️ {seq['name']} sekansı atlandı")
        
        return created_videos
    
    def create_matplotlib_animation(self):
        """Matplotlib ile teknik süreç animasyonu"""
        print("🎨 Matplotlib animasyonu oluşturuluyor...")
        
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_title("ARTPS: Otonom Bilimsel Keşif Sistemi", 
                     fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel("Sistem İşlem Akışı", fontsize=14)
        ax.set_ylabel("Veri İşleme Seviyesi", fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Süreç kutuları
        processes = [
            {"name": "Girdi\nİyileştirme", "x": 15, "y": 85, "color": "#4CAF50", "desc": "Sis giderme + Fotometrik düzenleme"},
            {"name": "Derinlik\nAnalizi", "x": 35, "y": 85, "color": "#2196F3", "desc": "DPT Model + Süreksizlik tespiti"},
            {"name": "Anomali\nTespiti", "x": 55, "y": 85, "color": "#FF9800", "desc": "Autoencoder + PADIM"},
            {"name": "Hibrit\nFüzyon", "x": 75, "y": 85, "color": "#9C27B0", "desc": "Çok-bileşenli veri birleştirme"},
            {"name": "İlginçlik\nPuanı", "x": 45, "y": 65, "color": "#F44336", "desc": "Curiosity Score hesaplama"},
            {"name": "Bilimsel\nKarar", "x": 45, "y": 45, "color": "#795548", "desc": "Otonom keşif kararı"}
        ]
        
        self.boxes = []
        self.texts = []
        self.descriptions = []
        
        for proc in processes:
            # Ana kutu
            box = patches.Rectangle((proc["x"]-12, proc["y"]-8), 24, 16, 
                                  linewidth=2, edgecolor='black', facecolor=proc["color"], alpha=0.8)
            ax.add_patch(box)
            self.boxes.append(box)
            
            # İsim
            text = ax.text(proc["x"], proc["y"], proc["name"], 
                          ha='center', va='center', fontweight='bold', fontsize=10)
            self.texts.append(text)
            
            # Açıklama
            desc = ax.text(proc["x"], proc["y"]-15, proc["desc"], 
                          ha='center', va='center', fontsize=8, style='italic')
            self.descriptions.append(desc)
            desc.set_alpha(0)
        
        # Oklar
        self.create_arrows()
        
        # Mars sahnesi
        self.create_mars_scene()
        
        # Animasyon
        anim = animation.FuncAnimation(fig, self.animate, frames=120, 
                                     interval=150, blit=False, repeat=True)
        
        # Kaydet
        print("📹 Animasyon kaydediliyor...")
        anim.save('artps_runwayml_style.gif', writer='pillow', fps=7, dpi=100)
        print("✅ RunwayML tarzı animasyon kaydedildi: artps_runwayml_style.gif")
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def create_arrows(self):
        """Süreç oklarını oluştur"""
        self.arrows = []
        
        # Yatay oklar (üst sıra)
        for i in range(3):
            arrow = ax.annotate("", 
                               xy=(processes[i+1]["x"]-12, processes[i+1]["y"]),
                               xytext=(processes[i]["x"]+12, processes[i]["y"]),
                               arrowprops=dict(arrowstyle="->", lw=3, color='red', alpha=0.7))
            self.arrows.append(arrow)
        
        # Dikey oklar
        arrow1 = ax.annotate("", 
                             xy=(processes[4]["x"], processes[4]["y"]+8),
                             xytext=(processes[3]["x"], processes[3]["y"]-8),
                             arrowprops=dict(arrowstyle="->", lw=3, color='red', alpha=0.7))
        self.arrows.append(arrow1)
        
        arrow2 = ax.annotate("", 
                             xy=(processes[5]["x"], processes[5]["y"]+8),
                             xytext=(processes[4]["x"], processes[4]["y"]-8),
                             arrowprops=dict(arrowstyle="->", lw=3, color='red', alpha=0.7))
        self.arrows.append(arrow2)
    
    def create_mars_scene(self):
        """Mars sahnesi ekle"""
        # Mars yüzeyi simülasyonu
        x = np.linspace(0, 100, 100)
        y = 10 + 5 * np.sin(x/10) + 2 * np.random.randn(100)
        
        ax.plot(x, y, color='#8B4513', linewidth=3, alpha=0.8, label='Mars Yüzeyi')
        ax.fill_between(x, y, 0, color='#CD853F', alpha=0.3)
        
        # Rover simgesi
        rover = patches.Rectangle((20, 12), 8, 4, linewidth=2, edgecolor='black', 
                                facecolor='#666666', alpha=0.9)
        ax.add_patch(rover)
        ax.text(24, 14, "ROVER", ha='center', va='center', fontweight='bold', color='white')
        
        # Curiosity Score göstergesi
        self.score_bar = ax.barh(25, 0, height=3, color='#4CAF50', alpha=0.7)
        ax.text(50, 26.5, "Curiosity Score: 0.0", ha='center', va='center', 
                fontweight='bold', fontsize=12)
    
    def animate(self, frame):
        """Animasyon fonksiyonu"""
        # Süreç aktivasyonu
        for i, box in enumerate(self.boxes):
            if frame >= i * 15:  # Her süreç 15 frame'de aktif
                box.set_alpha(1.0)
                box.set_linewidth(3)
                self.texts[i].set_color('black')
                if i < len(self.descriptions):
                    self.descriptions[i].set_alpha(0.8)
            else:
                box.set_alpha(0.3)
                box.set_linewidth(1)
                self.texts[i].set_color('gray')
                if i < len(self.descriptions):
                    self.descriptions[i].set_alpha(0)
        
        # Curiosity Score güncelleme
        if frame > 60:  # Son 40 frame'de score artışı
            score = min(1.0, (frame - 60) / 40)
            self.score_bar[0].set_width(score * 100)
            ax.texts[-1].set_text(f"Curiosity Score: {score:.2f}")
        
        # Okların parlaması
        for arrow in self.arrows:
            if frame % 20 < 10:
                arrow.set_alpha(1.0)
            else:
                arrow.set_alpha(0.5)
        
        return self.boxes + self.texts + self.descriptions + self.arrows

def main():
    print("🚀 RunwayML Gen-3 Benzeri ARTPS Animasyon Oluşturucu")
    print("=" * 60)
    
    # RunwayML Gen-3 başlat
    gen3 = RunwayMLGen3ARTPS()
    
    # Text-to-video sekansları oluştur
    print("\n🎬 Text-to-Video sekansları oluşturuluyor...")
    videos = gen3.create_artps_animation_sequence()
    
    if videos:
        print(f"\n✅ {len(videos)} video sekansı başarıyla oluşturuldu!")
        print("\n📁 Oluşturulan dosyalar:")
        for video in videos:
            print(f"   - {video}")
        print(f"\n🎬 Toplam süre: {len(videos) * 4} saniye")
        
        # Video birleştirme önerisi
        print("\n💡 Sonraki adım: Video birleştirme")
        print("   - FFmpeg ile birleştirme")
        print("   - Adobe Premiere Pro")
        print("   - DaVinci Resolve")
        
    else:
        print("\n🎨 Matplotlib animasyonu kullanılıyor...")
        gen3.create_matplotlib_animation()

if __name__ == "__main__":
    main()

