#!/usr/bin/env python3
"""
ARTPS Basit Animasyon Oluşturucu
GPU gerektirmez, matplotlib ile çalışır
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import os
import cv2

class SimpleARTPSAnimator:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(16, 10))
        self.setup_plot()
        
    def setup_plot(self):
        """Ana plot ayarları"""
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.set_title("ARTPS: Otonom Bilimsel Keşif Sistemi", 
                         fontsize=20, fontweight='bold', pad=20)
        self.ax.set_xlabel("Sistem İşlem Akışı", fontsize=14)
        self.ax.set_ylabel("Veri İşleme Seviyesi", fontsize=14)
        self.ax.grid(True, alpha=0.3)
        
    def create_process_boxes(self):
        """Süreç kutularını oluştur"""
        self.processes = [
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
        
        for proc in self.processes:
            # Ana kutu
            box = patches.Rectangle((proc["x"]-12, proc["y"]-8), 24, 16, 
                                  linewidth=2, edgecolor='black', facecolor=proc["color"], alpha=0.8)
            self.ax.add_patch(box)
            self.boxes.append(box)
            
            # İsim
            text = self.ax.text(proc["x"], proc["y"], proc["name"], 
                              ha='center', va='center', fontweight='bold', fontsize=10)
            self.texts.append(text)
            
            # Açıklama
            desc = self.ax.text(proc["x"], proc["y"]-15, proc["desc"], 
                              ha='center', va='center', fontsize=8, style='italic')
            self.descriptions.append(desc)
            desc.set_alpha(0)
        
        # Oklar
        self.create_arrows()
        
    def create_arrows(self):
        """Süreç oklarını oluştur"""
        self.arrows = []
        
        # Yatay oklar (üst sıra)
        for i in range(3):
            arrow = self.ax.annotate("", 
                                   xy=(self.processes[i+1]["x"]-12, self.processes[i+1]["y"]),
                                   xytext=(self.processes[i]["x"]+12, self.processes[i]["y"]),
                                   arrowprops=dict(arrowstyle="->", lw=3, color='red', alpha=0.7))
            self.arrows.append(arrow)
        
        # Dikey oklar
        arrow1 = self.ax.annotate("", 
                                 xy=(self.processes[4]["x"], self.processes[4]["y"]+8),
                                 xytext=(self.processes[3]["x"], self.processes[3]["y"]-8),
                                 arrowprops=dict(arrowstyle="->", lw=3, color='red', alpha=0.7))
        self.arrows.append(arrow1)
        
        arrow2 = self.ax.annotate("", 
                                 xy=(self.processes[5]["x"], self.processes[5]["y"]+8),
                                 xytext=(self.processes[4]["x"], self.processes[4]["y"]-8),
                                 arrowprops=dict(arrowstyle="->", lw=3, color='red', alpha=0.7))
        self.arrows.append(arrow2)
        
        # Geri dönüş oku
        arrow3 = self.ax.annotate("", 
                                 xy=(self.processes[0]["x"], self.processes[0]["y"]-8),
                                 xytext=(self.processes[5]["x"]-12, self.processes[5]["y"]),
                                 arrowprops=dict(arrowstyle="->", lw=3, color='blue', alpha=0.7))
        self.arrows.append(arrow3)
    
    def create_mars_scene(self):
        """Mars sahnesi ekle"""
        # Mars yüzeyi simülasyonu
        x = np.linspace(0, 100, 100)
        y = 10 + 5 * np.sin(x/10) + 2 * np.random.randn(100)
        
        self.ax.plot(x, y, color='#8B4513', linewidth=3, alpha=0.8, label='Mars Yüzeyi')
        self.ax.fill_between(x, y, 0, color='#CD853F', alpha=0.3)
        
        # Rover simgesi
        rover = patches.Rectangle((20, 12), 8, 4, linewidth=2, edgecolor='black', 
                                facecolor='#666666', alpha=0.9)
        self.ax.add_patch(rover)
        self.ax.text(24, 14, "ROVER", ha='center', va='center', fontweight='bold', color='white')
        
        # Curiosity Score göstergesi
        self.score_bar = self.ax.barh(25, 0, height=3, color='#4CAF50', alpha=0.7)
        self.ax.text(50, 26.5, "Curiosity Score: 0.0", ha='center', va='center', 
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
            self.ax.texts[-1].set_text(f"Curiosity Score: {score:.2f}")
        
        # Okların parlaması
        for arrow in self.arrows:
            if frame % 20 < 10:
                arrow.set_alpha(1.0)
            else:
                arrow.set_alpha(0.5)
        
        return self.boxes + self.texts + self.descriptions + self.arrows
    
    def create_animation(self):
        """Ana animasyonu oluştur"""
        print("🚀 ARTPS Animasyonu oluşturuluyor...")
        
        self.create_process_boxes()
        self.create_mars_scene()
        
        # Animasyon
        anim = animation.FuncAnimation(self.fig, self.animate, frames=120, 
                                     interval=150, blit=False, repeat=True)
        
        # Kaydet
        print("📹 Animasyon kaydediliyor...")
        anim.save('artps_animation.gif', writer='pillow', fps=7, dpi=100)
        print("✅ Animasyon kaydedildi: artps_animation.gif")
        
        plt.tight_layout()
        plt.show()
        
        return anim

def main():
    print("🎬 ARTPS Basit Animasyon Oluşturucu")
    print("=" * 40)
    
    animator = SimpleARTPSAnimator()
    animation = animator.create_animation()
    
    print("\n🎉 Animasyon başarıyla oluşturuldu!")
    print("📁 Dosya: artps_animation.gif")
    print("⏱️  Süre: ~17 saniye")
    print("🔄 Tekrar: Sonsuz döngü")

if __name__ == "__main__":
    main()

