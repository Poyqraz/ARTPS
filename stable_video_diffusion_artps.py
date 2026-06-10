#!/usr/bin/env python3
"""
Stable Video Diffusion (SVD) ile ARTPS Tanıtım Animasyonu (Local)
- Image-to-Video: Her adım için referans kareye teknik HUD/overlay eklenir
- 5 sahne × ~4 sn ≈ 20 sn toplam
- Çıktılar: svd_step_*.mp4 ve birleştirilmiş artps_svd_20s.mp4
"""

import os
import sys
import cv2
import math
import time
import torch
import random
import numpy as np
from typing import List, Tuple
from PIL import Image

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video

# Opsiyonel: klip birleştirme
try:
	from moviepy.editor import VideoFileClip, concatenate_videoclips
	HAS_MOVIEPY = True
except Exception:
	HAS_MOVIEPY = False

# --------------------------- Yardımcılar ---------------------------

def ensure_dir(path: str) -> None:
	if not os.path.isdir(path):
		os.makedirs(path, exist_ok=True)

def load_base_image() -> str:
	"""Projedeki mevcut görsellerden biriyle başlayalım."""
	candidates = [
		"results/paper_images/curiosity_0000_Sol_958__Mast_Camera_(Mastcam).jpg",
		"results/paper_images/0735MR0031500040403079E01_DXXX.jpg",
		"mars_images/0735MR0031500040403079E01_DXXX.jpg",
		"data/nasa_photos/sol_01000/1000ML0044631260305223E03_DXXX.jpg",
		"docs/docs/figures/curiosity_0000_Sol_958__Mast_Camera_(Mastcam).jpg",
	]
	for p in candidates:
		if os.path.isfile(p):
			return p
	raise FileNotFoundError("Uygun bir başlangıç görseli bulunamadı. Lütfen bir referans görsel ekleyin.")

def resize_keep_aspect(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
	"""Görüntüyü 16:9 (1024x576 gibi) boyutuna, kenar doldurmalı ölçekle."""
	h, w = img.shape[:2]
	scale = min(target_w / w, target_h / h)
	new_w = int(w * scale)
	new_h = int(h * scale)
	resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
	canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
	off_x = (target_w - new_w) // 2
	off_y = (target_h - new_h) // 2
	canvas[off_y:off_y+new_h, off_x:off_x+new_w] = resized
	return canvas

def draw_text(img: np.ndarray, text: str, org: Tuple[int, int], scale: float = 0.8, color=(255,255,255)) -> None:
	cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 3, cv2.LINE_AA)
	cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

def overlay_alpha(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
	return (base.astype(np.float32) * (1.0 - alpha) + overlay.astype(np.float32) * alpha).astype(np.uint8)

def make_depth_overlay(img: np.ndarray) -> np.ndarray:
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray, 80, 160)
	# Derinlik benzeri yarı saydam ısı haritası (yaklaşık görselleştirme)
	edges_col = cv2.applyColorMap(edges, cv2.COLORMAP_TWILIGHT)
	return overlay_alpha(img, edges_col, 0.30)

def make_blue_contours(img: np.ndarray, thickness: int = 1, alpha: float = 0.35) -> np.ndarray:
	"""Mavi kontur çizgilerini tüm görüntüye düşük alfa ile uygular."""
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray, 100, 200)
	contour_layer = img.copy()
	contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(contour_layer, contours, -1, (255, 120, 30), thickness)  # BGR: hafif turuncu-mavi karışımı
	return overlay_alpha(img, contour_layer, alpha)

def make_boxes_overlay(img: np.ndarray, num_boxes: int = 4, with_labels: bool = True) -> np.ndarray:
	h, w = img.shape[:2]
	out = img.copy()
	random.seed(42)
	for i in range(num_boxes):
		bw = random.randint(w//10, w//5)
		bh = random.randint(h//12, h//6)
		x = random.randint(20, max(21, w - bw - 20))
		y = random.randint(40, max(41, h - bh - 40))
		cv2.rectangle(out, (x, y), (x+bw, y+bh), (50, 230, 80), 2)
		if with_labels:
			draw_text(out, f"Rock ID: R-{i+1:02d}", (x+6, y+22), 0.5, (50, 230, 80))
	return out

def make_score_overlay(img: np.ndarray, score: float = 0.86) -> np.ndarray:
	out = img.copy()
	h, w = out.shape[:2]
	bar_w = int(w * 0.6)
	bar_h = 18
	x0 = (w - bar_w)//2
	y0 = int(h * 0.88)
	cv2.rectangle(out, (x0, y0), (x0+bar_w, y0+bar_h), (180,180,180), 2)
	fill_w = int(bar_w * max(0.0, min(1.0, score)))
	cv2.rectangle(out, (x0+2, y0+2), (x0+2+fill_w, y0+bar_h-2), (76,175,80), -1)
	# Sadece skor etiketi; marka/metin yok
	draw_text(out, f"Curiosity Score: {score:.2f}", (x0, y0-8), 0.6, (220,220,220))
	return out

def make_hero_detection_overlay(img: np.ndarray, score: float = 0.86) -> np.ndarray:
	"""Kayaç tespitini kahraman sahne olarak vurgulayan overlay bileşimi."""
	base = make_depth_overlay(img)
	base = make_blue_contours(base, thickness=1, alpha=0.30)
	# Merkezde büyük bir kutu (hedef kayaç)
	h, w = base.shape[:2]
	bw = int(w * 0.36)
	bh = int(h * 0.30)
	x = (w - bw)//2
	y = int(h * 0.50) - bh//2
	out = base.copy()
	cv2.rectangle(out, (x, y), (x+bw, y+bh), (50, 230, 80), 2)
	# Etiketler (küçük, okunabilir)
	draw_text(out, "Rock ID: R-07", (x+6, y+22), 0.6, (50, 230, 80))
	draw_text(out, "Anomaly Score: 0.82", (x+6, y+44), 0.6, (80, 200, 255))
	draw_text(out, "Curiosity Score: 0.88", (x+6, y+66), 0.6, (255, 220, 120))
	# Skor barı altta
	out = make_score_overlay(out, score)
	return out

# --------------------------- SVD Üretim ---------------------------

class SVDAnimator:
	def __init__(self, width: int = 1024, height: int = 576, fps: int = 8):
		self.width = width
		self.height = height
		self.fps = fps
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		print(f"Cihaz: {self.device}")

		self.pipe = StableVideoDiffusionPipeline.from_pretrained(
			"stabilityai/stable-video-diffusion-img2vid-xt",
			torch_dtype=(torch.float16 if self.device == "cuda" else torch.float32),
		)
		if self.device == "cuda":
			self.pipe.enable_model_cpu_offload()
			self.pipe.to(self.device)

	def image_to_video(self, image_bgr: np.ndarray, out_path: str, seconds: int = 4, motion_bucket_id: int = 127, noise_aug_strength: float = 0.1, seed: int = 42) -> str:
		pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
		gen = torch.Generator(device=self.device).manual_seed(seed)
		num_frames = max(8, int(self.fps * seconds))

		frames = self.pipe(
			pil_img,
			num_frames=num_frames,
			height=self.height,
			width=self.width,
			motion_bucket_id=motion_bucket_id,
			noise_aug_strength=noise_aug_strength,
			generator=gen,
		).frames[0]

		export_to_video(frames, out_path, fps=self.fps)
		return out_path

# --------------------------- Ana Akış ---------------------------

def main():
	print("Stable Video Diffusion ile ARTPS animasyonu başlıyor...")
	ensure_dir("svd_outputs")
	base_path = load_base_image()
	print(f"Referans görsel: {base_path}")

	# Görseli yükle ve SVD boyutuna getir (1024x576)
	img = cv2.imread(base_path, cv2.IMREAD_COLOR)
	if img is None:
		raise RuntimeError(f"Görsel yüklenemedi: {base_path}")
	img = resize_keep_aspect(img, 1024, 576)

	# Adım varyantları (overlay)
	intro = img.copy()  # Ekranda başlık/metin yok

	enh = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
	enh = overlay_alpha(enh, img, 0.2)

	depth = make_depth_overlay(img)

	# Anomali tespiti (kahraman sahne için ayrı kompozit)
	anom = make_boxes_overlay(make_blue_contours(depth, thickness=1, alpha=0.25), num_boxes=4, with_labels=True)
	anom_hero = make_hero_detection_overlay(img, 0.88)

	fusion = make_score_overlay(anom, 0.86)

	steps = [
		("svd_outputs/svd_step1_intro.mp4", intro, 4),
		("svd_outputs/svd_step2_enhance.mp4", enh, 4),
		("svd_outputs/svd_step3_depth.mp4", depth, 4),
		("svd_outputs/svd_step4_anomaly_hero.mp4", anom_hero, 6),  # Kayaç tespiti kahraman sahne (3-4sn+)
		("svd_outputs/svd_step5_anomaly_multi.mp4", anom, 4),
		("svd_outputs/svd_step6_fusion.mp4", fusion, 4),
	]

	anim = SVDAnimator(width=1024, height=576, fps=8)

	made = []
	for out_path, frame, secs in steps:
		print(f"Üretiliyor: {out_path} ({secs}s)")
		try:
			anim.image_to_video(frame, out_path, seconds=secs, motion_bucket_id=127, noise_aug_strength=0.1, seed=42)
			made.append(out_path)
		except Exception as e:
			print(f"Hata: {out_path} → {e}")

	# Birleştirme
	final_out = "svd_outputs/artps_svd_final.mp4"
	if HAS_MOVIEPY and len(made) >= 2:
		print("Klipler birleştiriliyor (moviepy)...")
		clips = [VideoFileClip(p) for p in made]
		concat = concatenate_videoclips(clips, method="compose")
		concat.write_videofile(final_out, fps=24, codec="libx264", audio=False, verbose=False, logger=None)
		for c in clips:
			c.close()
		print(f"Hazır: {final_out}")
	else:
		print("Birleştirme atlandı. FFmpeg ile manuel birleştirebilirsiniz:")
		print("ffmpeg -y -i \"concat:svd_outputs/svd_step1_intro.mp4|svd_outputs/svd_step2_enhance.mp4|svd_outputs/svd_step3_depth.mp4|svd_outputs/svd_step4_anomaly_hero.mp4|svd_outputs/svd_step5_anomaly_multi.mp4|svd_outputs/svd_step6_fusion.mp4\" -c copy svd_outputs/artps_svd_final.mp4")

	print("Bitti.")

if __name__ == "__main__":
	main()
