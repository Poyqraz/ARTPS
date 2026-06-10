import argparse
from pathlib import Path
import pandas as pd


TEMPLATE = """# ARTPS Makale Figür Özeti

- Toplam örnek: {n}
- Curiosity ort/Std: {c_mean:.3f} / {c_std:.3f}
- AE MSE ort/Std: {a_mean:.6f} / {a_std:.6f}
- Depth variance ort/Std: {d_mean:.6f} / {d_std:.6f}
- Roughness ort/Std: {r_mean:.6f} / {r_std:.6f}

## Dağılım ve İlişkiler

![Dataset Summary](dataset_summary.png)

![Korelasyon Isı Haritası](corr_heatmap.png)

## En İlginç Örnekler (Top-{k})

![Top Grid](topk_grid.png)

## En Düşük Curiosity (Bottom-{k})

![Bottom Grid](bottomk_grid.png)

## Anomali Tespit Örnekleri (Kutulu Overlay)

Bu bölümde, her görsel için birleşik anomali haritası üzerinden üretilen kutulu tespit overlay örnekleri verilmektedir. (Yöntem: AE farkı + derinlik kenarı + doku/gölge + PaDiM/PatchCore füzyonu.) Yaklaşım, gezgin otonomisinde hedef önceliklendirmeye yönelik literatürle uyumludur [Estlin et al., 2014] (bkz. [JPL 2014 ISAIRAS](https://ai.jpl.nasa.gov/public/documents/papers/estlin-isairas2014-automated.pdf)).

{overlay_gallery}
"""


def main():
	p = argparse.ArgumentParser(description="Makale raporu (Markdown)")
	p.add_argument('--fig_dir', type=str, default='results/paper_figs')
	p.add_argument('--top_k', type=int, default=9)
	args = p.parse_args()

	fig_dir = Path(args.fig_dir)
	df = pd.read_csv(fig_dir / 'summary.csv')
	out = fig_dir / 'paper_report.md'

	# Overlay galerisi
	over_dir = fig_dir / 'detection_overlays'
	over_imgs = sorted([p.name for p in over_dir.glob('*_det_overlay.png')]) if over_dir.exists() else []
	gallery_lines = []
	for name in over_imgs:
		gallery_lines.append(f"![{name}](detection_overlays/{name})")
	overlay_gallery = "\n\n".join(gallery_lines) if gallery_lines else "(overlay bulunamadı)"

	md = TEMPLATE.format(
		n=len(df),
		c_mean=df['curiosity'].mean(),
		c_std=df['curiosity'].std(ddof=0),
		a_mean=df['anomaly_mse'].mean(),
		a_std=df['anomaly_mse'].std(ddof=0),
		d_mean=df['depth_variance'].mean(),
		d_std=df['depth_variance'].std(ddof=0),
		r_mean=df['roughness'].mean(),
		r_std=df['roughness'].std(ddof=0),
		k=int(args.top_k),
		overlay_gallery=overlay_gallery,
	)

	out.write_text(md, encoding='utf-8')
	print(f"Rapor yazıldı: {out}")


if __name__ == '__main__':
	main()
