import argparse
from pathlib import Path
import csv
from typing import List, Dict

from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_summary(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def build_grid(image_paths: List[Path], cols: int, cell_w: int = 480) -> Image.Image:
    # Load images; resize maintaining aspect ratio to width=cell_w
    imgs: List[Image.Image] = []
    for p in image_paths:
        if not p.exists():
            continue
        im = Image.open(p).convert('RGB')
        w, h = im.size
        new_h = int(h * (cell_w / max(1, w)))
        im = im.resize((cell_w, new_h), Image.BILINEAR)
        imgs.append(im)
    if not imgs:
        return Image.new('RGB', (cell_w, cell_w), (0, 0, 0))
    # Grid
    rows = (len(imgs) + cols - 1) // cols
    heights = [max(imgs[i*cols + j].size[1] if i*cols + j < len(imgs) else 0 for j in range(cols)) for i in range(rows)]
    W = cols * cell_w
    H = sum(heights)
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    y = 0
    k = 0
    for i in range(rows):
        x = 0
        for j in range(cols):
            if k >= len(imgs):
                break
            canvas.paste(imgs[k], (x, y))
            x += cell_w
            k += 1
        y += heights[i]
    return canvas


def save_corr_heatmap(rows: List[Dict[str, str]], out_path: Path) -> None:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    keys = ['curiosity', 'anomaly_mse', 'depth_variance', 'roughness', 'known_value_score']
    data = {k: [] for k in keys}
    for r in rows:
        for k in keys:
            try:
                data[k].append(float(r.get(k, 0.0)))
            except Exception:
                data[k].append(0.0)
    df = pd.DataFrame(data)
    corr = df.corr(method='pearson')
    plt.figure(figsize=(4.8, 4.2))
    im = plt.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)
    plt.xticks(range(len(keys)), keys, rotation=45, ha='right')
    plt.yticks(range(len(keys)), keys)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('Korelasyon Isı Haritası')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    p = argparse.ArgumentParser(description='Curiosity Top/Bottom grid ve korelasyon figürleri')
    p.add_argument('--fig_dir', type=str, default='results/paper_figs')
    p.add_argument('--top_k', type=int, default=9)
    p.add_argument('--cols', type=int, default=3)
    args = p.parse_args()

    fig_dir = Path(args.fig_dir)
    rows = read_summary(fig_dir / 'summary.csv')
    # sort by curiosity desc
    rows_sorted = sorted(rows, key=lambda r: float(r.get('curiosity', 0.0)), reverse=True)
    top = rows_sorted[:args.top_k]
    bottom = rows_sorted[-args.top_k:][::-1] if len(rows_sorted) >= args.top_k else rows_sorted[::-1]

    def fig_path_from_row(r: Dict[str, str]) -> Path:
        stem = Path(r['path']).stem
        return fig_dir / f'{stem}_fig.png'

    top_paths = [fig_path_from_row(r) for r in top]
    bot_paths = [fig_path_from_row(r) for r in bottom]

    top_grid = build_grid(top_paths, cols=max(1, int(args.cols)))
    bot_grid = build_grid(bot_paths, cols=max(1, int(args.cols)))

    top_grid.save(fig_dir / 'topk_grid.png')
    bot_grid.save(fig_dir / 'bottomk_grid.png')

    save_corr_heatmap(rows, fig_dir / 'corr_heatmap.png')
    print(f'Grid ve ısı haritası üretildi: {fig_dir}')


if __name__ == '__main__':
    main()


