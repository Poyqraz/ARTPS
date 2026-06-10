import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch

from src.models.depth_estimation import MiDaSDepthEstimator


def _normalize_map(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float32)
    lo, hi = np.percentile(arr, [2, 98])
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def render_compare_fig(
    img_path: Path,
    depth_a: np.ndarray,
    depth_b: np.ndarray,
    out_dir: Path,
    label_a: str,
    label_b: str,
) -> None:
    img = Image.open(img_path).convert('RGB')
    da = _normalize_map(depth_a)
    db = _normalize_map(depth_b)
    diff = np.abs(da - db)
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(da, cmap='plasma')
    ax1.set_title(label_a)
    ax1.axis('off')
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(db, cmap='plasma')
    ax2.set_title(label_b)
    ax2.axis('off')
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(diff, cmap='inferno')
    ax3.set_title('|A-B| (norm)')
    ax3.axis('off')
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{img_path.stem}_depth_compare.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def estimate_depth(model: MiDaSDepthEstimator, img: Image.Image) -> np.ndarray:
    arr = np.array(img, dtype=np.float32) / 255.0
    depth, _ = model.estimate_depth(arr)
    return depth.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description='Derinlik modelleri karşılaştırma (DPT_Large vs DPT_Hybrid gibi)')
    p.add_argument('--images_dir', type=str, required=True)
    p.add_argument('--models', type=str, default='DPT_Large,DPT_Hybrid')
    p.add_argument('--out_dir', type=str, default='results/paper_figs/depth_compare')
    p.add_argument('--limit', type=int, default=12)
    p.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'auto' else ('cuda' if args.device == 'cuda' else 'cpu'))
    types = [t.strip() for t in args.models.split(',') if t.strip()]
    if len(types) < 2:
        raise ValueError('En az iki model tipi veriniz (örn. DPT_Large,DPT_Hybrid)')

    # İlk iki modeli kullan (ikili karşılaştırma)
    a_type, b_type = types[0], types[1]
    model_a = MiDaSDepthEstimator(model_type=a_type, device=device)
    model_b = MiDaSDepthEstimator(model_type=b_type, device=device)

    root = Path(args.images_dir)
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = [p for p in root.rglob('*') if p.suffix.lower() in exts]
    if args.limit and args.limit > 0:
        files = files[:args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, float]] = []
    for pth in files:
        img = Image.open(pth).convert('RGB')
        da = estimate_depth(model_a, img)
        db = estimate_depth(model_b, img)
        # Metrikler
        na = _normalize_map(da).flatten()
        nb = _normalize_map(db).flatten()
        corr = float(np.corrcoef(na, nb)[0, 1]) if na.size and nb.size else 0.0
        mad = float(np.mean(np.abs(na - nb))) if na.size and nb.size else 0.0
        rows.append({'path': str(pth), 'corr': corr, 'mad': mad})
        render_compare_fig(pth, da, db, out_dir, a_type, b_type)

    # Özet grafikleri
    if rows:
        corr_vals = [r['corr'] for r in rows]
        mad_vals = [r['mad'] for r in rows]
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.hist(corr_vals, bins=15, color='#3f51b5', alpha=0.85)
        plt.title('Pearson Corr(A,B)')
        plt.subplot(1, 2, 2)
        plt.hist(mad_vals, bins=15, color='#ff9800', alpha=0.85)
        plt.title('Mean |A-B| (norm)')
        plt.tight_layout()
        plt.savefig(out_dir / 'depth_compare_summary.png', dpi=150)
        plt.close()

        # CSV
        import csv
        with open(out_dir / 'depth_compare.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['path', 'corr', 'mad'])
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f'Karşılaştırma tamamlandı: {out_dir} (n={len(rows)})')


if __name__ == '__main__':
    main()



