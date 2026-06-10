import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch

try:
    from src.core import CuriosityScorer, CuriosityWeights
    from src.models.optimized_autoencoder import OptimizedAutoencoder
    from src.models.depth_enhanced_classifier import DepthEnhancedClassifier
    from src.models.depth_estimation import MiDaSDepthEstimator
except ModuleNotFoundError:
    # Proje kökünü PYTHONPATH'e ekle
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.core import CuriosityScorer, CuriosityWeights
    from src.models.optimized_autoencoder import OptimizedAutoencoder
    from src.models.depth_enhanced_classifier import DepthEnhancedClassifier
    from src.models.depth_estimation import MiDaSDepthEstimator
 


def _normalize_map(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float32)
    lo, hi = np.percentile(arr, [2, 98])
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    norm = (arr - lo) / (hi - lo)
    return np.clip(norm, 0.0, 1.0)


@torch.no_grad()
def _ae_forward(ae: OptimizedAutoencoder, image: Image.Image, device: torch.device) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    img_rz = image.resize((128, 128), Image.LANCZOS)
    arr = np.array(img_rz, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0).to(device)
    reconstructed, latent = ae(x)
    rec = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
    mse = float(np.mean((arr - rec) ** 2))
    return mse, arr, rec, latent.squeeze().cpu().numpy()


def _ensure_models(device: torch.device) -> Dict[str, object]:
    models: Dict[str, object] = {}
    # Autoencoder
    ae = OptimizedAutoencoder(input_channels=3, latent_dim=1024)
    try:
        ckpt = torch.load('results/optimized_autoencoder_curiosity_extended.pth', map_location=device, weights_only=True)  # type: ignore
    except TypeError:
        ckpt = torch.load('results/optimized_autoencoder_curiosity_extended.pth', map_location=device)
    ae.load_state_dict(ckpt['model_state_dict'])
    ae.to(device).eval()
    models['autoencoder'] = ae

    # Classifier (opsiyonel)
    clf_path = Path('results/depth_enhanced_classifier.pth')
    if clf_path.exists():
        clf = DepthEnhancedClassifier(num_classes=5, rgb_features=1024, depth_features=14)
        try:
            ckp = torch.load(str(clf_path), map_location=device, weights_only=True)  # type: ignore
        except TypeError:
            ckp = torch.load(str(clf_path), map_location=device)
        clf.load_state_dict(ckp['model_state_dict'])
        clf.to(device).eval()
        models['classifier'] = clf

    # Depth
    models['depth'] = MiDaSDepthEstimator(model_type='DPT_Large', device=device)
    return models


def _flatten_depth_features(depth_features: Dict[str, float]) -> List[float]:
    keys = [
        'depth_mean', 'depth_std', 'depth_min', 'depth_max',
        'depth_median', 'depth_percentile_25', 'depth_percentile_75',
        'depth_variance', 'depth_skewness', 'depth_kurtosis',
        'surface_complexity', 'depth_gradient_mean', 'depth_gradient_std', 'depth_gradient_max',
    ]
    return [float(depth_features.get(k, 0.0)) for k in keys]


def _overlay_heatmap(base_rgb: np.ndarray, heat_norm: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    h, w = base_rgb.shape[:2]
    heat_col = plt.cm.inferno(heat_norm)[..., :3]
    base = base_rgb.astype(np.float32)
    if base.max() > 1.0:
        base = base / 255.0
    if heat_col.shape[:2] != (h, w):
        heat_col = np.array(Image.fromarray((heat_col * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
    out = (1.0 - alpha) * base + alpha * heat_col
    return np.clip(out, 0.0, 1.0)


def render_image_figure(img_path: Path, models: Dict[str, object], scorer: CuriosityScorer, device: torch.device, out_dir: Path) -> Optional[Dict[str, float]]:
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception:
        return None
    # AE
    mse, arr128, rec128, latent = _ae_forward(models['autoencoder'], img, device)
    diff = _normalize_map(((arr128 - rec128) ** 2).mean(axis=2))
    # Depth
    arr_full = np.array(img, dtype=np.float32) / 255.0
    depth_map, _ = models['depth'].estimate_depth(arr_full)
    dnorm = _normalize_map(depth_map)
    dfeats = models['depth'].extract_depth_features(depth_map)
    # Known value (opsiyonel)
    known_val = None
    if 'classifier' in models:
        rgb_t = torch.tensor(latent, dtype=torch.float32, device=device).unsqueeze(0)
        dvec = torch.tensor(_flatten_depth_features(dfeats), dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            pred = models['classifier'](rgb_t, dvec)
        known_val = float(torch.argmax(pred, dim=1).item() / 4.0)
    # Curiosity
    score, breakdown = scorer.compute(
        known_value_score=known_val,
        anomaly_mse=mse,
        combined_anomaly_score=None,
        depth_features=dfeats,
        reference_mse=0.003,
    )

    # Figure: 2x3 (orig, recon, diff overlay, depth, diff raw, bar)
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(img)
    ax1.set_title('Orijinal')
    ax1.axis('off')

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(rec128)
    ax2.set_title('AE Rekonstrüksiyon (128)')
    ax2.axis('off')

    ax3 = fig.add_subplot(2, 3, 3)
    overlay = _overlay_heatmap(np.array(img.resize((128, 128))), diff, alpha=0.6)
    ax3.imshow(overlay)
    ax3.set_title('AE Fark Overlay')
    ax3.axis('off')

    ax4 = fig.add_subplot(2, 3, 4)
    im4 = ax4.imshow(dnorm, cmap='plasma')
    ax4.set_title('Derinlik (norm)')
    ax4.axis('off')
    fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(diff, cmap='inferno')
    ax5.set_title('AE Fark (norm)')
    ax5.axis('off')

    ax6 = fig.add_subplot(2, 3, 6)
    bars = [breakdown.get('known', 0.0), breakdown.get('anomaly', 0.0), breakdown.get('depth_variance', 0.0), breakdown.get('roughness', 0.0)]
    ax6.bar(['known', 'anom', 'dvar', 'rough'], bars, color=['#4caf50', '#f44336', '#2196f3', '#ff9800'])
    ax6.set_ylim(0.0, max(0.01, max(bars) * 1.25))
    ax6.set_title(f'Curiosity: {score:.3f}')

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{img_path.stem}_fig.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        'path': str(img_path),
        'curiosity': float(score),
        'anomaly_mse': float(mse),
        'depth_variance': float(dfeats.get('depth_variance', 0.0)),
        'roughness': float(dfeats.get('roughness', 0.0)),
        'known_value_score': float(known_val if known_val is not None else 0.0),
    }


def render_dataset_summary(rows: List[Dict[str, float]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Histogramlar
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.hist([r['curiosity'] for r in rows], bins=20, color='#3f51b5', alpha=0.8)
    ax1.set_title('Curiosity Dağılımı')
    ax1.set_xlabel('skor')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter([r['anomaly_mse'] for r in rows], [r['roughness'] for r in rows], s=12, alpha=0.7, c='#ff5722')
    ax2.set_xlabel('AE MSE')
    ax2.set_ylabel('Roughness')
    ax2.set_title('AE MSE vs Roughness')

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.scatter([r['depth_variance'] for r in rows], [r['curiosity'] for r in rows], s=12, alpha=0.7, c='#009688')
    ax3.set_xlabel('Depth Variance')
    ax3.set_ylabel('Curiosity')
    ax3.set_title('DepthVar vs Curiosity')

    fig.tight_layout()
    fig.savefig(out_dir / 'dataset_summary.png', dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description='Makale figür üreteci')
    p.add_argument('--images_dir', type=str, required=True)
    p.add_argument('--weights', type=str, default='results/curiosity_weights.json')
    p.add_argument('--out_dir', type=str, default='results/paper_figs')
    p.add_argument('--limit', type=int, default=12)
    p.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'auto' else ('cuda' if args.device == 'cuda' else 'cpu'))

    # Curiosity scorer
    scorer = CuriosityScorer(CuriosityWeights())
    wpath = Path(args.weights)
    if wpath.exists():
        import json
        with open(wpath, 'r', encoding='utf-8') as f:
            scorer = CuriosityScorer(CuriosityWeights(**json.load(f)))

    models = _ensure_models(device)

    root = Path(args.images_dir)
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = [p for p in root.rglob('*') if p.suffix.lower() in exts]
    if args.limit and args.limit > 0:
        files = files[:args.limit]

    out_dir = Path(args.out_dir)
    results: List[Dict[str, float]] = []
    for pth in files:
        row = render_image_figure(pth, models, scorer, device, out_dir)
        if row:
            results.append(row)

    render_dataset_summary(results, out_dir)
    # Save CSV benzeri
    import csv
    with open(out_dir / 'summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['path', 'curiosity', 'anomaly_mse', 'depth_variance', 'roughness', 'known_value_score'])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f'Figürler: {out_dir} (adet: {len(results)})')


if __name__ == '__main__':
    main()


