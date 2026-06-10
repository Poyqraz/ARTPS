import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
from PIL import Image
import torch

try:
    from src.models.optimized_autoencoder import OptimizedAutoencoder
    from src.models.depth_enhanced_classifier import DepthEnhancedClassifier
    from src.models.depth_estimation import MiDaSDepthEstimator
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.models.optimized_autoencoder import OptimizedAutoencoder
    from src.models.depth_enhanced_classifier import DepthEnhancedClassifier
    from src.models.depth_estimation import MiDaSDepthEstimator


def _load_models(device: torch.device) -> Dict[str, object]:
    models: Dict[str, object] = {}
    # Autoencoder
    ae_path = Path("results/optimized_autoencoder_curiosity_extended.pth")
    if ae_path.exists():
        ae = OptimizedAutoencoder(input_channels=3, latent_dim=1024)
        try:
            ckpt = torch.load(str(ae_path), map_location=device, weights_only=True)  # type: ignore
        except TypeError:
            ckpt = torch.load(str(ae_path), map_location=device)
        ae.load_state_dict(ckpt['model_state_dict'])
        ae.to(device)
        ae.eval()
        models['autoencoder'] = ae
    # Classifier
    clf_path = Path("results/depth_enhanced_classifier.pth")
    if clf_path.exists():
        clf = DepthEnhancedClassifier(num_classes=5, rgb_features=1024, depth_features=14)
        try:
            ckpt = torch.load(str(clf_path), map_location=device, weights_only=True)  # type: ignore
        except TypeError:
            ckpt = torch.load(str(clf_path), map_location=device)
        clf.load_state_dict(ckpt['model_state_dict'])
        clf.to(device)
        clf.eval()
        models['classifier'] = clf
    # Depth estimator
    models['depth'] = MiDaSDepthEstimator(model_type="DPT_Large", device=device)
    return models


@torch.no_grad()
def _ae_anomaly(ae: OptimizedAutoencoder, img: Image.Image, device: torch.device):
    img_rz = img.resize((128, 128), Image.LANCZOS)
    arr = np.array(img_rz, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0).to(device)
    reconstructed, latent = ae(x)
    rec = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
    mse = float(np.mean((arr - rec) ** 2))
    return mse, arr, rec, latent.squeeze().cpu().numpy()


def _flatten_depth_features(depth_features: Dict[str, float]) -> List[float]:
    keys = [
        'depth_mean', 'depth_std', 'depth_min', 'depth_max',
        'depth_median', 'depth_percentile_25', 'depth_percentile_75',
        'depth_variance', 'depth_skewness', 'depth_kurtosis',
        'surface_complexity', 'depth_gradient_mean', 'depth_gradient_std', 'depth_gradient_max',
    ]
    return [float(depth_features.get(k, 0.0)) for k in keys]


def process_image(models: Dict[str, object], img_path: Path, device: torch.device) -> Optional[Dict]:
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception:
        return None

    # AE anomaly
    anomaly_mse = None
    latent_vec = None
    if 'autoencoder' in models:
        try:
            anomaly_mse, arr128, rec128, latent_vec = _ae_anomaly(models['autoencoder'], img, device)
        except Exception:
            anomaly_mse = None

    # Depth
    depth_variance = None
    roughness = None
    known_value_score = None
    if 'depth' in models:
        arr_full = np.array(img, dtype=np.float32) / 255.0
        depth_map, _ = models['depth'].estimate_depth(arr_full)
        dfeats = models['depth'].extract_depth_features(depth_map)
        depth_variance = float(dfeats.get('depth_variance', 0.0))
        roughness = float(dfeats.get('roughness', 0.0))
        # Classifier: combine rgb latent + depth features
        if 'classifier' in models and latent_vec is not None:
            rgb_t = torch.tensor(latent_vec, dtype=torch.float32, device=device).unsqueeze(0)
            depth_vec = torch.tensor(_flatten_depth_features(dfeats), dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                if device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        pred = models['classifier'](rgb_t, depth_vec)
                else:
                    pred = models['classifier'](rgb_t, depth_vec)
                pred_cls = int(torch.argmax(pred, dim=1).item())
            known_value_score = float(pred_cls / 4.0)

    row = {
        'path': str(img_path),
        'anomaly_mse': float(anomaly_mse) if anomaly_mse is not None else None,
        'combined_anomaly_score': None,  # isteğe bağlı: ayrı betikte üretilebilir
        'depth_variance': float(depth_variance) if depth_variance is not None else None,
        'roughness': float(roughness) if roughness is not None else None,
        'known_value_score': float(known_value_score) if known_value_score is not None else None,
    }
    return row


def main():
    parser = argparse.ArgumentParser(description="Görüntülerden Curiosity özelliklerini çıkar")
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'auto' else ('cuda' if args.device == 'cuda' else 'cpu'))
    models = _load_models(device)

    root = Path(args.images_dir)
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = [p for p in root.rglob('*') if p.suffix.lower() in exts]
    if args.limit and args.limit > 0:
        files = files[:args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_path, 'w', encoding='utf-8') as f:
        for p in files:
            row = process_image(models, p, device)
            if row is None:
                continue
            f.write(json.dumps(row) + "\n")
            count += 1
    print(f"Özellik çıkarımı tamamlandı: {count} görüntü → {out_path}")


if __name__ == '__main__':
    main()


