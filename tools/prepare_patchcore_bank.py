import os
import sys
from argparse import ArgumentParser

# Proje kökünü python path'e ekle
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from typing import List
from src.models.anomaly import PatchCore, PatchCoreConfig


def main():

    parser = ArgumentParser(description="PatchCore bellek bankası hazırlama")
    parser.add_argument('--dirs', nargs='+', type=str, required=True, help='Normal görüntü klasörleri')
    parser.add_argument('--out', type=str, default='results/patchcore_bank.pth', help='Çıktı yolu')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '' else 'cuda')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--coreset_pct', type=float, default=0.10)
    parser.add_argument('--feature_dim', type=int, default=256)
    parser.add_argument('--nn_k', type=int, default=5)

    args = parser.parse_args()

    print(f"Folders: {args.dirs}")
    print(f"Device: {args.device} | ImageSize: {args.image_size} | Batch: {args.batch_size}")

    cfg = PatchCoreConfig(image_size=args.image_size, device=args.device, coreset_pct=args.coreset_pct, feature_dim=args.feature_dim, nn_k=args.nn_k)
    model = PatchCore(cfg)
    model.prepare_from_dirs(args.dirs, save_path=args.out, batch_size=args.batch_size)
    print(f"Saved: {args.out}")


if __name__ == '__main__':
    main()


