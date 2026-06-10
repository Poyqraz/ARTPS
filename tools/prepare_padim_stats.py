import argparse
from pathlib import Path
import sys

import torch

# Proje kökünü PYTHONPATH'e ekle
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.anomaly import PaDiM, PaDiMConfig


def main():
    parser = argparse.ArgumentParser(description="PaDiM istatistiklerini hazırla ve kaydet")
    parser.add_argument("--dirs", nargs="+", help="Normal (anomalisiz) eğitim klasörleri listesi")
    parser.add_argument("--out", default="results/padim_stats.pth", help="Çıktı istatistik yolu")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="İstatistik çıkarım cihazı")
    parser.add_argument("--batch_size", type=int, default=16, help="Özellik çıkarım batch boyutu")
    args = parser.parse_args()

    cfg = PaDiMConfig(image_size=args.image_size, device=args.device)
    model = PaDiM(cfg)
    print(f"Folders: {args.dirs}")
    print(f"Device: {args.device} | ImageSize: {args.image_size} | Batch: {args.batch_size}")
    model.prepare_from_dirs(args.dirs, save_path=args.out, batch_size=args.batch_size)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()


