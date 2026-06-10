import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2


class _Spinner:
    def __init__(self, *_args, **_kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False


class _StStub:
    def __init__(self):
        self.session_state: Dict[str, object] = {}
    def info(self, *args, **kwargs):
        pass
    def warning(self, *args, **kwargs):
        pass
    def error(self, *args, **kwargs):
        pass
    def success(self, *args, **kwargs):
        pass
    def spinner(self, *args, **kwargs):
        return _Spinner()
    # Placeholders used by app; not needed here but kept to avoid attribute errors
    def markdown(self, *args, **kwargs):
        pass
    def image(self, *args, **kwargs):
        pass
    def set_page_config(self, *args, **kwargs):
        pass
    def cache_resource(self, func=None, **_kwargs):
        # Basit dekoratör: fonksiyonu aynen döndür (cache yok)
        if func is None:
            def decorator(f):
                return f
            return decorator
        return func


def _overlay(base_rgb: np.ndarray, heat_norm: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = base_rgb.shape[:2]
    heat_col = plt.cm.inferno(heat_norm)[..., :3]
    base = base_rgb.astype(np.float32)
    if base.max() > 1.0:
        base = base / 255.0
    if heat_col.shape[:2] != (h, w):
        heat_col = np.array(Image.fromarray((heat_col * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
    out = (1.0 - alpha) * base + alpha * heat_col
    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description="App analiz akışıyla doğru kutulama overlay'leri üret")
    p.add_argument('--images_dir', type=str, required=True)
    p.add_argument('--out_dir', type=str, default='results/paper_figs/detection_overlays_app')
    args = p.parse_args()

    # streamlit stub enjekte et
    sys.modules['streamlit'] = _StStub()

    # Proje kökünü sys.path'e ekle
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import app as appmod  # type: ignore

    # Modelleri yükle
    models = appmod.load_models()

    root = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = [p for p in root.rglob('*') if p.suffix.lower() in exts]

    for pth in files:
        img = Image.open(pth).convert('RGB')
        res = appmod.analyze_mars_image(models, img)
        base_u8 = (res['original'] * 255).astype(np.uint8)
        heat = res['combined_anomaly_map'].astype(np.float32)
        overlay = _overlay(base_u8, heat, alpha=0.45)
        # kutular
        disp = overlay.copy()
        for i, det in enumerate(res.get('detections') or [], start=1):
            x, y, w, h = det['x'], det['y'], det['w'], det['h']
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"#{i}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(disp, (x, max(0, y - th - 6)), (x + tw + 6, y - 2), (0, 255, 0), -1)
            cv2.putText(disp, label, (x + 3, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        Image.fromarray(disp).save(out_dir / f"{pth.stem}_det_overlay_app.png")
    print(f"Overlay üretildi: {out_dir} (adet: {len(files)})")


if __name__ == '__main__':
    main()


