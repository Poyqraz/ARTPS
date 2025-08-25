from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from .base import AnomalyModel


def _normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


@dataclass
class PaDiMConfig:
    backbone: str = "wide_resnet50_2"  # ResNet50 tabanlı, PaDiM için yaygın
    layers: Tuple[str, ...] = ("layer2", "layer3")
    image_size: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class _BackboneFeatures(nn.Module):
    def __init__(self, backbone: nn.Module, layers: Tuple[str, ...]):
        super().__init__()
        self.backbone = backbone
        self.layers = layers

    def forward(self, x: torch.Tensor):
        feats = {}
        for name, module in self.backbone.named_children():
            # Tam sınıflandırma başına geçmeden önce dur (PaDiM için ara katmanlar yeterli)
            if name in ("avgpool", "fc"):
                break
            x = module(x)
            if name in self.layers:
                feats[name] = x
        return feats


class PaDiM(AnomalyModel):
    """Hafif PaDiM uygulaması (tek-sınıf istatistik ile)."""

    def __init__(self, config: Optional[PaDiMConfig] = None):
        self.cfg = config or PaDiMConfig()
        self.device = torch.device(self.cfg.device)

        try:
            if self.cfg.backbone == "wide_resnet50_2":
                backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)
            else:
                backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except Exception:
            # Çevrimdışı/indirme hatasında ağırlıksız backbone'a düş
            if self.cfg.backbone == "wide_resnet50_2":
                backbone = models.wide_resnet50_2(weights=None)
            else:
                backbone = models.resnet50(weights=None)

        self.backbone = _BackboneFeatures(backbone, self.cfg.layers).to(self.device).eval()

        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.cfg.image_size, self.cfg.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # PaDiM istatistikleri
        self.mean: Optional[torch.Tensor] = None  # [C]
        self.cov_inv: Optional[torch.Tensor] = None  # [C,C]
        self.spatial_size: Optional[Tuple[int, int]] = None

    @torch.no_grad()
    def _extract(self, img_uint8: np.ndarray) -> torch.Tensor:
        x = self.preprocess(img_uint8).unsqueeze(0).to(self.device)
        feats = self.backbone(x)
        # Çoklu katman kanallarını birleştir
        feat_maps = [F.interpolate(feats[l], size=list(feats.values())[-1].shape[-2:], mode="bilinear", align_corners=False) for l in self.cfg.layers]
        fmap = torch.cat(feat_maps, dim=1)  # [1, C, H, W]
        self.spatial_size = fmap.shape[-2:]
        fmap = F.adaptive_avg_pool2d(fmap, output_size=self.spatial_size)  # no-op, tutarlılık
        C = fmap.shape[1]
        fmap = fmap.permute(0, 2, 3, 1).reshape(-1, C)  # [H*W, C]
        return fmap  # tek görüntü için uzamsal vektörler

    @torch.no_grad()
    def _extract_batch(self, imgs_uint8: List[np.ndarray]) -> torch.Tensor:
        # Toplu ön-işleme
        batch = torch.stack([self.preprocess(im) for im in imgs_uint8], dim=0).to(self.device)  # [B,3,H,W]
        feats = self.backbone(batch)
        # Katmanlar aynı uzamsal boyuta getirilsin
        last_size = list(feats.values())[-1].shape[-2:]
        feat_maps = [F.interpolate(feats[l], size=last_size, mode="bilinear", align_corners=False) for l in self.cfg.layers]
        fmap = torch.cat(feat_maps, dim=1)  # [B, C, H, W]
        self.spatial_size = fmap.shape[-2:]
        B, C, H, W = fmap.shape
        fmap = fmap.permute(0, 2, 3, 1).reshape(B * H * W, C)  # [B*H*W, C]
        return fmap

    @torch.no_grad()
    def prepare(self, dataset_dir: str, save_path: Optional[str] = None) -> None:
        from pathlib import Path
        return self.prepare_from_dirs([str(Path(dataset_dir))], save_path=save_path)

    @torch.no_grad()
    def prepare_from_dirs(self, dataset_dirs: List[str], save_path: Optional[str] = None, batch_size: int = 8) -> None:
        # Birden fazla klasörden özellik istatistiği çıkar (uzantı büyük/küçük fark etmeksizin)
        all_feats: List[torch.Tensor] = []
        from pathlib import Path
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'}
        total_found = 0
        for d in dataset_dirs:
            p = Path(d)
            if not p.exists():
                print(f"WARN: klasör bulunamadı: {p}")
                continue
            img_paths = [ip for ip in p.rglob('*') if ip.suffix.lower() in valid_ext]
            print(f"INFO: {p} içinde {len(img_paths)} görüntü bulundu")
            total_found += len(img_paths)
            # Toplu işleme
            batch: List[np.ndarray] = []
            for ip in img_paths:
                try:
                    img_bgr = cv2.imread(str(ip))
                    if img_bgr is None:
                        continue
                    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    batch.append(img)
                    if len(batch) >= max(1, int(batch_size)):
                        fmap = self._extract_batch(batch).cpu()
                        all_feats.append(fmap)
                        batch = []
                except Exception:
                    continue
            if batch:
                fmap = self._extract_batch(batch).cpu()
                all_feats.append(fmap)
        if len(all_feats) == 0:
            raise RuntimeError("Normal veri bulunamadı")
        print(f"INFO: Toplam işlenen görüntü: {total_found}")
        feats = torch.cat(all_feats, dim=0)  # [N, C]
        self.mean = feats.mean(dim=0)
        # Kovaryans — sayısal kararlılık için shrinkage
        X = feats - self.mean
        cov = (X.t() @ X) / max(1, X.shape[0] - 1)
        eps = 1e-5 * torch.eye(cov.shape[0])
        self.cov_inv = torch.linalg.pinv(cov + eps)

        if save_path:
            torch.save({"mean": self.mean, "cov_inv": self.cov_inv, "spatial": self.spatial_size, "cfg": self.cfg.__dict__}, save_path)

    def load(self, stats_path: str) -> None:
        try:
            ckpt = torch.load(stats_path, map_location=self.device, weights_only=True)  # type: ignore
        except TypeError:
            ckpt = torch.load(stats_path, map_location=self.device)
        self.mean = ckpt["mean"].to(self.device)
        self.cov_inv = ckpt["cov_inv"].to(self.device)
        self.spatial_size = tuple(ckpt.get("spatial", (self.cfg.image_size // 8, self.cfg.image_size // 8)))  # type: ignore
        # Config yükleme opsiyonel

    @torch.no_grad()
    def predict_anomaly_map(self, image_rgb_uint8: np.ndarray) -> np.ndarray:
        if self.mean is None or self.cov_inv is None:
            raise RuntimeError("PaDiM istatistikleri yüklenmedi/hazırlanmadı")
        fmap = self._extract(image_rgb_uint8)  # [H*W, C]
        X = fmap - self.mean  # [H*W, C]
        # Mahalanobis uzaklığı
        m = torch.einsum("nc,cc,nd->n", X, self.cov_inv, X.t()).sqrt()  # [H*W]
        H, W = self.spatial_size  # type: ignore
        amap = m.reshape(H, W).cpu().numpy()
        amap = cv2.resize(amap, (image_rgb_uint8.shape[1], image_rgb_uint8.shape[0]), interpolation=cv2.INTER_CUBIC)
        return _normalize_map(amap)


