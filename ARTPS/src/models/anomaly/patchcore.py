from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import cv2


def _normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)


@dataclass
class PatchCoreConfig:
    backbone: str = "wide_resnet50_2"
    layers: Tuple[str, ...] = ("layer2", "layer3")
    image_size: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    feature_dim: int = 256  # rastgele izdüşüm boyutu
    coreset_pct: float = 0.10  # bellek azaltma oranı
    nn_k: int = 5  # k-NN


class _BackboneFeatures(nn.Module):
    def __init__(self, backbone: nn.Module, layers: Tuple[str, ...]):
        super().__init__()
        self.backbone = backbone
        self.layers = layers

    def forward(self, x: torch.Tensor):
        feats = {}
        for name, module in self.backbone.named_children():
            if name in ("avgpool", "fc"):
                break
            x = module(x)
            if name in self.layers:
                feats[name] = x
        return feats


class RandomProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # Sabit rastgele projeksiyon (eğitimsiz)
        W = torch.randn(in_dim, out_dim) / (in_dim ** 0.5)
        self.register_buffer("W", W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W


class PatchCore:
    """Basitleştirilmiş PatchCore: orta katman özellik bankası + kNN uzaklık ısı haritası."""

    def __init__(self, config: Optional[PatchCoreConfig] = None):
        self.cfg = config or PatchCoreConfig()
        self.device = torch.device(self.cfg.device)

        # Backbone
        try:
            if self.cfg.backbone == "wide_resnet50_2":
                backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)
            else:
                backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except Exception:
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

        # Bellek bankası
        self.memory_bank: Optional[torch.Tensor] = None  # [N, D]
        self.spatial_size: Optional[Tuple[int, int]] = None
        self.projector: Optional[RandomProjector] = None

    @torch.no_grad()
    def _extract_feats_batch(self, imgs_uint8: List[np.ndarray]) -> torch.Tensor:
        batch = torch.stack([self.preprocess(im) for im in imgs_uint8], dim=0).to(self.device)  # [B,3,H,W]
        feats = self.backbone(batch)
        last_size = list(feats.values())[-1].shape[-2:]
        feat_maps = [F.interpolate(feats[l], size=last_size, mode="bilinear", align_corners=False) for l in self.cfg.layers]
        fmap = torch.cat(feat_maps, dim=1)  # [B,C,H,W]
        B, C, H, W = fmap.shape
        self.spatial_size = (H, W)
        fmap = fmap.permute(0, 2, 3, 1).reshape(B * H * W, C)
        return fmap  # [B*H*W, C]

    @torch.no_grad()
    def _ensure_projector(self, in_dim: int) -> None:
        if self.projector is None or self.projector.W.shape[0] != in_dim:
            self.projector = RandomProjector(in_dim, int(self.cfg.feature_dim)).to(self.device)

    @torch.no_grad()
    def prepare_from_dirs(self, dataset_dirs: List[str], save_path: Optional[str] = None, batch_size: int = 8) -> None:
        from pathlib import Path
        import cv2

        valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
        all_paths: List[Path] = []
        for d in dataset_dirs:
            p = Path(d)
            if not p.exists():
                print(f"WARN: klasör bulunamadı: {p}")
                continue
            imgs = [ip for ip in p.rglob('*') if ip.suffix.lower() in valid_ext]
            print(f"INFO: {p} içinde {len(imgs)} görüntü bulundu")
            all_paths.extend(imgs)
        if len(all_paths) == 0:
            raise RuntimeError("Normal veri bulunamadı")

        feats_list: List[torch.Tensor] = []
        batch: List[np.ndarray] = []
        for ip in all_paths:
            try:
                bgr = cv2.imread(str(ip))
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                batch.append(rgb)
                if len(batch) >= batch_size:
                    fmap = self._extract_feats_batch(batch)
                    self._ensure_projector(fmap.shape[1])
                    fmap = self.projector(fmap)
                    feats_list.append(fmap.detach().cpu())
                    batch = []
            except Exception:
                continue
        if batch:
            fmap = self._extract_feats_batch(batch)
            self._ensure_projector(fmap.shape[1])
            fmap = self.projector(fmap)
            feats_list.append(fmap.detach().cpu())

        feats = torch.cat(feats_list, dim=0)  # [N, D]

        # Coreset (basit örnekleme)
        n_total = feats.shape[0]
        n_keep = max(1, int(self.cfg.coreset_pct * n_total))
        perm = torch.randperm(n_total)[:n_keep]
        bank = feats[perm].contiguous()
        self.memory_bank = bank.to(self.device)

        if save_path:
            torch.save({
                "bank": self.memory_bank.detach().cpu(),
                "spatial": self.spatial_size,
                "cfg": self.cfg.__dict__,
                "proj_w": None if self.projector is None else self.projector.W.detach().cpu(),
            }, save_path)

    def load(self, bank_path: str) -> None:
        try:
            ckpt = torch.load(bank_path, map_location=self.device, weights_only=True)  # type: ignore
        except TypeError:
            ckpt = torch.load(bank_path, map_location=self.device)
        self.memory_bank = ckpt["bank"].to(self.device)
        self.spatial_size = tuple(ckpt.get("spatial", (self.cfg.image_size // 8, self.cfg.image_size // 8)))  # type: ignore
        cfgd = ckpt.get("cfg", None)
        if cfgd is not None:
            # bazı alanlar güncel olmayabilir; kritik olanları güncelle
            self.cfg.feature_dim = int(cfgd.get("feature_dim", self.cfg.feature_dim))
        proj_w = ckpt.get("proj_w", None)
        if proj_w is not None:
            self.projector = RandomProjector(in_dim=proj_w.shape[0], out_dim=proj_w.shape[1]).to(self.device)
            with torch.no_grad():
                self.projector.W.copy_(proj_w.to(self.device))

    @torch.no_grad()
    def predict_anomaly_map(self, image_rgb_uint8: np.ndarray) -> np.ndarray:
        if self.memory_bank is None:
            raise RuntimeError("PatchCore bellek bankası yüklenmedi/hazırlanmadı")
        # Özellik çıkar
        x = self.preprocess(image_rgb_uint8).unsqueeze(0).to(self.device)
        feats = self.backbone(x)
        last_size = list(feats.values())[-1].shape[-2:]
        fmap = torch.cat([F.interpolate(feats[l], size=last_size, mode="bilinear", align_corners=False) for l in self.cfg.layers], dim=1)
        _, C, H, W = fmap.shape
        self.spatial_size = (H, W)
        fmap = fmap.permute(0, 2, 3, 1).reshape(-1, C)  # [H*W, C]
        self._ensure_projector(fmap.shape[1])
        z = self.projector(fmap)  # [H*W, D]
        # kNN uzaklık (L2)
        # bank: [N, D], z: [M, D]
        # Dist^2 = ||z||^2 + ||bank||^2 - 2 z bank^T
        bank = self.memory_bank  # [N, D]
        z_norm2 = (z ** 2).sum(dim=1, keepdim=True)  # [M,1]
        bank_norm2 = (bank ** 2).sum(dim=1).unsqueeze(0)  # [1,N]
        sim = z @ bank.t()  # [M,N]
        dist2 = z_norm2 + bank_norm2 - 2.0 * sim
        k = max(1, int(self.cfg.nn_k))
        topk, _ = torch.topk(dist2, k=min(k, dist2.shape[1]), dim=1, largest=False)
        d = torch.sqrt(torch.clamp(topk.mean(dim=1), min=0.0))  # [M]
        amap = d.reshape(H, W).detach().cpu().numpy()
        amap = cv2.resize(amap, (image_rgb_uint8.shape[1], image_rgb_uint8.shape[0]), interpolation=cv2.INTER_CUBIC)
        return _normalize_map(amap)


