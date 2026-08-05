"""Compatibility re-export for root app.py imports.

Canonical implementation lives under ARTPS/src/models; this module keeps the
production import path (`src.models.optimized_autoencoder`) working without
duplicating architecture code.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_IMPL = Path(__file__).resolve().parents[2] / "ARTPS" / "src" / "models" / "optimized_autoencoder.py"
_spec = importlib.util.spec_from_file_location("_artps_optimized_autoencoder", _IMPL)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load OptimizedAutoencoder from {_IMPL}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

OptimizedAutoencoder = _mod.OptimizedAutoencoder
MarsRockDataset = _mod.MarsRockDataset
AutoencoderTrainer = _mod.AutoencoderTrainer
visualize_reconstruction = _mod.visualize_reconstruction

__all__ = [
    "OptimizedAutoencoder",
    "MarsRockDataset",
    "AutoencoderTrainer",
    "visualize_reconstruction",
]
