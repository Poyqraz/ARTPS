"""Manuscript-aligned serif typography for IAC qualitative figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

FIG_DPI = 180
TITLE_PT = 8.5


def apply_manuscript_serif(title_pt: float = TITLE_PT) -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "font.size": title_pt,
            "axes.titlesize": title_pt,
            "axes.labelsize": title_pt,
        }
    )


def save_manuscript_figure(fig, path: Path, dpi: int = FIG_DPI) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    im = Image.open(path)
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    im.save(path, format="PNG", optimize=True, compress_level=9)
