"""
ARTPS — Grafik teması (Matplotlib + Plotly koyu/uzay teması).

Tek `apply_chart_theme()` çağrısı ile tüm Matplotlib figürleri ve Plotly
grafikleri koyu mission-control paletine oturur. Grafiklerin verisi/colormap
mantığı değişmez; yalnızca zemin/eksen/metin renkleri temalanır.
"""

import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go

# Tema paleti (assets/style.css ile uyumlu)
BG = "#0E131C"
PANEL = "#141A24"
TXT = "#E6EAF2"
DIM = "#9AA7BD"
GRID = "#243044"
MARS = "#E2725B"
ICE = "#7DD3FC"

# Mars/uzay vurgu renk dizisi (kategorik grafikler için)
COLORWAY = ["#E2725B", "#7DD3FC", "#FBBF24", "#34D399", "#C1440E", "#A78BFA"]


def _register_plotly_template() -> None:
    """plotly_dark üzerine ARTPS paletini uygulayan bir şablon kaydeder.

    Yalnızca override edilecek alanları içeren boş bir Template oluşturup
    `plotly_dark+artps` birleşimini varsayılan yapar (taban temayı korur).
    """
    tpl = go.layout.Template()
    tpl.layout.paper_bgcolor = BG
    tpl.layout.plot_bgcolor = PANEL
    tpl.layout.font = dict(color=TXT, family="Inter, sans-serif")
    tpl.layout.colorway = COLORWAY
    pio.templates["artps"] = tpl
    pio.templates.default = "plotly_dark+artps"


def apply_chart_theme() -> None:
    """Matplotlib rcParams ve Plotly varsayılan şablonunu koyu temaya ayarlar."""
    plt.rcParams.update({
        "figure.facecolor": BG,
        "figure.edgecolor": BG,
        "savefig.facecolor": BG,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TXT,
        "axes.titlecolor": TXT,
        "axes.grid": False,
        "text.color": TXT,
        "xtick.color": DIM,
        "ytick.color": DIM,
        "grid.color": GRID,
        "legend.facecolor": PANEL,
        "legend.edgecolor": GRID,
    })
    # Plotly teması opsiyonel; sürüm farklarında app'i çökertmesin
    try:
        _register_plotly_template()
    except Exception:
        pass
