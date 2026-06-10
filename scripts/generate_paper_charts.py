import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _apply_q1_style() -> None:
    # Font: prefer serif (Times-like). If Times New Roman is unavailable,
    # Matplotlib will fall back to the next available serif.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,  # TrueType fonts in PDF (selectable text)
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_grouped_bars(out_path: Path) -> None:
    """
    Strategy 1: "I am better" grouped bar chart for Table-2 metrics.
    Notes:
    - ARTPS bars are emphasized (color + value labels).
    - Competitors are desaturated (grays) + hatches (color-blind friendly).
    - Y-axis starts at 0.50 to visually amplify gaps (per author strategy).
    """
    metrics = ["AUROC", "AUPRC", "F1-Score"]

    series: Dict[str, List[float]] = {
        "ARTPS": [0.894, 0.847, 0.823],
        "PaDiM/PatchCore": [0.856, 0.812, 0.794],
        "Depth-only": [0.781, 0.698, 0.689],
        "OpenCV": [0.723, 0.645, 0.612],
    }

    x = np.arange(len(metrics))
    width = 0.18

    colors = {
        "ARTPS": "#1f4e79",  # deep blue
        "PaDiM/PatchCore": "#7a7a7a",
        "Depth-only": "#a8a8a8",
        "OpenCV": "#c8c8c8",
    }
    hatches = {
        "ARTPS": None,
        "PaDiM/PatchCore": "//",
        "Depth-only": "\\\\",
        "OpenCV": "..",
    }

    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    order = ["ARTPS", "PaDiM/PatchCore", "Depth-only", "OpenCV"]
    offsets = np.linspace(-(len(order) - 1) / 2, (len(order) - 1) / 2, len(order)) * width

    for name, off in zip(order, offsets):
        bars = ax.bar(
            x + off,
            series[name],
            width=width,
            label=name,
            color=colors[name],
            edgecolor="#4d4d4d" if name != "ARTPS" else "#1f4e79",
            linewidth=0.6,
        )
        if hatches[name]:
            for b in bars:
                b.set_hatch(hatches[name])

        # Only label ARTPS values (avoid clutter)
        if name == "ARTPS":
            for b, v in zip(bars, series[name]):
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    b.get_height() + 0.006,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color=colors[name],
                )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0.50, 1.00)
    ax.set_yticks(np.arange(0.50, 1.01, 0.10))

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35, color="#888888")
    ax.set_axisbelow(True)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(ncols=4, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 1.12))
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_ablation_drop(out_path: Path) -> None:
    """
    Strategy 2: "Every part was necessary" ablation drop plot (horizontal bars).
    IMPORTANT: Drops mix metrics in the source text (AUROC drops vs nDCG drop).
    We keep that honestly in labels so reviewers don't accuse metric mixing.
    """
    rows = [
        ("No curiosity scoring (nDCG drop)", 25.7),
        ("No anomaly fusion (AUROC drop)", 16.9),
        ("No depth estimation (AUROC drop)", 9.2),
        ("No input enhancement (AUROC drop)", 4.2),
    ]
    rows = sorted(rows, key=lambda t: t[1], reverse=True)
    labels = [r[0] for r in rows]
    drops = np.array([r[1] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    # Color intensity encodes severity (red/orange scale)
    cmap = plt.cm.OrRd
    norm = (drops - drops.min()) / max(1e-6, (drops.max() - drops.min()))
    colors = [cmap(0.35 + 0.55 * n) for n in norm]

    y = np.arange(len(labels))
    bars = ax.barh(y, drops, color=colors, edgecolor="#7a1f1f", linewidth=0.6)
    ax.invert_yaxis()

    ax.axvline(0.0, color="#444444", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Performance drop (%)")
    ax.set_xlim(0.0, max(30.0, float(drops.max() * 1.15)))
    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35, color="#888888")
    ax.set_axisbelow(True)

    for b, v in zip(bars, drops):
        ax.text(
            b.get_width() + 0.5,
            b.get_y() + b.get_height() / 2.0,
            f"{v:.1f}%",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color="#7a1f1f",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Q1-style vector charts for ARTPS paper.")
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(Path("docs") / "docs_second" / "figures"),
        help="Output directory for PDF figures.",
    )
    args = p.parse_args()

    _apply_q1_style()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_grouped_bars(out_dir / "fig_benchmark_grouped_bar.pdf")
    plot_ablation_drop(out_dir / "fig_ablation_drop.pdf")

    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()

