"""
2D plots for PCA scatter and expression heatmaps. Figures are written to PNG when out_path is given.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_pca_2d(
    X_pca: np.ndarray,
    labels: np.ndarray,
    out_path: Path | None = None,
    title: str = "2D PCA projection",
) -> None:
    """Scatter of samples using PC1 vs PC2; point color = class label."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", alpha=0.85, edgecolors="k", s=40)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(sc, ax=ax, label="label")
    plt.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    else:
        plt.show()


def plot_expression_heatmap(
    X: np.ndarray,
    out_path: Path | None = None,
    title: str = "Expression matrix (samples x genes)",
) -> None:
    """Heatmap: rows = samples, columns = genes (raw matrix values before scaling)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(X, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("gene index")
    ax.set_ylabel("sample index")
    fig.colorbar(im, ax=ax, label="value")
    plt.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    else:
        plt.show()
