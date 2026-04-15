"""
3D Matplotlib figures. Avoid plt.tight_layout() on mplot3d axes — it often yields blank PNGs on save.

Each function writes a PNG when out_path is set (default in workflows). Open that file to view results;
nothing is shown interactively unless you pass out_path=None and run in an environment with a GUI backend.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection


def _save_fig(fig: Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")


def plot_pca_3d(
    X_pca: np.ndarray,
    labels: np.ndarray,
    out_path: Path | None = None,
    title: str = "3D PCA projection",
) -> None:
    """Scatter of samples in PCA space (first three PCs); color = class label."""
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        X_pca[:, 2],
        c=labels,
        cmap="tab10",
        s=45,
        depthshade=True,
        edgecolors="k",
        linewidth=0.35,
    )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    fig.colorbar(sc, ax=ax, shrink=0.65, label="label", pad=0.1)
    fig.subplots_adjust(left=0.02, right=0.90, top=0.94, bottom=0.06)
    ax.view_init(elev=22, azim=-55)
    if out_path is not None:
        _save_fig(fig, out_path)
        plt.close(fig)
    else:
        plt.show()


def plot_3d_network(
    G: nx.Graph,
    out_path: Path | None = None,
    seed: int = 42,
    title: str = "3D PPI layout (spring)",
) -> None:
    """Nodes = proteins, edges = interactions; positions from 3D spring layout (no geographic meaning)."""
    pos = nx.spring_layout(G, dim=3, seed=seed)

    xs = [pos[n][0] for n in G.nodes()]
    ys = [pos[n][1] for n in G.nodes()]
    zs = [pos[n][2] for n in G.nodes()]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xs, ys, zs, s=42, c="#1f77b4", depthshade=True, edgecolors="k", linewidth=0.3)

    for u, v in G.edges():
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        z = [pos[u][2], pos[v][2]]
        ax.plot(x, y, z, color="#555", linewidth=0.85, alpha=0.65)

    ax.set_title(title)
    ax.set_xlabel("layout x")
    ax.set_ylabel("layout y")
    ax.set_zlabel("layout z")
    fig.subplots_adjust(left=0.02, right=0.96, top=0.94, bottom=0.06)
    ax.view_init(elev=18, azim=-60)
    if out_path is not None:
        _save_fig(fig, out_path)
        plt.close(fig)
    else:
        plt.show()


def plot_infection_surface(
    I: np.ndarray,
    out_path: Path | None = None,
    title: str = "Infection surface (grid)",
) -> None:
    """Plot a 2D spatial field I[y,x] as height on a 3D surface (grid cell indices on x,y)."""
    ny, nx = I.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X, Y, I, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95, rstride=1, cstride=1
    )
    ax.set_title(title)
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.set_zlabel("infected fraction")
    fig.colorbar(surf, ax=ax, shrink=0.55, label="I", pad=0.12)
    fig.subplots_adjust(left=0.02, right=0.88, top=0.94, bottom=0.06)
    ax.view_init(elev=35, azim=-50)
    if out_path is not None:
        _save_fig(fig, out_path)
        plt.close(fig)
    else:
        plt.show()
