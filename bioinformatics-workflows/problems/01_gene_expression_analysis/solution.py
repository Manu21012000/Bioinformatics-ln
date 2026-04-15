"""
Gene expression: PCA (2D/3D), clustering, Pydantic dataset model, Biopython FASTA smoke check.

Run from anywhere:
  python solution.py
from this directory, with repo root on PYTHONPATH (script adds the workflow package root automatically).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

WORKFLOW_ROOT = Path(__file__).resolve().parents[2]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))

from core.config import load_config
from core.models import GeneExpressionDataset
from core.validation import (
    clustering_accuracy,
    validate_clustering,
)
from visualizations.plots_2d import plot_expression_heatmap, plot_pca_2d
from visualizations.plots_3d import plot_pca_3d


def load_expression_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError("CSV must include a 'label' column.")
    y = df["label"].astype(int).to_numpy()
    feat = df.drop(columns=["label"]).to_numpy(dtype=float)
    return feat, y


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Gene expression PCA + clustering demo.")
    p.add_argument(
        "--data",
        type=Path,
        default=WORKFLOW_ROOT / "data" / "gene_expression_demo.csv",
        help="CSV: columns label, g0, g1, ...",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    args = p.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config()

    X, labels = load_expression_table(args.data)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(Xs)

    kmeans = KMeans(n_clusters=len(np.unique(labels)), n_init=10, random_state=42)
    pred = kmeans.fit_predict(Xs)

    acc = clustering_accuracy(labels, pred)
    ari = adjusted_rand_score(labels, pred)
    validate_clustering(acc, ari, cfg)

    dataset = GeneExpressionDataset(
        samples=X.shape[0],
        genes=X.shape[1],
        data=X.tolist(),
        labels=labels.tolist(),
    )
    _ = dataset  # explicit model validation side effect

    plot_expression_heatmap(X, args.out_dir / "expression_heatmap.png")
    plot_pca_2d(X_pca[:, :2], labels, args.out_dir / "pca_2d.png")
    plot_pca_3d(X_pca, labels, args.out_dir / "pca_3d.png")

    fasta = WORKFLOW_ROOT / "data" / "demo_sequences.fasta"
    if fasta.is_file():
        lengths = [len(rec.seq) for rec in SeqIO.parse(fasta, "fasta")]
        print("Biopython FASTA demo:", lengths)

    print(f"Clustering accuracy (Hungarian): {acc:.3f}")
    print(f"Adjusted Rand index: {ari:.3f}")
    print(f"Pydantic dataset OK: {dataset.samples} samples x {dataset.genes} genes")
    print(f"Wrote figures under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
