# Gene expression analysis

Loads `data/gene_expression_demo.csv` (labels + gene features), standardizes features, runs PCA to three components, fits KMeans, and validates clustering accuracy and adjusted Rand index against `config/default.yaml` thresholds.

Outputs under `outputs/`: expression heatmap, 2D PCA, 3D PCA. Demonstrates `GeneExpressionDataset` in `core/models.py` and Biopython parsing of `data/demo_sequences.fasta`.
