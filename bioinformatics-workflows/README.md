# Bioinformatics workflows

Runnable **bioinformatics workflows** in one place: **gene expression** (PCA, clustering), **protein–protein interaction networks** (STRING, graph metrics, dynamics), and **epidemiological** simulation (spatial field + mean-field SIR). Shared **Pydantic** models and **validation gates** keep outputs consistent and checkable.

This folder is the multi-problem workflow suite; a smaller standalone PPI write-up still lives under `computational-biology-problems/problem_1_ppi_analysis/`.

## Highlights

- Structured problems with short READMEs per topic
- Statistics, machine learning, graph analysis, and dynamical systems
- Typed data with **Pydantic** (`core/models.py`)
- Programmatic checks (`core/validation.py`, optional `config/default.yaml`)
- **2D and 3D** figures (`visualizations/plots_2d.py`, `visualizations/plots_3d.py`)
- CLI entrypoints under `problems/`

## Layout

```text
bioinformatics-workflows/
├── problems/
│   ├── 01_gene_expression_analysis/
│   ├── 02_ppi_network_dynamics/
│   └── 03_epidemiological_simulation/
├── core/
│   ├── models.py
│   ├── validation.py
│   └── config.py
├── visualizations/
│   ├── plots_2d.py
│   └── plots_3d.py
├── config/
│   └── default.yaml
├── data/
├── requirements.txt
└── README.md
```

## What each workflow does

1. **Gene expression** — PCA (2D/3D), clustering with accuracy + ARI checks, Biopython FASTA smoke test.
2. **PPI networks** — STRING PPI, NetworkX metrics, Pydantic edge models, 3D spring layout, illustrative hub ODEs.
3. **Epidemiology** — Spatial infection surface (3D plot) plus mean-field SIR with SciPy.

## Quick start

From `bioinformatics-workflows/`:

```bash
pip install -r requirements.txt
```

Each script prepends this directory to `sys.path` so `core` and `visualizations` import cleanly:

```bash
python problems/01_gene_expression_analysis/solution.py
python problems/02_ppi_network_dynamics/solution.py
python problems/03_epidemiological_simulation/solution.py
```

Workflow 02 calls the STRING API (one request per run, with a short delay). Use `--ppi-tsv PATH` with a saved STRING network TSV for offline runs.

## Figures (where to look)

Scripts **do not open a plot window** by default: they save **PNG files** under each problem’s `outputs/` folder (for example `problems/01_gene_expression_analysis/outputs/pca_3d.png`). Open those files in an image viewer or IDE preview.

What each type draws:

| File pattern | What it is |
|--------------|------------|
| `expression_heatmap.png` | Rows = samples, columns = genes; color = raw expression value |
| `pca_2d.png` | PC1 vs PC2 scatter; color = class label |
| `pca_3d.png` | PC1–PC3 scatter; color = class label |
| `ppi_network_3d.png` | Proteins as points, edges as lines; 3D spring layout (layout axes are not biological coordinates) |
| `ode_hub_dynamics.png` | Time series of the illustrative ODE on top hub proteins |
| `infection_surface.png` | Grid `x`,`y` vs infected fraction `I` as a height field |

If a saved 3D PNG looked **blank**, that was often Matplotlib’s `tight_layout()` on 3D axes; the plotting code avoids that and uses explicit `savefig(..., bbox_inches="tight")`.

## Purpose

Demonstrate end-to-end translation from biological questions to code: multi-step pipelines, explicit validation, and typed data—not a generic “portfolio” label, but **named workflows** you can extend or reuse.
