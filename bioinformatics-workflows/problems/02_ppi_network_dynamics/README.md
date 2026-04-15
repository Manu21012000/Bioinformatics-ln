# PPI network dynamics

Maps DEGs from `data/deg_data.csv` to STRING, builds a NetworkX graph, exports hub metrics, validates `NetworkMetrics`, materializes edges as `PPIEdge` models, renders a 3D spring layout, and integrates an illustrative ODE on top degree-centrality hubs.

Requires internet for STRING unless you pass `--ppi-tsv` with a saved network TSV.
