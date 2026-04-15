"""
DEG -> STRING PPI -> NetworkX metrics -> illustrative hub ODE + 3D network visualization.

Requires network access unless --ppi-tsv is set.
"""

from __future__ import annotations

import argparse
import sys
import time
from io import StringIO
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import requests
from scipy.integrate import solve_ivp

WORKFLOW_ROOT = Path(__file__).resolve().parents[2]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))

from core.config import load_config
from core.models import NetworkMetrics, PPIEdge
from core.validation import validate_network, validate_simulation_solution
from visualizations.plots_3d import plot_3d_network

STRING_NETWORK_URL = "https://string-db.org/api/tsv/network"
STRING_REQUEST_DELAY_SEC = 1.0


def load_deg_table(path: Path, padj_max: float | None, abs_log2fc_min: float | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "gene" not in df.columns:
        raise ValueError("DEG table must include a 'gene' column.")
    if padj_max is not None and "padj" in df.columns:
        df = df[df["padj"] <= padj_max]
    if abs_log2fc_min is not None and "log2FC" in df.columns:
        df = df[df["log2FC"].abs() >= abs_log2fc_min]
    if df.empty:
        raise ValueError("No genes left after filtering; relax --padj-max / --abs-log2fc-min.")
    return df


def fetch_string_network(genes: list[str], species: int, required_score: int) -> str:
    params = {
        "identifiers": "%0d".join(genes),
        "species": species,
        "required_score": required_score,
    }
    time.sleep(STRING_REQUEST_DELAY_SEC)
    r = requests.get(STRING_NETWORK_URL, params=params, timeout=120)
    r.raise_for_status()
    return r.text


def parse_string_tsv(tsv_text: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(tsv_text), sep="\t")
    if df.empty:
        return df
    if "score" not in df.columns:
        raise ValueError("STRING TSV must contain a 'score' column.")
    return df


def normalize_min_score_threshold(min_score: float, observed_max: float) -> float:
    if observed_max <= 1.5:
        return min_score if min_score <= 1.5 else min_score / 1000.0
    return min_score if min_score > 1.5 else min_score * 1000.0


def ppi_edges_dataframe(ppi_df: pd.DataFrame, min_score: float) -> pd.DataFrame:
    obs_max = float(ppi_df["score"].max()) if not ppi_df.empty else 0.0
    thr = normalize_min_score_threshold(min_score, obs_max)
    return ppi_df[ppi_df["score"] >= thr].copy()


def build_graph(ppi_df: pd.DataFrame) -> nx.Graph:
    g = nx.Graph()
    for _, row in ppi_df.iterrows():
        a = row["preferredName_A"]
        b = row["preferredName_B"]
        s = float(row["score"])
        w = s if s <= 1.5 else s / 1000.0
        if g.has_edge(a, b):
            g[a][b]["weight"] = max(g[a][b]["weight"], w)
        else:
            g.add_edge(a, b, weight=w)
    return g


def graph_summary(g: nx.Graph) -> dict[str, float]:
    n = g.number_of_nodes()
    m = g.number_of_edges()
    if n == 0:
        return {
            "n_nodes": 0.0,
            "n_edges": 0.0,
            "density": 0.0,
            "avg_clustering": 0.0,
            "n_components": 0.0,
        }
    return {
        "n_nodes": float(n),
        "n_edges": float(m),
        "density": float(nx.density(g)),
        "avg_clustering": float(nx.average_clustering(g)),
        "n_components": float(nx.number_connected_components(g)),
    }


def compute_centralities(g: nx.Graph) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    return (
        nx.degree_centrality(g),
        nx.betweenness_centrality(g),
        nx.clustering(g),
    )


def centralities_table(
    g: nx.Graph,
    degree: dict[str, float],
    betweenness: dict[str, float],
    clustering: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for node in g.nodes():
        rows.append(
            {
                "gene": node,
                "degree_centrality": degree[node],
                "betweenness_centrality": betweenness[node],
                "clustering_coefficient": clustering[node],
                "degree": g.degree[node],
            }
        )
    return pd.DataFrame(rows).sort_values("degree_centrality", ascending=False).reset_index(drop=True)


def top_hubs(degree: dict[str, float], k: int) -> list[tuple[str, float]]:
    ranked = sorted(degree.items(), key=lambda x: x[1], reverse=True)
    return ranked[:k]


def induced_weight_matrix(g: nx.Graph, nodes: list[str]) -> np.ndarray:
    index = {name: i for i, name in enumerate(nodes)}
    n = len(nodes)
    w = np.zeros((n, n))
    for u, v, data in g.edges(data=True):
        if u in index and v in index:
            iu, iv = index[u], index[v]
            weight = float(data.get("weight", 1.0))
            w[iu, iv] = weight
            w[iv, iu] = weight
    return w


def ode_dynamics(_t: np.ndarray, p: np.ndarray, w: np.ndarray) -> np.ndarray:
    n = len(p)
    dp = np.zeros(n)
    for i in range(n):
        input_flux = np.sum(w[:, i] * p)
        output_flux = p[i] * np.sum(w[i, :])
        dp[i] = input_flux - output_flux
    return dp


def run_simulation(w: np.ndarray, t_span: tuple[float, float], p0: np.ndarray | None = None):
    n = w.shape[0]
    if p0 is None:
        p0 = np.ones(n)
    t_eval = np.linspace(t_span[0], t_span[1], 200)
    return solve_ivp(
        lambda t, y: ode_dynamics(t, y, w),
        t_span,
        p0,
        t_eval=t_eval,
        dense_output=True,
        rtol=1e-6,
        atol=1e-9,
    )


def write_ode_plot(sol, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    t_eval = sol.t
    if hasattr(sol, "sol") and sol.sol is not None:
        y = sol.sol(t_eval)
    else:
        y = sol.y
    plt.figure(figsize=(8, 4))
    for i in range(y.shape[0]):
        plt.plot(t_eval, y[i], label=f"P{i+1}")
    plt.xlabel("time (abstract units)")
    plt.ylabel("concentration (illustrative)")
    plt.title("Hub subgraph ODE demo (illustrative)")
    plt.legend(loc="best", fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def edges_to_models(ppi_df: pd.DataFrame) -> list[PPIEdge]:
    out: list[PPIEdge] = []
    for _, row in ppi_df.iterrows():
        s = float(row["score"])
        s01 = s if s <= 1.5 else s / 1000.0
        s01 = min(max(s01, 1e-6), 1.0)
        out.append(
            PPIEdge(
                protein_a=str(row["preferredName_A"]),
                protein_b=str(row["preferredName_B"]),
                score=s01,
            )
        )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="DEG -> STRING PPI -> network metrics -> ODE demo.")
    p.add_argument(
        "--deg-file",
        type=Path,
        default=WORKFLOW_ROOT / "data" / "deg_data.csv",
    )
    p.add_argument("--species", type=int, default=9606)
    p.add_argument("--required-score", type=int, default=400)
    p.add_argument("--min-score", type=float, default=0.7)
    p.add_argument("--padj-max", type=float, default=None)
    p.add_argument("--abs-log2fc-min", type=float, default=None)
    p.add_argument("--top-k-hubs", type=int, default=10)
    p.add_argument("--ppi-tsv", type=Path, default=None)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    p.add_argument("--t-end", type=float, default=100.0)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    deg_df = load_deg_table(args.deg_file, args.padj_max, args.abs_log2fc_min)
    genes = deg_df["gene"].astype(str).str.strip().unique().tolist()

    if args.ppi_tsv is not None:
        tsv_text = args.ppi_tsv.read_text(encoding="utf-8")
    else:
        try:
            tsv_text = fetch_string_network(genes, args.species, args.required_score)
        except requests.RequestException as e:
            print(
                "STRING request failed. Save a network TSV and pass --ppi-tsv PATH.\n"
                f"Error: {e}",
                file=sys.stderr,
            )
            return 2

    ppi_raw = parse_string_tsv(tsv_text)
    if ppi_raw.empty:
        print("No interactions returned after parsing.", file=sys.stderr)
        return 3

    ppi_f = ppi_edges_dataframe(ppi_raw, args.min_score)
    if ppi_f.empty:
        print("No edges left after --min-score filter.", file=sys.stderr)
        return 4

    _edges = edges_to_models(ppi_f)
    print(f"Validated {len(_edges)} PPI edges as Pydantic models.")

    g = build_graph(ppi_f)
    summary = graph_summary(g)
    metrics = NetworkMetrics(
        num_nodes=int(summary["n_nodes"]),
        num_edges=int(summary["n_edges"]),
        density=float(summary["density"]),
        avg_clustering=float(summary["avg_clustering"]),
        n_components=int(summary["n_components"]),
    )
    validate_network(metrics, cfg)

    deg_c, bet_c, clu_c = compute_centralities(g)
    table = centralities_table(g, deg_c, bet_c, clu_c)

    hubs_path = args.out_dir / "hubs.csv"
    table.to_csv(hubs_path, index=False)
    summary_path = args.out_dir / "network_summary.csv"
    pd.DataFrame([metrics.model_dump()]).to_csv(summary_path, index=False)

    print("Network metrics:", metrics.model_dump())
    print(f"Wrote {hubs_path}")

    if not args.no_plot and g.number_of_nodes() > 0:
        plot_3d_network(g, args.out_dir / "ppi_network_3d.png")

    hubs = top_hubs(deg_c, args.top_k_hubs)
    hub_names = [h[0] for h in hubs]
    if len(hub_names) < 2:
        print("Fewer than two hub nodes; skipping ODE demo.", file=sys.stderr)
        return 0

    w = induced_weight_matrix(g, hub_names)
    if not np.any(w > 0):
        print("Hub subgraph has no edges; skipping ODE demo.", file=sys.stderr)
        return 0

    sol = run_simulation(w, (0.0, args.t_end))
    validate_simulation_solution(sol, cfg)
    if not sol.success:
        print(f"ODE integration failed: {sol.message}", file=sys.stderr)
        return 5

    ode_path = args.out_dir / "ode_timeseries.csv"
    t_eval = sol.t
    if hasattr(sol, "sol") and sol.sol is not None:
        y_eval = sol.sol(t_eval)
    else:
        y_eval = sol.y
    ode_df = pd.DataFrame(y_eval.T, columns=[f"P_{name}" for name in hub_names])
    ode_df.insert(0, "t", t_eval)
    ode_df.to_csv(ode_path, index=False)
    print(f"Wrote {ode_path}")

    if not args.no_plot:
        write_ode_plot(sol, args.out_dir / "ode_hub_dynamics.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
