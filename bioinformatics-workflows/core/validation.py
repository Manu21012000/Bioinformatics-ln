from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

from core.config import WorkflowConfig
from core.models import NetworkMetrics


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Best-match accuracy under permutation of cluster ids (Hungarian on confusion matrix)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 0:
        return 0.0
    row_ind, col_ind = linear_sum_assignment(cm.max() - cm)
    return float(cm[row_ind, col_ind].sum() / len(y_true))


def validate_clustering(accuracy: float, ari: float, config: WorkflowConfig | None = None) -> bool:
    cfg = config or WorkflowConfig()
    if accuracy < cfg.min_clustering_accuracy:
        raise ValueError(
            f"Clustering accuracy {accuracy:.3f} below threshold {cfg.min_clustering_accuracy}"
        )
    if ari < cfg.min_ari:
        raise ValueError(f"Adjusted Rand index {ari:.3f} below threshold {cfg.min_ari}")
    return True


def validate_network(metrics: NetworkMetrics, config: WorkflowConfig | None = None) -> bool:
    cfg = config or WorkflowConfig()
    if metrics.num_nodes < cfg.min_network_nodes:
        raise ValueError(
            f"Too few nodes ({metrics.num_nodes}); expected >= {cfg.min_network_nodes}"
        )
    if cfg.max_network_density is not None and metrics.density > cfg.max_network_density:
        raise ValueError(
            f"Density {metrics.density:.4f} exceeds max {cfg.max_network_density} "
            "(expected a sparser graph for this gate)"
        )
    return True


def validate_simulation_solution(solution: Any, config: WorkflowConfig | None = None) -> bool:
    """Accepts SciPy `solve_ivp` result or any object with attribute `t` (time points)."""
    cfg = config or WorkflowConfig()
    t = getattr(solution, "t", None)
    if t is None:
        raise ValueError("Simulation solution must expose time array `.t`")
    if len(t) < cfg.min_simulation_time_points:
        raise ValueError(
            f"Simulation has {len(t)} time points; need >= {cfg.min_simulation_time_points}"
        )
    return True


def validate_time_points(times: Sequence[float] | np.ndarray, config: WorkflowConfig | None = None) -> bool:
    cfg = config or WorkflowConfig()
    n = len(times)
    if n < cfg.min_simulation_time_points:
        raise ValueError(
            f"Time series has {n} points; need >= {cfg.min_simulation_time_points}"
        )
    return True
