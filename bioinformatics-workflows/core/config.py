from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class WorkflowConfig(BaseModel):
    """Defaults for validation gates and optional paths (override via YAML)."""

    min_clustering_accuracy: float = Field(default=0.75, ge=0.0, le=1.0)
    min_ari: float = Field(default=0.7, ge=-1.0, le=1.0)
    min_network_nodes: int = Field(default=3, ge=0)
    max_network_density: float | None = Field(
        default=None,
        description="If set, fail validation when density exceeds this (sparse networks).",
    )
    min_simulation_time_points: int = Field(default=50, ge=1)


def load_config(path: Path | None = None) -> WorkflowConfig:
    if path is None:
        here = Path(__file__).resolve().parent.parent
        candidate = here / "config" / "default.yaml"
        path = candidate if candidate.is_file() else None
    if path is None or not path.is_file():
        return WorkflowConfig()
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return WorkflowConfig.model_validate(raw)
