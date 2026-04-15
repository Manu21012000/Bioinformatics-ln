from __future__ import annotations

from typing import List

import numpy as np
from pydantic import BaseModel, Field, model_validator


class GeneExpressionDataset(BaseModel):
    """Matrix of expression values with sample-wise class labels (e.g., conditions)."""

    samples: int = Field(ge=1)
    genes: int = Field(ge=1)
    data: List[List[float]]
    labels: List[int]

    @model_validator(mode="after")
    def _consistent_shape(self):
        if len(self.data) != self.samples:
            raise ValueError("data row count must equal samples")
        for row in self.data:
            if len(row) != self.genes:
                raise ValueError("each data row must have length genes")
        if len(self.labels) != self.samples:
            raise ValueError("labels length must equal samples")
        return self

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self.data, dtype=float)


class PPIEdge(BaseModel):
    protein_a: str = Field(min_length=1)
    protein_b: str = Field(min_length=1)
    score: float = Field(gt=0, le=1, description="STRING combined score on (0, 1]")

    @model_validator(mode="after")
    def _no_self_loop(self):
        if self.protein_a == self.protein_b:
            raise ValueError("Self-loops are not represented as PPIEdge rows")
        return self


class NetworkMetrics(BaseModel):
    num_nodes: int = Field(ge=0)
    num_edges: int = Field(ge=0)
    density: float = Field(ge=0.0, le=1.0)
    avg_clustering: float = Field(ge=0.0, le=1.0)
    n_components: int = Field(ge=0)


class SimulationResult(BaseModel):
    """Time series: each inner list is one species/track over time_points."""

    time_points: List[float]
    concentrations: List[List[float]]

    @model_validator(mode="after")
    def _align(self):
        if not self.time_points:
            return self
        t = len(self.time_points)
        for row in self.concentrations:
            if len(row) != t:
                raise ValueError("each concentration series must match len(time_points)")
        return self
