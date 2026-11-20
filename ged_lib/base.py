"""Abstract base class for all GED solvers."""

from abc import ABC, abstractmethod
from typing import Optional

import networkx as nx
import numpy as np

from ged_lib.cost import CostModel


class GEDSolver(ABC):
    """Base class for all Graph Edit Distance solvers.

    All solvers must implement the ``compute`` method.
    Optionally override ``compute_pairwise`` for a more efficient
    batch implementation.

    Args:
        cost_model: A CostModel instance defining edit costs.
            Defaults to unit costs.
    """

    def __init__(self, cost_model: Optional[CostModel] = None) -> None:
        self.cost_model = cost_model if cost_model is not None else CostModel()

    @abstractmethod
    def compute(self, g1: nx.Graph, g2: nx.Graph) -> float:
        """Compute the Graph Edit Distance between two graphs.

        Args:
            g1: First graph (networkx.Graph).
            g2: Second graph (networkx.Graph).

        Returns:
            The approximate (or exact) Graph Edit Distance as a float.
        """
        ...

    def compute_pairwise(self, graphs: list[nx.Graph]) -> np.ndarray:
        """Compute the full pairwise GED distance matrix.

        Args:
            graphs: List of networkx.Graph objects.

        Returns:
            A symmetric numpy array of shape (n, n) where D[i, j] = GED(graphs[i], graphs[j]).
        """
        n = len(graphs)
        D = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = self.compute(graphs[i], graphs[j])
                D[i, j] = d
                D[j, i] = d
        return D

    @property
    def name(self) -> str:
        """Human-readable name of this solver."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cost_model={self.cost_model!r})"
