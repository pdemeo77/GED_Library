"""FGWAlign solver for Graph Edit Distance approximation.

Reference:
    Tang, Z. et al. (2024). Fast Graph Edit Distance via Optimal Transport
    with Graph Edit Distance Lower Bound.
    GitHub: https://github.com/squareRoot3/FGWAlign

This solver projects the Fused Gromov-Wasserstein transport plan to a
permutation matrix, producing an integer GED estimate directly — without
requiring any scaling factor. It is unsupervised and requires no training.

Requires: pot (POT library), torch
"""

from typing import Optional

import networkx as nx
import numpy as np
import torch

from ged_lib.base import GEDSolver
from ged_lib.cost import CostModel
from ged_lib.solvers import register_solver
from ged_lib.solvers._fgwalign_core import FGWAlign as _FGWAlign
from ged_lib.solvers.neural.data_adapter import get_label_map


def _nx_to_adj_tensor(g: nx.Graph, nodes: list) -> torch.Tensor:
    """Build a float32 adjacency tensor from a networkx Graph.

    Args:
        g: Input graph.
        nodes: Ordered list of node identifiers.

    Returns:
        Float32 tensor of shape (n, n).
    """
    node_idx = {nd: i for i, nd in enumerate(nodes)}
    n = len(nodes)
    adj = np.zeros((n, n), dtype=np.float32)
    for u, v in g.edges():
        i, j = node_idx[u], node_idx[v]
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    return torch.tensor(adj, dtype=torch.float32)


def _nx_to_label_tensor(
    g: nx.Graph, nodes: list, label_map: Optional[dict]
) -> Optional[torch.Tensor]:
    """Extract node label indices as a long tensor.

    Returns None if the graph has no ``label`` attributes on any node.

    Args:
        g: Input graph.
        nodes: Ordered list of node identifiers.
        label_map: Optional mapping from string labels to integer indices.

    Returns:
        Long tensor of shape (n,), or None if unlabeled.
    """
    if not nodes or not any("label" in g.nodes[nd] for nd in nodes):
        return None

    labels = []
    for nd in nodes:
        raw = g.nodes[nd].get("label", -1)
        if isinstance(raw, str):
            idx = label_map.get(raw, -1) if label_map else -1
        elif isinstance(raw, (int, np.integer)):
            idx = int(raw)
        else:
            idx = -1
        labels.append(idx)

    return torch.tensor(labels, dtype=torch.long)


@register_solver("fgwalign")
class FGWAlignSolver(GEDSolver):
    """Graph Edit Distance via FGWAlign (Fused Gromov-Wasserstein alignment).

    Unsupervised method — no training required. The FGW transport plan is
    projected to a permutation matrix, yielding an integer GED estimate
    directly (no scaling factor required, unlike GED-GW).

    Args:
        cost_model: CostModel instance (unused; kept for API compatibility).
        patience: Number of outer loop trials without improvement before stopping.
        topk: Number of top assignments explored per trial.
        alpha: Balance between label and structure cost (1.0 = full label weight).
        light: If True, use lighter (faster) inner Sinkhorn iterations.
        sparse: If True, use sparse matrix operations (for large graphs).
        dataset_name: Dataset name used to load the label map (e.g. "AIDS").
        device: Torch device string, e.g. "cpu" or "cuda".
    """

    def __init__(
        self,
        cost_model: CostModel = None,
        patience: int = 15,
        topk: int = 5,
        alpha: float = 1.0,
        light: bool = False,
        sparse: bool = False,
        dataset_name: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(cost_model)
        self.patience = patience
        self.topk = topk
        self.alpha = alpha
        self.light = light
        self.sparse = sparse
        self.device = device
        self._label_map = get_label_map(dataset_name) if dataset_name else None

    def compute(self, g1: nx.Graph, g2: nx.Graph) -> float:
        """Compute approximate GED using FGWAlign.

        Args:
            g1: First graph.
            g2: Second graph.

        Returns:
            Approximate Graph Edit Distance as a non-negative float.
        """
        n1 = g1.number_of_nodes()
        n2 = g2.number_of_nodes()

        if n1 == 0 and n2 == 0:
            return 0.0

        nodes1 = sorted(g1.nodes())
        nodes2 = sorted(g2.nodes())

        adj1 = _nx_to_adj_tensor(g1, nodes1)
        adj2 = _nx_to_adj_tensor(g2, nodes2)

        lab1 = _nx_to_label_tensor(g1, nodes1, self._label_map)
        lab2 = _nx_to_label_tensor(g2, nodes2, self._label_map)

        ged, _ = _FGWAlign(
            g1_adj=adj1,
            g2_adj=adj2,
            g1_labels=lab1,
            g2_labels=lab2,
            patience=self.patience,
            topk=self.topk,
            alpha=self.alpha,
            sparse=self.sparse,
            light=self.light,
            device=self.device,
        )

        return float(ged)
