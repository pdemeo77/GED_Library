"""GED via Gromov-Wasserstein (GEDGW) solver.

Reference:
    Cheng, Q. et al. (2025).
    Computing Approximate Graph Edit Distance via Optimal Transport.
    SIGMOD'25.

This solver uses the Fused Gromov-Wasserstein (FGW) optimal transport distance
to approximate Graph Edit Distance. It is unsupervised — no training required.

The method constructs:
1. A cross-cost matrix from node label comparison
2. Structural cost matrices from adjacency information
3. Solves the FGW optimal transport problem using POT library

Requires: pot, torch (optional dependencies in pyproject.toml [ot])
"""

from typing import Optional

import networkx as nx
import numpy as np

from ged_lib.base import GEDSolver
from ged_lib.cost import CostModel
from ged_lib.solvers import register_solver
from ged_lib.solvers.neural.data_adapter import get_label_map


def _union_label_map(g1: nx.Graph, g2: nx.Graph) -> dict:
    """Shared label -> index mapping over the UNION of both graphs' labels.

    Used when no dataset label map is available: encoding each graph
    separately can place the same label in different one-hot columns,
    making identical labels look different to the cross cost.
    """
    labels = set()
    for g in (g1, g2):
        for _, data in g.nodes(data=True):
            if "label" in data:
                labels.add(data["label"])
    ordered = sorted(labels, key=lambda x: (type(x).__name__, str(x)))
    return {lab: i for i, lab in enumerate(ordered)}


def _get_label_features(
    g: nx.Graph,
    num_labels: Optional[int] = None,
    label_map: Optional[dict[str, int]] = None,
    label_to_idx: Optional[dict] = None,
) -> np.ndarray:
    """Extract one-hot node label features from a graph.

    Args:
        g: Input graph.
        num_labels: Total number of distinct labels. If None, auto-detect.
        label_map: Optional mapping from string labels to integer indices.
        label_to_idx: Optional shared label -> index mapping (e.g. from
            _union_label_map) so that both graphs of a pair use the same
            one-hot columns.

    Returns:
        Numpy array of shape (n, d) with one-hot features.
    """
    nodes = sorted(g.nodes())
    n = len(nodes)
    has_labels = n > 0 and all("label" in g.nodes[nd] for nd in nodes)

    if not has_labels:
        return np.full((n, 1), 2.0, dtype=np.float64)

    labels_raw = [g.nodes[nd].get("label", -1) for nd in nodes]

    if num_labels is not None and num_labels > 0:
        d = num_labels
        # Convert string labels to indices using label_map
        labels = []
        for lab in labels_raw:
            if isinstance(lab, str):
                if label_map:
                    labels.append(label_map.get(lab, -1))
                else:
                    labels.append(-1)
            elif isinstance(lab, (int, np.integer)):
                labels.append(int(lab))
            else:
                labels.append(-1)
    else:
        if label_to_idx is None:
            unique = sorted(set(labels_raw))
            label_to_idx = {l: i for i, l in enumerate(unique)}
        d = max(len(label_to_idx), 1)
        labels = [label_to_idx.get(l, -1) for l in labels_raw]

    features = np.zeros((n, d), dtype=np.float64)
    for i, label in enumerate(labels):
        if isinstance(label, (int, np.integer)) and 0 <= int(label) < d:
            features[i, int(label)] = 1.0
    return features


@register_solver("gedgw")
class GEDGWSolver(GEDSolver):
    """Graph Edit Distance via Fused Gromov-Wasserstein Optimal Transport.

    Unsupervised method — no training required. Uses POT's
    fused_gromov_wasserstein to compute an OT-based GED approximation.

    Args:
        cost_model: CostModel instance (used for alpha parameter guidance).
        alpha: FGW trade-off parameter between feature and structure cost.
            0 = pure structural, 1 = pure feature. Default 1/3.
        reverse_scale: Factor to convert FGW distance to GED estimate.
            Default 3/2 as in the reference paper.
    """

    def __init__(
        self,
        cost_model: CostModel = None,
        alpha: float = 1.0 / 3.0,
        reverse_scale: float = 3.0 / 2.0,
        dataset_name: Optional[str] = None,
    ) -> None:
        super().__init__(cost_model)
        self.alpha = alpha
        self.reverse_scale = reverse_scale
        self._label_map = get_label_map(dataset_name) if dataset_name else None
        if dataset_name and dataset_name.upper() == "AIDS":
            self._num_labels: Optional[int] = 29
        else:
            self._num_labels = None

    def compute(self, g1: nx.Graph, g2: nx.Graph) -> float:
        """Compute approximate GED using Fused Gromov-Wasserstein.

        Args:
            g1: First graph.
            g2: Second graph.

        Returns:
            Approximate Graph Edit Distance as a float.
        """
        import torch
        from ot.gromov import fused_gromov_wasserstein

        n1 = g1.number_of_nodes()
        n2 = g2.number_of_nodes()

        if n1 == 0 and n2 == 0:
            return 0.0

        nodes1 = sorted(g1.nodes())
        nodes2 = sorted(g2.nodes())

        shared_map = _union_label_map(g1, g2) if self._num_labels is None else None
        f1 = _get_label_features(
            g1, num_labels=self._num_labels, label_map=self._label_map, label_to_idx=shared_map
        )
        f2 = _get_label_features(
            g2, num_labels=self._num_labels, label_map=self._label_map, label_to_idx=shared_map
        )

        d1 = f1.shape[1]
        d2 = f2.shape[1]

        if d1 != d2:
            max_d = max(d1, d2)
            if d1 < max_d:
                f1 = np.pad(f1, ((0, 0), (0, max_d - d1)))
            if d2 < max_d:
                f2 = np.pad(f2, ((0, 0), (0, max_d - d2)))

        adj1 = nx.to_numpy_array(g1, nodelist=nodes1, dtype=np.float64)
        adj2 = nx.to_numpy_array(g2, nodelist=nodes2, dtype=np.float64)

        cross_cost = self._build_cross_cost(f1, f2)

        size = max(n1, n2)
        mu, nu, adj1_sq, adj2_sq, cross_sq = self._pad_to_square(
            n1, n2, size, adj1, adj2, cross_cost
        )

        mu_t = torch.from_numpy(mu).double()
        nu_t = torch.from_numpy(nu).double()
        cross_t = torch.from_numpy(cross_sq).double()
        c1_t = torch.from_numpy(adj1_sq).double()
        c2_t = torch.from_numpy(adj2_sq).double()

        T, log = fused_gromov_wasserstein(
            cross_t,
            c1_t,
            c2_t,
            mu_t,
            nu_t,
            "square_loss",
            alpha=self.alpha,
            armijo=True,
            verbose=False,
            log=True,
        )

        ged = log["fgw_dist"].item() * self.reverse_scale
        return float(ged)

    def _build_cross_cost(self, f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
        """Build cross-cost matrix from node features.

        Entry (i, j) = 0 if features match, 1 otherwise (scaled by n2).
        Uses vectorized comparison instead of nested Python loops.
        """
        n2 = f2.shape[0]
        match_matrix = (f1[:, None, :] == f2[None, :, :]).all(axis=2).astype(np.float64)
        cross_cost = (1.0 - match_matrix) * n2
        return cross_cost

    def _pad_to_square(
        self,
        n1: int,
        n2: int,
        size: int,
        adj1: np.ndarray,
        adj2: np.ndarray,
        cross_cost: np.ndarray,
    ) -> tuple:
        """Pad matrices to be square (size x size) and create weight vectors.

        Returns:
            Tuple of (mu, nu, adj1_sq, adj2_sq, cross_sq).
        """
        mu = np.ones(size, dtype=np.float64) / size
        nu = np.ones(size, dtype=np.float64) / size

        adj1_sq = np.zeros((size, size), dtype=np.float64)
        adj2_sq = np.zeros((size, size), dtype=np.float64)
        cross_sq = np.zeros((size, size), dtype=np.float64)

        adj1_sq[:n1, :n1] = adj1 * size
        adj2_sq[:n2, :n2] = adj2 * size
        cross_sq[:n1, :n2] = cross_cost

        if n1 < size:
            cross_sq[n1:, :] = 0.0

        return mu, nu, adj1_sq, adj2_sq, cross_sq
