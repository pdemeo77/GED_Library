"""Feature extraction for GED-SF solvers.

Ported from GED-SF/ged_computation.py and GED-SF/run_final_experiment.py.

Provides three groups of features used by the ML regressor:

1. **Structural features** (per-node): degree centrality, clustering coefficient,
   PageRank, and optionally betweenness centrality.  Used as input to the
   Frank-Wolfe solver.

2. **OT features** (8): extracted from the coupling matrix Pi produced by FW.
   Capture alignment quality, transport cost, sparsity, and structural mismatch.

3. **Graph-level features** (5): global comparison of the two input graphs
   (size difference, edge difference, density ratio, average degree difference,
   size ratio).

4. **Label distance matrix**: for labeled graphs (AIDS), 0 for same label, 2 for
   different; for unlabeled graphs, uniform matrix of 1s.

5. **Graph equalisation**: pad the smaller graph with dummy isolated nodes so
   both have the same number of nodes.
"""

from typing import Tuple

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Structural features
# ---------------------------------------------------------------------------


def compute_structural_features(
    graph: nx.Graph,
    feature_config: Tuple[str, ...] = ("Deg", "CC", "PR"),
) -> np.ndarray:
    """Compute structural (centrality) features for every node in *graph*.

    Args:
        graph: Input graph.
        feature_config: Tuple of feature names to compute.
            Supported: ``'Deg'``, ``'CC'``, ``'PR'``, ``'Betw'``.

    Returns:
        Array of shape ``(n_nodes, n_features)``.

    Raises:
        ValueError: If an unsupported feature name is given.
    """
    valid = {"Deg", "CC", "PR", "Betw"}
    if not set(feature_config).issubset(valid):
        raise ValueError(f"feature_config can only contain {sorted(valid)}, got {feature_config}")

    n = graph.number_of_nodes()
    features_list: list[np.ndarray] = []

    if "Deg" in feature_config:
        deg = np.array(list(nx.degree_centrality(graph).values()))
        features_list.append(deg)

    if "CC" in feature_config:
        cc = np.array(list(nx.clustering(graph).values()))
        features_list.append(cc)

    if "PR" in feature_config:
        pr = np.array(list(nx.pagerank(graph).values()))
        features_list.append(pr)

    if "Betw" in feature_config:
        k_sample = min(100, n)
        betw = np.array(
            list(nx.betweenness_centrality(graph, k=k_sample, normalized=True).values())
        )
        features_list.append(betw)

    return np.column_stack(features_list)


# ---------------------------------------------------------------------------
# Cross-cost matrix
# ---------------------------------------------------------------------------


def compute_cross_cost_matrix(F1: np.ndarray, F2: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distance between rows of F1 and F2.

    Uses the efficient expansion:
        ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>

    Args:
        F1: Feature matrix of graph 1, shape ``(n1, d)``.
        F2: Feature matrix of graph 2, shape ``(n2, d)``.

    Returns:
        Distance matrix of shape ``(n1, n2)``.
    """
    return np.sum(F1**2, axis=1, keepdims=True) + np.sum(F2**2, axis=1) - 2.0 * (F1 @ F2.T)


# ---------------------------------------------------------------------------
# OT features
# ---------------------------------------------------------------------------


def extract_ot_features(
    Pi: np.ndarray,
    cross_matrix: np.ndarray,
    A1: np.ndarray,
    A2: np.ndarray,
) -> dict:
    """Extract 8 OT-derived features from the coupling matrix Pi.

    Args:
        Pi: Coupling matrix from the Frank-Wolfe solver, shape ``(n, m)``.
        cross_matrix: Combined label + structural cost matrix, shape ``(n, m)``.
        A1: Adjacency matrix of graph 1, shape ``(n, n)``.
        A2: Adjacency matrix of graph 2, shape ``(m, m)``.

    Returns:
        Dictionary with 8 float entries:
        ``alignment_entropy``, ``alignment_confidence``, ``transport_cost``,
        ``marginal_balance``, ``coupling_sparsity``, ``max_coupling``,
        ``coupling_variance``, ``structural_mismatch``.
    """
    eps = 1e-10

    # 1. Alignment entropy
    non_zero = Pi[Pi > eps]
    alignment_entropy = (
        float(-np.sum(non_zero * np.log(non_zero + eps))) if len(non_zero) > 0 else 0.0
    )

    # 2. Alignment confidence: mean of row-wise max
    alignment_confidence = float(np.max(Pi, axis=1).mean())

    # 3. Transport cost
    transport_cost = float(np.sum(Pi * cross_matrix))

    # 4. Marginal balance
    n, m = Pi.shape
    target_row = np.ones(n) / n
    target_col = np.ones(m) / m
    row_marginal = Pi.sum(axis=1)
    col_marginal = Pi.sum(axis=0)
    marginal_balance = float(
        (np.abs(row_marginal - target_row).sum() + np.abs(col_marginal - target_col).sum()) / 2.0
    )

    # 5. Coupling sparsity
    coupling_sparsity = float((Pi < 1e-3).sum() / Pi.size)

    # 6. Max coupling
    max_coupling = float(np.max(Pi))

    # 7. Coupling variance
    coupling_variance = float(np.var(Pi))

    # 8. Structural mismatch (weighted degree difference)
    degrees_1 = A1.sum(axis=1)
    degrees_2 = A2.sum(axis=1)
    degree_diff = np.abs(degrees_1[:, np.newaxis] - degrees_2[np.newaxis, :])
    mask = Pi > eps
    total_mass = np.sum(Pi[mask])
    structural_mismatch = (
        float(np.sum(Pi[mask] * degree_diff[mask]) / total_mass) if total_mass > 0 else 0.0
    )

    return {
        "alignment_entropy": alignment_entropy,
        "alignment_confidence": alignment_confidence,
        "transport_cost": transport_cost,
        "marginal_balance": marginal_balance,
        "coupling_sparsity": coupling_sparsity,
        "max_coupling": max_coupling,
        "coupling_variance": coupling_variance,
        "structural_mismatch": structural_mismatch,
    }


# ---------------------------------------------------------------------------
# Graph-level features
# ---------------------------------------------------------------------------


def compute_graph_level_features(G1: nx.Graph, G2: nx.Graph) -> dict:
    """Compute 5 graph-level comparison features.

    These are computed on the **original** (un-padded) graphs.

    Args:
        G1: First graph.
        G2: Second graph.

    Returns:
        Dictionary with 5 float entries:
        ``size_diff``, ``edge_diff``, ``density_ratio``, ``avg_deg_diff``,
        ``size_ratio``.
    """
    n1, n2 = G1.number_of_nodes(), G2.number_of_nodes()
    e1, e2 = G1.number_of_edges(), G2.number_of_edges()

    d1 = nx.density(G1) if n1 > 1 else 0.0
    d2 = nx.density(G2) if n2 > 1 else 0.0
    density_ratio = min(d1, d2) / max(d1, d2) if max(d1, d2) > 0 else 1.0

    avg_deg1 = (2.0 * e1 / n1) if n1 > 0 else 0.0
    avg_deg2 = (2.0 * e2 / n2) if n2 > 0 else 0.0
    size_ratio = min(n1, n2) / max(n1, n2) if max(n1, n2) > 0 else 1.0

    return {
        "size_diff": float(abs(n1 - n2)),
        "edge_diff": float(abs(e1 - e2)),
        "density_ratio": float(density_ratio),
        "avg_deg_diff": float(abs(avg_deg1 - avg_deg2)),
        "size_ratio": float(size_ratio),
    }


# ---------------------------------------------------------------------------
# Label distance matrix
# ---------------------------------------------------------------------------


def compute_label_distance(g1: nx.Graph, g2: nx.Graph, n: int) -> np.ndarray:
    """Compute the label distance matrix between two equal-size graphs.

    For labeled graphs: 0 if labels match, 2 otherwise.
    For unlabeled graphs: uniform matrix of 1s.

    Labels are read from the ``'label'`` node attribute (GED_Library convention).

    Args:
        g1: First graph (must have *n* nodes).
        g2: Second graph (must have *n* nodes).
        n: Number of nodes in each graph.

    Returns:
        Label distance matrix of shape ``(n, n)``.
    """
    has_labels_1 = n > 0 and all("label" in g1.nodes[i] for i in sorted(g1.nodes())[:n])
    has_labels_2 = n > 0 and all("label" in g2.nodes[i] for i in sorted(g2.nodes())[:n])

    if not has_labels_1 or not has_labels_2:
        return np.ones((n, n), dtype=np.float64)

    nodes1 = sorted(g1.nodes())[:n]
    nodes2 = sorted(g2.nodes())[:n]
    labels1 = [g1.nodes[nd].get("label", "?") for nd in nodes1]
    labels2 = [g2.nodes[nd].get("label", "?") for nd in nodes2]

    dist = np.zeros((n, n), dtype=np.float64)
    for i, l1 in enumerate(labels1):
        for j, l2 in enumerate(labels2):
            dist[i, j] = 0.0 if l1 == l2 else 2.0
    return dist


# ---------------------------------------------------------------------------
# Graph equalisation
# ---------------------------------------------------------------------------


def equalize_graphs(G1: nx.Graph, G2: nx.Graph) -> Tuple[nx.Graph, nx.Graph, bool]:
    """Pad the smaller graph with dummy isolated nodes so both have equal size.

    Returns **copies** — the original graphs are never mutated.  If G1 is
    larger than G2, they are swapped so that G1 <= G2.

    Args:
        G1: First graph.
        G2: Second graph.

    Returns:
        Tuple of ``(G1_eq, G2_eq, swapped)`` where:
        - ``G1_eq`` and ``G2_eq`` have the same number of nodes.
        - ``swapped`` is True if the inputs were swapped.
    """
    swapped = False
    G1 = G1.copy()
    G2 = G2.copy()

    if G1.number_of_nodes() > G2.number_of_nodes():
        G1, G2 = G2, G1
        swapped = True

    existing = set(G1.nodes())
    next_id = max(existing, default=-1) + 1
    while G1.number_of_nodes() < G2.number_of_nodes():
        G1.add_node(next_id)
        next_id += 1

    return G1, G2, swapped
