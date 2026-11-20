"""Data adapter: convert networkx.Graph to torch tensor format for neural solvers.

The neural models from GED-via-Optimal-Transport expect a dict of torch tensors
with keys: edge_index_1, edge_index_2, features_1, features_2, n1, n2, etc.
This module bridges our internal networkx.Graph format to that representation.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import torch


def _load_label_map(dataset_name: str = "") -> dict[str, int]:
    """Load label-to-index mapping from labels.json if available.

    Args:
        dataset_name: Dataset name hint (e.g. "AIDS").

    Returns:
        Dictionary mapping label strings to integer indices, or empty dict.
    """
    if dataset_name.upper() == "AIDS":
        label_path = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "Dataset"
            / "AIDS"
            / "labels.json"
        )
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


@lru_cache(maxsize=8)
def _get_label_map_cached(dataset_name: str) -> dict[str, int]:
    """Cached version of _load_label_map."""
    return _load_label_map(dataset_name)


def resolve_label(label, num_labels: int, label_map: Optional[dict[str, int]] = None) -> int:
    """Convert a node label to an integer index for one-hot encoding.

    Args:
        label: The node label value (str, int, or np.integer).
        num_labels: Total number of label dimensions.
        label_map: Optional mapping from string labels to indices.

    Returns:
        Integer index in [0, num_labels), or -1 if unmappable.
    """
    if isinstance(label, str):
        if label_map:
            return label_map.get(label, -1)
        # Fallback: no mapping available
        return -1
    if isinstance(label, (int, np.integer)):
        return int(label) if 0 <= int(label) < num_labels else -1
    return -1


def nx_to_neural_data(
    g1: nx.Graph,
    g2: nx.Graph,
    num_labels: int = 0,
    target_mode: str = "linear",
    true_ged: Optional[float] = None,
    device: torch.device = None,
    label_map: Optional[dict[str, int]] = None,
) -> dict:
    """Convert two networkx graphs to a data dict for neural GED models.

    Args:
        g1: First graph.
        g2: Second graph.
        num_labels: Number of one-hot label dimensions. 0 = unlabeled.
        target_mode: "linear" or "exp" — how GED targets are normalized.
        true_ged: Optional ground truth GED for computing targets.
        device: Torch device to place tensors on.
        label_map: Optional mapping from string labels to integer indices.

    Returns:
        Dictionary with torch tensors suitable for neural model forward pass.
    """
    if device is None:
        device = torch.device("cpu")

    nodes1 = sorted(g1.nodes())
    nodes2 = sorted(g2.nodes())
    n1 = len(nodes1)
    n2 = len(nodes2)

    edge_index_1 = _build_edge_index(g1, nodes1, device)
    edge_index_2 = _build_edge_index(g2, nodes2, device)

    features_1 = _build_features(g1, nodes1, num_labels, device, label_map=label_map)
    features_2 = _build_features(g2, nodes2, num_labels, device, label_map=label_map)

    data = {
        "edge_index_1": edge_index_1,
        "edge_index_2": edge_index_2,
        "features_1": features_1,
        "features_2": features_2,
        "n1": n1,
        "n2": n2,
    }

    m1 = g1.number_of_edges()
    m2 = g2.number_of_edges()
    avg_v = (n1 + n2) / 2.0
    hb = max(n1, n2) + max(m1, m2)
    data["avg_v"] = avg_v
    data["hb"] = hb

    if true_ged is not None:
        if target_mode == "exp":
            data["target"] = torch.exp(torch.tensor([-true_ged / avg_v])).float().to(device)
        elif target_mode == "linear":
            data["target"] = torch.tensor([true_ged / hb]).float().to(device)

    return data


def _build_edge_index(g: nx.Graph, nodes: list, device: torch.device) -> torch.Tensor:
    """Build edge index tensor with self-loops (format expected by GCN/GIN).

    Returns:
        Tensor of shape (2, 2*E + N) with bidirectional edges + self-loops.
    """
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    edges = []
    for u, v in g.edges():
        ui, vi = node_to_idx[u], node_to_idx[v]
        edges.append([ui, vi])
        edges.append([vi, ui])
    for i in range(len(nodes)):
        edges.append([i, i])

    if len(edges) == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    return torch.tensor(edges, dtype=torch.long, device=device).t()


def _build_features(
    g: nx.Graph,
    nodes: list,
    num_labels: int,
    device: torch.device,
    label_map: Optional[dict[str, int]] = None,
) -> torch.Tensor:
    """Build node feature matrix (one-hot labels or constant features).

    Args:
        g: Input graph.
        nodes: Sorted node list.
        num_labels: Number of one-hot dimensions. 0 = unlabeled.
        device: Torch device.
        label_map: Optional mapping from string labels to integer indices.

    Returns:
        Feature tensor of shape (n, d).
    """
    n = len(nodes)
    has_labels = n > 0 and all("label" in g.nodes[nd] for nd in nodes)

    if has_labels and num_labels > 0:
        features = torch.zeros((n, num_labels), device=device)
        for i, node in enumerate(nodes):
            label = g.nodes[node].get("label", 0)
            idx = resolve_label(label, num_labels, label_map)
            if 0 <= idx < num_labels:
                features[i, idx] = 1.0
        return features
    else:
        return torch.full((n, 1), 2.0, device=device)


def get_num_labels(graphs: dict, dataset_name: str = "") -> int:
    """Determine the number of label dimensions from a graph collection.

    Args:
        graphs: Dictionary of graph_id -> networkx.Graph.
        dataset_name: Dataset name hint ("AIDS" has 29 labels).

    Returns:
        Number of one-hot dimensions.
    """
    if dataset_name.upper() == "AIDS":
        return 29

    sample = list(graphs.values())[0]
    nodes = sorted(sample.nodes())
    if nodes and all("label" in sample.nodes[n] for n in nodes):
        labels = set()
        for g in graphs.values():
            for node in g.nodes():
                if "label" in g.nodes[node]:
                    labels.add(g.nodes[node]["label"])
        return len(labels)

    return 1


def get_label_map(dataset_name: str = "") -> Optional[dict[str, int]]:
    """Get label-to-index mapping for a dataset.

    Args:
        dataset_name: Dataset name (e.g. "AIDS").

    Returns:
        Dictionary mapping label strings to indices, or None.
    """
    m = _get_label_map_cached(dataset_name)
    return m if m else None
