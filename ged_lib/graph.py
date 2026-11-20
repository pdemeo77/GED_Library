"""Graph I/O: CSV loading and conversion to networkx.Graph."""

import ast
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np


def _load_label_map_for_graph(dataset_name: str = "") -> dict[str, int]:
    """Load label-to-index mapping from labels.json if available."""
    if dataset_name.upper() == "AIDS":
        label_path = Path(__file__).resolve().parent.parent / "Dataset" / "AIDS" / "labels.json"
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


@lru_cache(maxsize=8)
def _get_label_map_cached_graph(dataset_name: str) -> dict[str, int]:
    """Cached version of _load_label_map_for_graph."""
    return _load_label_map_for_graph(dataset_name)


def _resolve_label(label, num_labels: int, label_map: Optional[dict[str, int]] = None) -> int:
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
        return -1
    if isinstance(label, (int, np.integer)):
        return int(label) if 0 <= int(label) < num_labels else -1
    return -1


def csv_row_to_graph(
    edge_list_str: str,
    node_labels: Optional[list[str]] = None,
) -> nx.Graph:
    """Convert a CSV row representation to a networkx Graph.

    Args:
        edge_list_str: String representation of edge list, e.g. "[[0, 1], [1, 2]]".
        node_labels: Optional list of label strings for each node.

    Returns:
        A networkx.Graph with optional 'label' attributes on nodes.
    """
    edges = ast.literal_eval(edge_list_str)
    G = nx.Graph()

    if node_labels:
        for i, label in enumerate(node_labels):
            G.add_node(i, label=label)
    else:
        node_set = set()
        for u, v in edges:
            node_set.add(u)
            node_set.add(v)
        # Ensure all nodes in [0, max_id] are present (preserves isolated nodes)
        if node_set:
            max_id = max(node_set)
            for node in range(max_id + 1):
                G.add_node(node)

    G.add_edges_from(edges)
    return G


def load_graphs_csv(path: Path) -> dict[int, nx.Graph]:
    """Load graphs from a CSV file.

    Expects columns: graph_id, graph_edge_list[, node_labels]

    Args:
        path: Path to the CSV file.

    Returns:
        Dictionary mapping graph_id to networkx.Graph.
    """
    import csv

    graphs: dict[int, nx.Graph] = {}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_labels = "node_labels" in (reader.fieldnames or [])

        for row in reader:
            gid = int(row["graph_id"])
            edge_list_str = row["graph_edge_list"]
            node_labels = None

            if has_labels:
                node_labels = ast.literal_eval(row["node_labels"])

            graphs[gid] = csv_row_to_graph(edge_list_str, node_labels)

    return graphs


def load_ground_truth_csv(path: Path) -> list[tuple[int, int, float]]:
    """Load ground truth GED values from a CSV file.

    Expects columns: id_1, id_2, true_ged

    Args:
        path: Path to the CSV file.

    Returns:
        List of (id_1, id_2, true_ged) tuples.
    """
    import csv

    pairs: list[tuple[int, int, float]] = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            id1 = int(row["id_1"])
            id2 = int(row["id_2"])
            ged = float(row["true_ged"])
            pairs.append((id1, id2, ged))

    return pairs


def load_labels_json(path: Path) -> dict[str, int]:
    """Load label encoding from a JSON file.

    Args:
        path: Path to the labels.json file.

    Returns:
        Dictionary mapping label strings to integer indices.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def nx_to_tensors(
    g: nx.Graph,
    num_labels: Optional[int] = None,
    label_map: Optional[dict[str, int]] = None,
) -> dict:
    """Convert a networkx.Graph to a dict of numpy arrays for OT/neural solvers.

    Args:
        g: A networkx.Graph. Nodes may have a 'label' attribute (string).
        num_labels: Total number of distinct labels (for one-hot encoding).
            If None, labels are inferred from the graph. For unlabeled graphs,
            pass 0 or leave as None.
        label_map: Optional mapping from string labels to integer indices.

    Returns:
        Dictionary with keys:
            - "n": number of nodes
            - "m": number of edges
            - "edge_index": numpy array of shape (2, 2*E) with bidirectional edges
            - "adj": numpy adjacency matrix of shape (n, n)
            - "features": numpy array of shape (n, d) with node features
            - "has_labels": bool, whether the graph has node labels
    """
    nodes = sorted(g.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    edges = list(g.edges())
    m = len(edges)

    bidir_edges = []
    for u, v in edges:
        ui, vi = node_to_idx[u], node_to_idx[v]
        bidir_edges.append([ui, vi])
        bidir_edges.append([vi, ui])
    edge_index = (
        np.array(bidir_edges, dtype=np.int64).T if bidir_edges else np.zeros((2, 0), dtype=np.int64)
    )

    adj = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        ui, vi = node_to_idx[u], node_to_idx[v]
        adj[ui, vi] = 1.0
        adj[vi, ui] = 1.0

    has_labels = all("label" in g.nodes[n] for n in nodes)

    if has_labels and num_labels is not None and num_labels > 0:
        features = np.zeros((n, num_labels), dtype=np.float64)
        for node in nodes:
            label = g.nodes[node].get("label", 0)
            idx = node_to_idx[node]
            resolved = _resolve_label(label, num_labels, label_map)
            if 0 <= resolved < num_labels:
                features[idx, resolved] = 1.0
    elif has_labels:
        labels = [g.nodes[node].get("label", 0) for node in nodes]
        unique_labels = sorted(set(labels))
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        d = len(unique_labels)
        features = np.zeros((n, d), dtype=np.float64)
        for i, node in enumerate(nodes):
            label = g.nodes[node].get("label", 0)
            features[i, label_to_idx[label]] = 1.0
        has_labels = True
    else:
        features = np.full((n, 1), 2.0, dtype=np.float64)
        has_labels = False

    return {
        "n": n,
        "m": m,
        "edge_index": edge_index,
        "adj": adj,
        "features": features,
        "has_labels": has_labels,
    }


def nx_to_label_indices(g: nx.Graph) -> list[int]:
    """Extract integer label indices from a labeled networkx.Graph.

    For unlabeled graphs, returns a list of -1 for each node.

    Args:
        g: A networkx.Graph with optional 'label' node attributes.

    Returns:
        List of integer label indices, one per node (sorted by node order).
    """
    nodes = sorted(g.nodes())
    has_labels = all("label" in g.nodes[n] for n in nodes)
    if not has_labels:
        return [-1] * len(nodes)
    return [g.nodes[node].get("label", -1) for node in nodes]
