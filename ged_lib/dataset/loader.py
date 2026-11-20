"""Dataset loader: load AIDS, IMDB, and Linux benchmark datasets."""

from pathlib import Path
from typing import Optional

import networkx as nx

from ged_lib.graph import load_graphs_csv, load_ground_truth_csv, load_labels_json

DATASETS = {"AIDS": "AIDS", "IMDB": "IMDB", "LINUX": "Linux"}

# Resolve paths relative to the project root (parent of ged_lib/)
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_name(name: str) -> str:
    """Resolve a case-insensitive dataset name to the actual directory name."""
    key = name.upper()
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {sorted(DATASETS.values())}")
    return DATASETS[key]


def load_dataset(
    name: str,
    graph_root: Optional[str] = None,
    ged_root: Optional[str] = None,
) -> tuple[dict[int, nx.Graph], list[tuple[int, int, float]]]:
    """Load a benchmark dataset with graphs and ground truth GED values.

    Args:
        name: Dataset name. One of "AIDS", "IMDB", "Linux" (case-insensitive).
        graph_root: Root directory containing graph CSV files.
            Defaults to ``<project_root>/Dataset/``.
        ged_root: Root directory containing ground truth CSV files.
            Defaults to ``<project_root>/True_GED/``.

    Returns:
        A tuple of:
            - Dictionary mapping graph_id to networkx.Graph
            - List of (id_1, id_2, true_ged) tuples

    Raises:
        ValueError: If the dataset name is not recognized.
        FileNotFoundError: If the required CSV files are not found.
    """
    resolved = _resolve_name(name)

    if graph_root is None:
        graph_root = str(_PKG_ROOT / "Dataset")
    if ged_root is None:
        ged_root = str(_PKG_ROOT / "True_GED")

    graph_path = Path(graph_root) / resolved / f"{resolved}_graphs.csv"
    ged_path = Path(ged_root) / resolved / f"{resolved}_ged.csv"

    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    if not ged_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {ged_path}")

    graphs = load_graphs_csv(graph_path)
    ground_truth = load_ground_truth_csv(ged_path)

    return graphs, ground_truth


def load_labels(name: str, root: Optional[str] = None) -> Optional[dict[str, int]]:
    """Load label encoding for a dataset, if available.

    Args:
        name: Dataset name.
        root: Root directory containing label files.
            Defaults to ``<project_root>/Dataset/``.

    Returns:
        Dictionary mapping label strings to indices, or None if no labels file exists.
    """
    if root is None:
        root = str(_PKG_ROOT / "Dataset")
    label_path = Path(root) / _resolve_name(name) / "labels.json"
    if label_path.exists():
        return load_labels_json(label_path)
    return None
