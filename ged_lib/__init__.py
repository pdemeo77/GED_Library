"""GED_Library — Graph Edit Distance approximation library.

Provides a scikit-learn-like uniform interface to multiple GED algorithms.

Example:
    >>> from ged_lib import HungarianSolver, load_dataset
    >>> graphs, ground_truth = load_dataset("AIDS")
    >>> solver = HungarianSolver()
    >>> ged = solver.compute(graphs[0], graphs[1])
"""

from ged_lib.base import GEDSolver
from ged_lib.cost import CostModel
from ged_lib.dataset.loader import load_dataset
from ged_lib.evaluation.metrics import evaluate_solver

__all__ = [
    "GEDSolver",
    "CostModel",
    "load_dataset",
    "evaluate_solver",
    "list_solvers",
    "set_seed",
    "HungarianSolver",
    "HEDSolver",
    "GEDGWSolver",
    "GEDIOTSolver",
    "GedGNNSolver",
    "SimGNNSolver",
    "TaGSimSolver",
    "GPNSolver",
    "EGSCSolver",
    "GREEDSolver",
    "GEDGNNFullSolver",
    "GraspSolver",
    "GEDSFSolver",
    "FGWAlignSolver",
]

__version__ = "0.1.0"


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all backends.

    Sets seeds for: Python hash, NumPy, PyTorch CPU, PyTorch CUDA,
    and enables deterministic CUDNN mode.

    Args:
        seed: Integer seed value.
    """
    import os

    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def list_solvers() -> dict[str, type[GEDSolver]]:
    """Return a dictionary of all registered solver names to their classes."""
    from ged_lib.solvers import SOLVER_REGISTRY

    return dict(SOLVER_REGISTRY)


from ged_lib.solvers.hungarian import HungarianSolver
from ged_lib.solvers.hed import HEDSolver
from ged_lib.solvers.gedgw import GEDGWSolver

try:
    from ged_lib.solvers.neural.models import (
        GEDIOTSolver,
        GedGNNSolver,
        SimGNNSolver,
        TaGSimSolver,
        GPNSolver,
    )
except ImportError:
    GEDIOTSolver = None
    GedGNNSolver = None
    SimGNNSolver = None
    TaGSimSolver = None
    GPNSolver = None

try:
    from ged_lib.solvers.neural.egsc import EGSCSolver
except ImportError:
    EGSCSolver = None

try:
    from ged_lib.solvers.neural.greed import GREEDSolver
except ImportError:
    GREEDSolver = None

try:
    from ged_lib.solvers.neural.gedgnn_full import GEDGNNFullSolver
except ImportError:
    GEDGNNFullSolver = None

try:
    from ged_lib.solvers.neural.grasp import GraspSolver
except ImportError:
    GraspSolver = None

try:
    from ged_lib.solvers.sf.gedsf_solver import GEDSFSolver
except ImportError:
    GEDSFSolver = None

try:
    from ged_lib.solvers.fgwalign import FGWAlignSolver
except ImportError:
    FGWAlignSolver = None
