"""Solver registry for GED_Library.

All solver implementations should register themselves here so they can be
discovered via ``list_solvers()`` and instantiated by name.
"""

from typing import Callable

from ged_lib.base import GEDSolver

SOLVER_REGISTRY: dict[str, type[GEDSolver]] = {}


def register_solver(name: str) -> Callable[[type[GEDSolver]], type[GEDSolver]]:
    """Decorator to register a solver class in the global registry.

    Args:
        name: The string identifier for this solver.

    Example:
        @register_solver("bipartite")
        class BPSolver(GEDSolver):
            ...
    """

    def decorator(cls: type[GEDSolver]) -> type[GEDSolver]:
        SOLVER_REGISTRY[name] = cls
        return cls

    return decorator


from ged_lib.solvers.hungarian import HungarianSolver
from ged_lib.solvers.gedgw import GEDGWSolver

try:
    from ged_lib.solvers.neural.models import GEDIOTSolver, GedGNNSolver, SimGNNSolver
except ImportError:
    pass

try:
    from ged_lib.solvers.neural.egsc import EGSCSolver
except ImportError:
    pass

try:
    from ged_lib.solvers.neural.greed import GREEDSolver
except ImportError:
    pass

try:
    from ged_lib.solvers.neural.gedgnn_full import GEDGNNFullSolver
except ImportError:
    pass

try:
    from ged_lib.solvers.neural.grasp import GraspSolver
except ImportError:
    pass

try:
    from ged_lib.solvers.fgwalign import FGWAlignSolver
except ImportError:
    pass

try:
    from ged_lib.solvers.sf.gedsf_solver import GEDSFSolver
except ImportError:
    pass
