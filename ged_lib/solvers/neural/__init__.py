"""Neural GED solvers based on the GED-via-Optimal-Transport framework.

Reference:
    Cheng, Q. et al. (2025).
    Computing Approximate Graph Edit Distance via Optimal Transport.
    SIGMOD'25.

This subpackage contains neural network-based GED solvers that require
PyTorch and optionally torch_geometric. They are registered as solvers
in the main solver registry when the dependencies are available.
"""

from ged_lib.solvers.neural.layers import (
    AttentionModule,
    TensorNetworkModule,
    OTLayer,
)
from ged_lib.solvers.neural.ged_matrix import CostMatrixModule, GedMatrixModule

__all__ = [
    "AttentionModule",
    "TensorNetworkModule",
    "OTLayer",
    "CostMatrixModule",
    "GedMatrixModule",
]
