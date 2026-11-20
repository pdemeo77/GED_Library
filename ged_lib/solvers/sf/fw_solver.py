"""Frank-Wolfe solver for Graph Edit Distance approximation.

Ported from GED-SF/ablation_linear_surrogate_simplified.py ``solve_ged_fw()``.

Two formulations are supported:

- **Linear**: Uses a static linear surrogate for the structural cost gradient.
  Fast but less adaptive.  The gradient term is pre-computed once as a pairwise
  squared Euclidean distance matrix Lambda, so each FW iteration is cheap.

- **Quadratic**: Uses the exact quadratic structural cost gradient.  More
  accurate but slower because the gradient depends on the current coupling Pi
  and must be recomputed at each iteration.

Algorithm outline:
    1. Precompute row-norm vectors V1, V2 from adjacency matrices.
    2. Initialise coupling Pi = (1/n) * ones.
    3. For each iteration:
       a. Compute GW gradient from adjacency structure.
       b. Add structural gradient (linear or quadratic).
       c. Solve a linear assignment problem (Hungarian) to find a permutation S.
       d. FW update: Pi = (1-gamma)*Pi + gamma*S,  gamma = 2/(t+2).
    4. Return the final coupling matrix Pi.

References:
    Frank, M., & Wolfe, P. (1956). An algorithm for quadratic programming.
    Naval Research Logistics Quarterly, 3(1-2), 95-110.
"""

import time
from typing import Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def solve_ged_fw(
    A1: np.ndarray,
    A2: np.ndarray,
    F1: np.ndarray,
    F2: np.ndarray,
    mu: float,
    max_iter: int = 150,
    formulation: str = "linear",
) -> Tuple[np.ndarray, float]:
    """Solve GED alignment via Frank-Wolfe optimisation.

    Args:
        A1: Adjacency matrix of graph 1, shape (n, n).
        A2: Adjacency matrix of graph 2, shape (n, n).
        F1: Structural feature matrix of graph 1, shape (n, d).
        F2: Structural feature matrix of graph 2, shape (n, d).
        mu: Weight for the structural regularisation term.
        max_iter: Maximum number of Frank-Wolfe iterations.
        formulation: ``'linear'`` or ``'quadratic'``.

    Returns:
        Tuple of (Pi, elapsed_seconds) where Pi is the n x n coupling matrix.

    Raises:
        ValueError: If formulation is not ``'linear'`` or ``'quadratic'``.
    """
    if formulation not in ("linear", "quadratic"):
        raise ValueError(f"formulation must be 'linear' or 'quadratic', got '{formulation}'")

    n = A1.shape[0]

    # --- Precompute GW gradient constants (independent of Pi) ---
    V1 = np.sum(A1**2, axis=1).reshape(-1, 1)  # (n, 1)
    V2 = np.sum(A2**2, axis=1).reshape(-1, 1)  # (n, 1)
    ones_col = np.ones((n, 1))

    # Initialise coupling as uniform
    Pi = np.ones((n, n)) / n

    # --- Precompute structural gradient terms ---
    if formulation == "linear":
        # Static pairwise squared Euclidean distance: Lambda[i,j] = ||F1[i]-F2[j]||^2
        Lambda = np.sum(F1**2, axis=1, keepdims=True) + np.sum(F2**2, axis=1) - 2.0 * (F1 @ F2.T)
    else:
        # Quadratic: gradient depends on Pi via M_quad and C_quad_const
        M_quad = F2 @ F2.T  # (n, n)
        C_quad_const = F1 @ F2.T  # (n, n)

    # --- Frank-Wolfe iterations ---
    start_time = time.time()

    for t in range(max_iter):
        # GW gradient: V1*1^T + 1*V2^T - 2*A1*Pi*A2^T
        term_A = (V1 @ ones_col.T) + (ones_col @ V2.T)
        term_B = 2.0 * (A1 @ Pi @ A2.T)
        grad_edge = term_A - term_B

        # Structural gradient
        if formulation == "linear":
            grad_struct = mu * Lambda
        else:
            grad_struct = 2.0 * mu * ((Pi @ M_quad) - C_quad_const)

        # Full gradient
        full_grad = grad_edge + grad_struct

        # Solve linear assignment (Hungarian)
        row_ind, col_ind = linear_sum_assignment(full_grad)

        # Build permutation matrix S
        S = np.zeros((n, n))
        S[row_ind, col_ind] = 1.0

        # FW update: Pi = (1-gamma)*Pi + gamma*S
        gamma = 2.0 / (t + 2)
        Pi = (1.0 - gamma) * Pi + gamma * S

    elapsed = time.time() - start_time
    return Pi, elapsed


def compute_fw_objective(
    Pi: np.ndarray,
    A1: np.ndarray,
    A2: np.ndarray,
    F1: np.ndarray,
    F2: np.ndarray,
    mu: float,
    formulation: str,
) -> float:
    """Compute the Frank-Wolfe objective value f(Pi).

    f(Pi) = edge_cost(Pi) + mu * struct_cost(Pi)

    Args:
        Pi: Coupling matrix, shape (n, n).
        A1, A2: Adjacency matrices, shape (n, n).
        F1, F2: Structural feature matrices, shape (n, d).
        mu: Structural regularisation weight.
        formulation: ``'linear'`` or ``'quadratic'``.

    Returns:
        Scalar objective value.
    """
    p = Pi.sum(axis=1)
    q = Pi.sum(axis=0)

    # Edge cost: p^T (A1^2) p + q^T (A2^2) q - 2*tr(A1*Pi*A2*Pi^T)
    edge_cost = p @ (A1**2) @ p + q @ (A2**2) @ q - 2.0 * np.trace(A1 @ Pi @ A2 @ Pi.T)

    # Structural cost
    if formulation == "linear":
        Lambda = np.sum(F1**2, axis=1, keepdims=True) + np.sum(F2**2, axis=1) - 2.0 * (F1 @ F2.T)
        struct_cost = np.sum(Lambda * Pi)
    else:
        diff = F1 - Pi @ F2
        struct_cost = np.sum(diff**2)

    return float(edge_cost + mu * struct_cost)
