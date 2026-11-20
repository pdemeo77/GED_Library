"""Core FGWAlign functions copied verbatim from the upstream repository.

Source: https://github.com/squareRoot3/FGWAlign
License: MIT (see upstream repository)

Citation:
    Tang, Z. et al. (2024). Fast Graph Edit Distance via Optimal Transport
    with Graph Edit Distance Lower Bound.

Only the three single-relation functions are included:
    - greedy_assignment
    - fusedGW_solver
    - FGWAlign

Multi-relational variants (fusedGW_multirel_solver, FGWAlign_multirel,
pad_adj_list) are intentionally omitted as they are not required for
single-relation molecular graphs.
"""

import logging
import random
from typing import List, Optional, Tuple

import numpy as np
import ot
import torch

# Suppress per-pair INFO spam during bulk evaluation
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def greedy_assignment(cost_matrix: List[List[float]], eps: float) -> Tuple[torch.Tensor, float]:
    """Implements an optimized greedy algorithm with an eps threshold for one-to-one
    assignment using PyTorch on CPU.

    When the cost is below eps, it directly matches the remaining rows and columns
    randomly to maximize the total cost.

    Parameters
    ----------
    cost_matrix : List[List[float]]
        An n x n cost matrix.
    eps : float
        Threshold; costs below this value will be randomly assigned.

    Returns
    -------
    Tuple[torch.Tensor, float]
        assignment_matrix : An n x n assignment matrix where 1 indicates assignment
            and 0 indicates no assignment.
        total_cost : The total cost of the assignments.
    """
    device = torch.device("cpu")
    cost_tensor = torch.tensor(cost_matrix, dtype=torch.float32, device=device)

    n = cost_tensor.size(0)

    # Flatten the cost matrix to (cost, row, col)
    cost_flat = cost_tensor.view(-1)
    indices = torch.arange(n * n, device=device)

    rows = indices // n
    cols = indices % n

    # Categorize high-cost and low-cost assignments
    high_cost_mask = cost_flat >= eps
    high_cost_costs = cost_flat[high_cost_mask]
    high_cost_rows = rows[high_cost_mask]
    high_cost_cols = cols[high_cost_mask]

    if high_cost_costs.numel() == 0:
        logger.warning("No high-cost assignments found with the given eps threshold.")

    # Sort high-cost assignments in descending order (greedy)
    sorted_high_cost, sorted_high_indices = torch.sort(high_cost_costs, descending=True)
    sorted_high_cost_rows = high_cost_rows[sorted_high_indices]
    sorted_high_cost_cols = high_cost_cols[sorted_high_indices]

    # Initialize assignment flags
    assigned_rows = torch.zeros(n, dtype=torch.bool, device=device)
    assigned_cols = torch.zeros(n, dtype=torch.bool, device=device)

    assignment = []
    total_cost = 0.0

    # First handle high-cost assignments (greedy selection)
    for row, col, cost in zip(
        sorted_high_cost_rows.tolist(),
        sorted_high_cost_cols.tolist(),
        sorted_high_cost.tolist(),
    ):
        if not assigned_rows[row] and not assigned_cols[col]:
            assignment.append((row, col, cost))
            assigned_rows[row] = True
            assigned_cols[col] = True
            total_cost += cost
            if len(assignment) == n:
                break

    # If assignments are not complete, handle low-cost assignments (random matching)
    remaining_rows = torch.nonzero(~assigned_rows, as_tuple=False).squeeze().tolist()
    remaining_cols = torch.nonzero(~assigned_cols, as_tuple=False).squeeze().tolist()

    # Ensure that remaining_rows and remaining_cols are lists
    if isinstance(remaining_rows, int):
        remaining_rows = [remaining_rows]
    if isinstance(remaining_cols, int):
        remaining_cols = [remaining_cols]

    if remaining_rows and remaining_cols:
        random.shuffle(remaining_rows)
        random.shuffle(remaining_cols)

        # Ensure the number of remaining rows and columns are equal
        min_len = min(len(remaining_rows), len(remaining_cols))
        for i in range(min_len):
            row = remaining_rows[i]
            col = remaining_cols[i]
            cost = cost_tensor[row, col].item()
            assignment.append((row, col, cost))
            total_cost += cost
            if len(assignment) == n:
                break

    # Construct the assignment matrix
    assignment_matrix = torch.zeros((n, n), dtype=torch.int32, device=device)
    for row, col, _ in assignment:
        assignment_matrix[row, col] = 1

    logger.info(f"Total assignments made: {len(assignment)} with total cost: {total_cost}")
    return assignment_matrix, total_cost


def fusedGW_solver(
    cost_s: torch.Tensor,
    cost_t: torch.Tensor,
    cost_st: Optional[torch.Tensor] = None,
    p_s: Optional[torch.Tensor] = None,
    p_t: Optional[torch.Tensor] = None,
    trans0: Optional[torch.Tensor] = None,
    error_bound: float = 1e-5,
    alpha: float = 1,
    sparse: bool = False,
    light: bool = False,
) -> torch.Tensor:
    """Solves the Fused Gromov-Wasserstein problem.

    Parameters
    ----------
    cost_s : torch.Tensor
        Cost matrix for the source graph.
    cost_t : torch.Tensor
        Cost matrix for the target graph.
    cost_st : Optional[torch.Tensor], optional
        Cost matrix between node features (labels), by default None.
    p_s : Optional[torch.Tensor], optional
        Source distribution, by default None.
    p_t : Optional[torch.Tensor], optional
        Target distribution, by default None.
    trans0 : Optional[torch.Tensor], optional
        Initial transport plan, by default None.
    error_bound : float, optional
        Convergence threshold, by default 1e-5.
    alpha : float, optional
        Balance parameter for the label cost, by default 1.
    sparse : bool, optional
        Whether to use sparse matrices, by default False.
    light : bool, optional
        If True, use lighter computations, by default False.

    Returns
    -------
    torch.Tensor
        The optimal transport plan.
    """
    n1, n2 = cost_s.shape[0], cost_t.shape[0]
    device = cost_s.device

    if sparse:
        cost_s = cost_s.to_sparse()
        cost_t = cost_t.to_sparse()

    p_s = torch.ones(n1, 1, device=device) / n1 if p_s is None else p_s
    p_t = torch.ones(n2, 1, device=device) / n2 if p_t is None else p_t
    trans0 = p_s @ p_t.T if trans0 is None else trans0

    outer_iter = min(max(200, 10 * n1), 2000)
    inner_iter = 10

    beta = 0.1 if n1 < 100 else 0.01
    if light:
        outer_iter, inner_iter, beta = 20, 1, 0.1

    if not sparse:
        cost_s2 = 1 - cost_s
        cost_t2 = 1 - cost_t

    for oi in range(outer_iter):
        a = torch.ones_like(p_s) / p_s.shape[0]

        # Compute the cost tensor
        cost = -cost_t @ (cost_s @ trans0).T
        if not sparse:
            cost -= cost_t2 @ (cost_s2 @ trans0).T

        cost = cost.T

        if cost_st is not None:
            cost += (alpha / n1) * cost_st

        kernel = torch.exp(-cost / beta) * trans0

        for ii in range(inner_iter):
            denominator = kernel.T @ a
            # Prevent division by zero
            denominator = torch.where(
                denominator == 0, torch.ones_like(denominator), denominator
            )
            b = p_t / denominator

            denominator = kernel @ b
            denominator = torch.where(
                denominator == 0, torch.ones_like(denominator), denominator
            )
            a = p_s / denominator

        trans = (a @ b.T) * kernel
        relative_error = torch.norm(trans - trans0) / torch.norm(trans0)

        if relative_error < error_bound or (trans > 0.6 / n2).sum() == n2:
            break

        trans0 = trans

    return trans


def FGWAlign(
    g1_adj: torch.Tensor,
    g2_adj: torch.Tensor,
    g1_labels: Optional[torch.Tensor] = None,
    g2_labels: Optional[torch.Tensor] = None,
    patience: int = 15,
    topk: int = 5,
    alpha: float = 1,
    sparse: bool = False,
    light: bool = False,
    device: str = "cpu",
) -> Tuple[float, torch.Tensor]:
    """Computes the Graph Edit Distance (GED) between two graphs via Optimal Transport.

    Parameters
    ----------
    g1_adj : torch.Tensor
        Adjacency matrix of the first graph.
    g2_adj : torch.Tensor
        Adjacency matrix of the second graph.
    g1_labels : Optional[torch.Tensor], optional
        Node labels in the first graph, by default None.
    g2_labels : Optional[torch.Tensor], optional
        Node labels in the second graph, by default None.
    patience : int, optional
        Number of trials without improvement before stopping, by default 15.
    topk : int, optional
        Number of top assignments to consider, by default 5.
    alpha : float, optional
        Balance parameter for the label cost, by default 1.
    sparse : bool, optional
        Whether to use sparse matrices for the FGW solver, by default False.
    light : bool, optional
        If True, use lighter computations, by default False.
    device : str, optional
        Device to run the solver ('cpu' or 'cuda'), by default 'cpu'.

    Returns
    -------
    Tuple[float, torch.Tensor]
        min_ged : Minimum Graph Edit Distance found.
        optimal_align_mat : Optimal transport plan (alignment matrix).
    """
    device = torch.device(device)
    n1, n2 = g1_adj.shape[0], g2_adj.shape[0]
    n = max(n1, n2)
    trial_num = 1 if patience == 1 else 1000

    # Pad adjacency matrices to have the same size
    g1_adj_padded = torch.nn.functional.pad(
        g1_adj, (0, n - n1, 0, n - n1), mode="constant", value=0
    )
    g2_adj_padded = torch.nn.functional.pad(
        g2_adj, (0, n - n2, 0, n - n2), mode="constant", value=0
    )

    # Handle node labels
    if g1_labels is not None and g2_labels is not None:
        g1_labels_padded = torch.nn.functional.pad(
            g1_labels, (0, n - n1), mode="constant", value=-1
        )
        g2_labels_padded = torch.nn.functional.pad(
            g2_labels, (0, n - n2), mode="constant", value=-1
        )
        label_cost = (
            (g1_labels_padded.unsqueeze(1) != g2_labels_padded.unsqueeze(0)).float().to(device)
        )
    else:
        label_cost = None

    g1_adj_padded, g2_adj_padded = g1_adj_padded.to(device), g2_adj_padded.to(device)
    optimal_align_mat = None
    p = q = torch.ones(n, device=device) / n
    min_ged = float("inf")
    flag, pa, i = 0, 0, 0
    simple_LB = n - min(n1, n2) + abs(
        g1_adj_padded.nonzero().size(0) - g2_adj_padded.nonzero().size(0)
    ) // 2

    while i < trial_num:
        init_guess = None
        if i > 0:
            if n < 1000:
                init_guess_np = ot.sinkhorn(
                    p.cpu().numpy(),
                    q.cpu().numpy(),
                    torch.randn(n, n, device=device).cpu().numpy(),
                    reg=1,
                    numItermax=20,
                )
            else:
                init_guess_np = ot.sinkhorn(
                    p.cpu().numpy(),
                    q.cpu().numpy(),
                    1
                    - optimal_align_mat.to_dense().cpu().numpy()
                    + torch.randn(n, n, device=device).cpu().numpy(),
                    reg=1,
                    numItermax=10,
                )
            init_guess = torch.tensor(init_guess_np, dtype=torch.float32, device=device)

        pi = fusedGW_solver(
            cost_s=g1_adj_padded,
            cost_t=g2_adj_padded,
            cost_st=label_cost,
            trans0=init_guess,
            sparse=sparse,
            light=light,
            alpha=alpha,
        )

        for j in range(topk):
            trans = pi.clone()

            if (trans > (0.5 / n)).sum() < n:
                flag += 1
                if sparse and n > 1000:
                    trans, _ = greedy_assignment(pi.cpu().tolist(), 1 / (n * np.log(n)))
                    trans = trans.to(device)
                try:
                    trans_lp = ot.emd(p.cpu().numpy(), q.cpu().numpy(), -pi.cpu().numpy())
                    trans = torch.tensor(trans_lp, dtype=torch.float32, device=device)
                except Exception as e:
                    logger.error(f"EMD failed on trial {i}, topk {j}: {e}")
                    flag += 100
                    break

            trans = (trans > (0.5 / n)).float().to(device)
            if trans.sum() != n:
                # Not a feasible answer, try again.
                flag += 100
                break

            if sparse:
                trans = torch.sparse_coo_tensor(
                    trans.nonzero().t(),
                    torch.ones(trans.nonzero().size(0), device=device),
                    size=(n, n),
                    device=device,
                )

            # Compute Graph Edit Distance
            pre_ged = (
                torch.triu(trans.T @ g1_adj_padded @ trans != g2_adj_padded)
            ).sum().item()
            if label_cost is not None:
                pre_ged += (label_cost * trans).sum().item()
            else:
                pre_ged += n - min(n1, n2)

            if pre_ged < min_ged:
                pa = 0
                min_ged = pre_ged
                optimal_align_mat = trans
            pa += 1
            pi -= trans / (n**2)  # Add penalty based on the previous solution

        if min_ged <= simple_LB or pa / topk >= patience:
            break
        i += 0.01 if min_ged > 1e6 else 1

    logger.info(f"Final GED: {min_ged}.")
    logger.debug(f"Flags: {flag}, Patience count: {pa}, Trials conducted: {i + 1}")

    return min_ged, optimal_align_mat
