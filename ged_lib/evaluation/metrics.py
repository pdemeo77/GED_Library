"""Evaluation metrics: MAE, MSE, Spearman, Kendall, timing."""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ged_lib.base import GEDSolver

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for solver evaluation results."""

    solver_name: str
    dataset_name: str
    n_pairs: int = 0
    mae: float = 0.0
    mse: float = 0.0
    rmse: float = 0.0
    spearman_r: float = 0.0
    spearman_p: float = 1.0
    kendall_tau: float = 0.0
    kendall_p: float = 1.0
    mean_time_s: float = 0.0
    std_time_s: float = 0.0
    timeout_rate: float = 0.0
    error_rate: float = 0.0
    predicted: np.ndarray = field(default_factory=lambda: np.array([]))
    true_values: np.ndarray = field(default_factory=lambda: np.array([]))
    times: np.ndarray = field(default_factory=lambda: np.array([]))

    def __str__(self) -> str:
        return (
            f"EvaluationResult(solver={self.solver_name}, dataset={self.dataset_name}, "
            f"n_pairs={self.n_pairs}, MAE={self.mae:.4f}, RMSE={self.rmse:.4f}, "
            f"Spearman rho={self.spearman_r:.4f}, Kendall tau={self.kendall_tau:.4f}, "
            f"mean_time={self.mean_time_s:.4f}s)"
        )

    def to_dict(self) -> dict:
        """Convert to a flat dictionary (excludes arrays)."""
        return {
            "solver_name": self.solver_name,
            "dataset_name": self.dataset_name,
            "n_pairs": self.n_pairs,
            "mae": self.mae,
            "mse": self.mse,
            "rmse": self.rmse,
            "spearman_r": self.spearman_r,
            "spearman_p": self.spearman_p,
            "kendall_tau": self.kendall_tau,
            "kendall_p": self.kendall_p,
            "mean_time_s": self.mean_time_s,
            "std_time_s": self.std_time_s,
            "timeout_rate": self.timeout_rate,
            "error_rate": self.error_rate,
        }


def evaluate_solver(
    solver: GEDSolver,
    graphs: dict,
    ground_truth: list[tuple[int, int, float]],
    dataset_name: str = "unknown",
    max_pairs: Optional[int] = None,
    timeout: Optional[float] = None,
) -> EvaluationResult:
    """Evaluate a GED solver against ground truth.

    Args:
        solver: A GEDSolver instance.
        graphs: Dictionary mapping graph_id to networkx.Graph.
        ground_truth: List of (id_1, id_2, true_ged) tuples.
        dataset_name: Name of the dataset for reporting.
        max_pairs: Maximum number of pairs to evaluate (None = all).
        timeout: Timeout in seconds per pair (None = no timeout).

    Returns:
        An EvaluationResult with computed metrics.
    """
    from scipy.stats import kendalltau as _kendalltau
    from scipy.stats import spearmanr as _spearmanr

    pairs = ground_truth[:max_pairs] if max_pairs else ground_truth

    predicted: list[float] = []
    true_values: list[float] = []
    times_list: list[float] = []
    n_timeouts = 0
    n_errors = 0

    for id1, id2, true_ged in pairs:
        g1 = graphs.get(id1)
        g2 = graphs.get(id2)

        if g1 is None or g2 is None:
            continue

        start = time.perf_counter()

        try:
            if timeout is not None:
                pred_ged = _compute_with_timeout_thread(solver, g1, g2, timeout)
            else:
                pred_ged = solver.compute(g1, g2)
        except Exception:
            n_errors += 1
            logger.warning(
                "Solver %r raised on pair (%s, %s); recorded as error.",
                solver.name,
                id1,
                id2,
                exc_info=True,
            )
            continue

        elapsed = time.perf_counter() - start

        if pred_ged is None:
            n_timeouts += 1
            continue

        predicted.append(float(pred_ged))
        true_values.append(float(true_ged))
        times_list.append(elapsed)

    pred_arr = np.array(predicted, dtype=np.float64)
    true_arr = np.array(true_values, dtype=np.float64)
    times_arr = np.array(times_list, dtype=np.float64)
    n_evaluated = len(predicted)

    if n_evaluated == 0:
        return EvaluationResult(
            solver_name=solver.name,
            dataset_name=dataset_name,
            timeout_rate=1.0,
            error_rate=n_errors / len(pairs) if len(pairs) > 0 else 0.0,
        )

    errors = pred_arr - true_arr
    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors**2))
    rmse = float(np.sqrt(mse))

    spearman_r: float = 0.0
    spearman_p: float = 1.0
    kendall_tau: float = 0.0
    kendall_p: float = 1.0

    if n_evaluated >= 2 and np.std(pred_arr) > 0 and np.std(true_arr) > 0:
        sp_result = _spearmanr(pred_arr, true_arr)
        spearman_r = float(sp_result.statistic)
        spearman_p = float(sp_result.pvalue)
        kt_result = _kendalltau(pred_arr, true_arr)
        kendall_tau = float(kt_result.statistic)
        kendall_p = float(kt_result.pvalue)

    return EvaluationResult(
        solver_name=solver.name,
        dataset_name=dataset_name,
        n_pairs=n_evaluated,
        mae=mae,
        mse=mse,
        rmse=rmse,
        spearman_r=spearman_r,
        spearman_p=spearman_p,
        kendall_tau=kendall_tau,
        kendall_p=kendall_p,
        mean_time_s=float(np.mean(times_arr)),
        std_time_s=float(np.std(times_arr)),
        timeout_rate=n_timeouts / len(pairs) if len(pairs) > 0 else 0.0,
        error_rate=n_errors / len(pairs) if len(pairs) > 0 else 0.0,
        predicted=pred_arr,
        true_values=true_arr,
        times=times_arr,
    )


def _compute_with_timeout_thread(solver: GEDSolver, g1, g2, timeout: float) -> Optional[float]:
    """Compute GED with timeout using threading (works on all platforms)."""
    from threading import Thread
    import queue

    result_queue: queue.Queue = queue.Queue()

    def _worker():
        try:
            result_queue.put(solver.compute(g1, g2))
        except Exception:
            result_queue.put(None)

    thread = Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive() or result_queue.empty():
        return None
    return result_queue.get()
