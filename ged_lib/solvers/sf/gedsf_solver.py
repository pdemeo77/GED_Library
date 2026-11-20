"""GEDSFSolver — GED approximation via Frank-Wolfe alignment + ML regression.

This solver follows a two-stage pipeline:

1. **Alignment**: Run a Frank-Wolfe solver to find an optimal node coupling
   matrix Pi between the two graphs.

2. **Prediction**: Extract 14 features (GW score + 8 OT features + 5 graph-level
   features) from the coupling and feed them into a trained ML regressor to
   predict the final GED value.

The regressor is trained via :meth:`fit`, which cross-validates over a grid
of mu values and regressor types (RandomForest, GradientBoosting, XGBoost,
LightGBM, Huber, and a fixed-weight ensemble).

References:
    Frank, M., & Wolfe, P. (1956). An algorithm for quadratic programming.
    Naval Research Logistics Quarterly, 3(1-2), 95-110.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from ged_lib.base import GEDSolver
from ged_lib.cost import CostModel
from ged_lib.solvers import register_solver
from ged_lib.solvers.sf.feature_extraction import (
    compute_cross_cost_matrix,
    compute_graph_level_features,
    compute_label_distance,
    compute_structural_features,
    equalize_graphs,
    extract_ot_features,
)
from ged_lib.solvers.sf.fw_solver import compute_fw_objective, solve_ged_fw

# Optional regressors — gracefully degrade if not installed
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None  # type: ignore[assignment,misc]

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Default hyperparameter grids
# ---------------------------------------------------------------------------

_PARAM_GRIDS: Dict[str, dict] = {
    "RandomForest": {
        "n_estimators": [100, 200, 500],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
    },
    "GradientBoosting": {
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5, 10],
    },
    "Huber": {
        "epsilon": [1.1, 1.35, 1.5],
        "alpha": [1e-4, 1e-3, 1e-2],
    },
}

_BASE_MODELS: Dict[str, object] = {
    "RandomForest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "Huber": HuberRegressor(max_iter=500),
}

if XGBRegressor is not None:
    _BASE_MODELS["XGBoost"] = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
    _PARAM_GRIDS["XGBoost"] = {
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5, 10],
    }

if LGBMRegressor is not None:
    _BASE_MODELS["LightGBM"] = LGBMRegressor(random_state=42, n_jobs=-1, verbosity=-1)
    _PARAM_GRIDS["LightGBM"] = {
        "n_estimators": [100, 200, 500],
        "learning_rate": [0.05, 0.1, 0.2],
        "num_leaves": [15, 31, 63],
    }

# Feature columns produced by _extract_features
_FEATURE_KEYS = (
    ["GW_Score"]
    + [
        "alignment_entropy",
        "alignment_confidence",
        "transport_cost",
        "marginal_balance",
        "coupling_sparsity",
        "max_coupling",
        "coupling_variance",
        "structural_mismatch",
    ]
    + ["size_diff", "edge_diff", "density_ratio", "avg_deg_diff", "size_ratio"]
)


# ---------------------------------------------------------------------------
# Helper: extract full feature vector for one pair
# ---------------------------------------------------------------------------


def _extract_features(
    G1: nx.Graph,
    G2: nx.Graph,
    mu: float,
    formulation: str,
    feature_config: Tuple[str, ...],
    max_iter_fw: int,
    max_graph_size: int,
    struct_feats_1: Optional[np.ndarray] = None,
    struct_feats_2: Optional[np.ndarray] = None,
) -> Optional[dict]:
    """Run the FW solver on a graph pair and extract the 14-dimensional feature dict.

    Returns None if the graphs are too large or an error occurs.
    """
    if G1.number_of_nodes() > max_graph_size or G2.number_of_nodes() > max_graph_size:
        return None

    try:
        # Graph-level features on original (un-padded) graphs
        graph_feats = compute_graph_level_features(G1, G2)

        # Equalise sizes (copies, no mutation)
        G1_eq, G2_eq, _swapped = equalize_graphs(G1, G2)
        n = G1_eq.number_of_nodes()

        # If graphs were swapped during equalization, swap pre-computed features too
        if _swapped:
            struct_feats_1, struct_feats_2 = struct_feats_2, struct_feats_1

        # Label distance matrix
        label_dist = compute_label_distance(G1_eq, G2_eq, n)

        # Structural features — use cache if available, else compute
        def _get_struct(cached: Optional[np.ndarray], G_eq: nx.Graph) -> np.ndarray:
            if cached is not None:
                n_raw = cached.shape[0]
                if n_raw < n:
                    return np.vstack([cached, np.zeros((n - n_raw, cached.shape[1]))])
                return cached[:n]
            return compute_structural_features(G_eq, feature_config)

        F1 = _get_struct(struct_feats_1, G1_eq)
        F2 = _get_struct(struct_feats_2, G2_eq)

        # Cross matrix for OT features
        struct_cross = compute_cross_cost_matrix(F1, F2)
        cross_matrix = label_dist + mu * struct_cross

        # Adjacency matrices
        A1 = nx.to_numpy_array(G1_eq)
        A2 = nx.to_numpy_array(G2_eq)

        # Run Frank-Wolfe solver
        Pi, _elapsed = solve_ged_fw(A1, A2, F1, F2, mu, max_iter_fw, formulation)

        # GW score (FW objective value)
        gw_score = compute_fw_objective(Pi, A1, A2, F1, F2, mu, formulation)

        # OT features
        ot_feats = extract_ot_features(Pi, cross_matrix, A1, A2)

        return {"GW_Score": gw_score, **ot_feats, **graph_feats}
    except Exception:
        return None


def _features_to_array(feats: dict) -> np.ndarray:
    """Convert a feature dict to a 1-D array in canonical order."""
    return np.array([feats[k] for k in _FEATURE_KEYS], dtype=np.float64)


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------


@register_solver("gedsf")
class GEDSFSolver(GEDSolver):
    """Graph Edit Distance via Frank-Wolfe alignment + ML regression.

    This solver requires a :meth:`fit` call before :meth:`compute` can be
    used.  During fitting it cross-validates over ``mu_range`` and multiple
    regressor types to find the best configuration.

    Args:
        formulation: ``'linear'`` or ``'quadratic'`` FW formulation.
        mu_range: Tuple of mu values to cross-validate over.
        max_iter_fw: Maximum Frank-Wolfe iterations.
        max_graph_size: Skip pairs where either graph exceeds this node count.
        feature_config: Structural features to compute per node.
        device: Device string (kept for API consistency; FW runs on CPU).
        cost_model: CostModel (unused — GED-SF uses its own cost model).
    """

    def __init__(
        self,
        formulation: str = "linear",
        mu_range: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 1.0),
        max_iter_fw: int = 150,
        max_graph_size: int = 100,
        feature_config: Tuple[str, ...] = ("Deg", "CC", "PR"),
        device: str = "cpu",
        cost_model: Optional[CostModel] = None,
    ) -> None:
        super().__init__(cost_model)
        self.formulation = formulation
        self.mu_range = mu_range
        self.max_iter_fw = max_iter_fw
        self.max_graph_size = max_graph_size
        self.feature_config = feature_config
        self.device = device

        # Trained state (populated by fit)
        self._scaler: Optional[StandardScaler] = None
        self._regressor = None
        self._best_mu: Optional[float] = None
        self._best_config: Optional[str] = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        graphs: Dict[int, nx.Graph],
        pairs: List[Tuple[int, int, float]],
    ) -> "GEDSFSolver":
        """Train the ML pipeline with cross-validation over mu x regressors.

        For each mu value, features are extracted for all training pairs.
        Then each regressor is tuned via 5-fold GridSearchCV.  The overall
        best (mu, regressor, hyperparameters) is selected by MAE.

        Args:
            graphs: Dictionary mapping graph IDs to ``nx.Graph`` objects.
            pairs: Training pairs as ``(id1, id2, true_ged)`` tuples.

        Returns:
            self (fitted solver).
        """
        # Cache structural features per graph (mu-independent)
        struct_cache: Dict[int, np.ndarray] = {}
        unique_ids: set = set()
        for id1, id2, _ in pairs:
            unique_ids.add(id1)
            unique_ids.add(id2)
        for gid in unique_ids:
            g = graphs.get(gid)
            if g is not None and g.number_of_nodes() <= self.max_graph_size:
                try:
                    struct_cache[gid] = compute_structural_features(g, self.feature_config)
                except Exception:
                    pass

        best_mae = float("inf")
        best_mu = self.mu_range[0]
        best_regressor_name = "RandomForest"
        best_estimator = None
        best_scaler = None
        best_X: Optional[np.ndarray] = None
        best_y: Optional[np.ndarray] = None

        for mu in self.mu_range:
            X_rows: list[np.ndarray] = []
            y_rows: list[float] = []

            for id1, id2, true_ged in pairs:
                g1 = graphs.get(id1)
                g2 = graphs.get(id2)
                if g1 is None or g2 is None:
                    continue

                feats = _extract_features(
                    g1,
                    g2,
                    mu,
                    self.formulation,
                    self.feature_config,
                    self.max_iter_fw,
                    self.max_graph_size,
                    struct_feats_1=struct_cache.get(id1),
                    struct_feats_2=struct_cache.get(id2),
                )
                if feats is not None:
                    X_rows.append(_features_to_array(feats))
                    y_rows.append(true_ged)

            if len(X_rows) < 20:
                warnings.warn(
                    f"Only {len(X_rows)} valid pairs for mu={mu} — skipping.",
                    stacklevel=2,
                )
                continue

            X = np.vstack(X_rows)
            y = np.array(y_rows, dtype=np.float64)

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val)

            # Store best predictions for ensemble
            best_preds: Dict[str, np.ndarray] = {}

            for model_name, base_model in _BASE_MODELS.items():
                grid = GridSearchCV(
                    clone(base_model),
                    _PARAM_GRIDS[model_name],
                    cv=5,
                    scoring="neg_mean_absolute_error",
                    n_jobs=-1,
                    refit=True,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    grid.fit(X_train_s, y_train)

                y_pred = grid.best_estimator_.predict(X_val_s)
                best_preds[model_name] = y_pred
                mae = float(np.mean(np.abs(y_pred - y_val)))

                if mae < best_mae:
                    best_mae = mae
                    best_mu = mu
                    best_regressor_name = model_name
                    best_estimator = grid.best_estimator_
                    best_scaler = scaler
                    best_X = X
                    best_y = y

            # Ensemble: 0.5*RF + 0.3*GB + 0.2*Huber
            if all(m in best_preds for m in ["RandomForest", "GradientBoosting", "Huber"]):
                y_ens = (
                    0.5 * best_preds["RandomForest"]
                    + 0.3 * best_preds["GradientBoosting"]
                    + 0.2 * best_preds["Huber"]
                )
                mae_ens = float(np.mean(np.abs(y_ens - y_val)))
                if mae_ens < best_mae:
                    best_mae = mae_ens
                    best_mu = mu
                    best_regressor_name = "Ensemble"
                    # For ensemble, store a simple wrapper
                    best_estimator = _EnsembleWrapper(
                        scaler,
                        X_train_s,  # not needed but kept for ref
                        y_train,
                    )
                    best_scaler = scaler
                    best_X = X
                    best_y = y

        if best_estimator is None:
            raise RuntimeError("No valid training configuration found.")

        self._scaler = best_scaler
        self._best_mu = best_mu
        self._best_config = best_regressor_name

        # For ensemble, we need to retrain on the full training data
        if best_regressor_name == "Ensemble":
            self._regressor = _EnsembleWrapper(best_scaler, None, None)
            self._regressor._train_full(best_X, best_y)
        else:
            # Retrain the best model on the full dataset
            X_full_s = best_scaler.transform(best_X)
            final_model = clone(_BASE_MODELS[best_regressor_name])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                final_model.fit(X_full_s, best_y)
            self._regressor = final_model

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def compute(self, g1: nx.Graph, g2: nx.Graph) -> float:
        """Predict GED using the trained pipeline.

        Args:
            g1: First graph.
            g2: Second graph.

        Returns:
            Predicted GED as a float (clipped to >= 0).

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if not self._is_fitted:
            raise RuntimeError("GEDSFSolver must be fit() before compute().")

        feats = _extract_features(
            g1,
            g2,
            self._best_mu,
            self.formulation,
            self.feature_config,
            self.max_iter_fw,
            self.max_graph_size,
        )
        if feats is None:
            raise RuntimeError(
                f"Feature extraction failed for graphs with "
                f"{g1.number_of_nodes()} and {g2.number_of_nodes()} nodes."
            )

        x = _features_to_array(feats).reshape(1, -1)
        x_s = self._scaler.transform(x)
        pred = float(self._regressor.predict(x_s)[0])
        return max(0.0, pred)

    def __repr__(self) -> str:
        state = "fitted" if self._is_fitted else "unfitted"
        return (
            f"GEDSFSolver(formulation={self.formulation!r}, mu={self._best_mu}, "
            f"config={self._best_config!r}, {state})"
        )


# ---------------------------------------------------------------------------
# Ensemble wrapper
# ---------------------------------------------------------------------------


class _EnsembleWrapper:
    """Fixed-weight ensemble: 0.5*RF + 0.3*GB + 0.2*Huber.

    Trained internally by GEDSFSolver.fit() when the ensemble wins.
    """

    def __init__(
        self,
        scaler: StandardScaler,
        X_dummy: Optional[np.ndarray],
        y_dummy: Optional[np.ndarray],
    ) -> None:
        self._scaler = scaler
        self._rf: Optional[RandomForestRegressor] = None
        self._gb: Optional[GradientBoostingRegressor] = None
        self._huber: Optional[HuberRegressor] = None

    def _train_full(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the three sub-models on the full dataset."""
        X_s = self._scaler.transform(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._rf = clone(_BASE_MODELS["RandomForest"])
            self._rf.fit(X_s, y)
            self._gb = clone(_BASE_MODELS["GradientBoosting"])
            self._gb.fit(X_s, y)
            self._huber = clone(_BASE_MODELS["Huber"])
            self._huber.fit(X_s, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return ensemble prediction."""
        return 0.5 * self._rf.predict(X) + 0.3 * self._gb.predict(X) + 0.2 * self._huber.predict(X)
