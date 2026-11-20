"""Fair comparison benchmark for all GED solvers on AIDS dataset.

Trains neural methods for 5 epochs on 60/20/20 split, evaluates on held-out
test pairs. Computes: MAE, MSE, Accuracy (|pred-true|<=1), Spearman rho,
Kendall tau, and pairs/second.

Uses the library's evaluate_solver (ged_lib.evaluation.metrics) for metric
computation to ensure consistency across the codebase.

Usage:
    python scripts/comparison.py
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ged_lib import load_dataset, set_seed, evaluate_solver
from ged_lib.solvers.neural.data_adapter import get_num_labels, get_label_map
from ged_lib.solvers.neural.trainer import GEDTrainer

from ged_lib import (
    HungarianSolver,
    GEDGWSolver,
    GEDIOTSolver,
    GedGNNSolver,
    SimGNNSolver,
    TaGSimSolver,
    GPNSolver,
    EGSCSolver,
    GREEDSolver,
    GEDGNNFullSolver,
    GraspSolver,
)

# SF solver is optional (requires scikit-learn)
try:
    from ged_lib import GEDSFSolver
except ImportError:
    GEDSFSolver = None

# FGWAlign solver is optional (requires pot + torch)
try:
    from ged_lib import FGWAlignSolver
except ImportError:
    FGWAlignSolver = None

DATASET = "AIDS"
N_EPOCHS = 2
MAX_TEST_PAIRS = 500
MAX_SF_TRAIN_PAIRS = 2000  # Limit SF training pairs for tractable fitting (~60s each)
SEED = 42

set_seed(SEED)
device = "cpu"


def build_train_test_pairs(graphs, gt, max_test_pairs=500, seed=42):
    """Split ground truth into train and test pairs by graph ID (80/20).

    Args:
        graphs: Dict of {graph_id: nx.Graph}.
        gt: List of (id1, id2, true_ged) ground truth tuples.
        max_test_pairs: Maximum number of test pairs to return.
        seed: Random seed for shuffling.

    Returns:
        Tuple of (train_pairs, test_pairs) where each is a list of
        (id1, id2, true_ged) tuples.
    """
    gt_dict = {}
    for id1, id2, ged in gt:
        key = (min(id1, id2), max(id1, id2))
        gt_dict[key] = ged

    graph_ids = sorted(graphs.keys())
    n = len(graph_ids)
    split = int(n * 0.8)
    train_ids = set(graph_ids[:split])
    test_ids = set(graph_ids[split:])

    rng = np.random.RandomState(seed)

    train_pairs = []
    for (id1, id2), ged in gt_dict.items():
        if id1 in train_ids and id2 in train_ids:
            train_pairs.append((id1, id2, ged))

    test_pairs = []
    for (id1, id2), ged in gt_dict.items():
        if id1 in test_ids and id2 in test_ids:
            test_pairs.append((id1, id2, ged))

    rng.shuffle(train_pairs)
    rng.shuffle(test_pairs)
    return train_pairs, test_pairs[:max_test_pairs]


def build_test_pairs(graphs, gt):
    """Build test pairs only (backward-compatible wrapper)."""
    _, test = build_train_test_pairs(graphs, gt, max_test_pairs=MAX_TEST_PAIRS, seed=SEED)
    return test


def run_evaluation(solver, graphs, test_pairs, dataset_name):
    """Evaluate a solver using the library's evaluate_solver and add extra metrics.

    Uses ged_lib.evaluation.metrics.evaluate_solver for core metrics (MAE, MSE,
    RMSE, Spearman, Kendall, timing), then computes Acc@1 and throughput from
    the stored predictions.
    """
    result = evaluate_solver(solver, graphs, test_pairs, dataset_name=dataset_name)

    if result.n_pairs == 0:
        return None

    # Compute Acc@1 and throughput from stored arrays
    errors = np.abs(result.predicted - result.true_values)
    accuracy = float(np.mean(errors <= 1.0))
    pairs_per_sec = float(result.n_pairs / result.times.sum()) if result.times.sum() > 0 else 0.0

    return {
        "n_pairs": result.n_pairs,
        "mae": result.mae,
        "mse": result.mse,
        "rmse": result.rmse,
        "accuracy": accuracy,
        "spearman_rho": result.spearman_r,
        "kendall_tau": result.kendall_tau,
        "pairs_per_sec": pairs_per_sec,
        "mean_time_ms": float(result.mean_time_s * 1000),
        "error_rate": result.error_rate,
    }


def main():
    print("=" * 70)
    print(f"GED Solver Comparison — {DATASET} dataset, {N_EPOCHS} training epochs")
    print("=" * 70)

    graphs, gt = load_dataset(DATASET)
    num_labels = get_num_labels(graphs, DATASET)
    train_pairs, test_pairs = build_train_test_pairs(graphs, gt, MAX_TEST_PAIRS, SEED)

    print(
        f"Graphs: {len(graphs)}, Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}, Labels: {num_labels}"
    )
    print(f"GED range: [{min(p[2] for p in test_pairs)}, {max(p[2] for p in test_pairs)}]")
    print(f"Mean GED: {np.mean([p[2] for p in test_pairs]):.2f}")
    print()

    # === Unsupervised solvers (no training) ===
    unsupervised = [
        ("Hungarian", lambda: HungarianSolver()),
        ("GED-GW", lambda: GEDGWSolver()),
        *(
            [
                (
                    "FGWAlign-Fast",
                    lambda: FGWAlignSolver(dataset_name="AIDS", patience=15, topk=5),
                ),
                (
                    "FGWAlign-Light",
                    lambda: FGWAlignSolver(dataset_name="AIDS", patience=1, topk=1, light=True),
                ),
            ]
            if FGWAlignSolver is not None
            else []
        ),
    ]

    # === Neural solvers (need training) ===
    # All solvers use target_mode="exp" for consistent normalization.
    neural_defs = [
        ("GEDIOt", lambda: GEDIOTSolver(num_labels=num_labels, device=device, target_mode="exp")),
        ("GedGNN", lambda: GedGNNSolver(num_labels=num_labels, device=device, target_mode="exp")),
        ("SimGNN", lambda: SimGNNSolver(num_labels=num_labels, device=device, target_mode="exp")),
        ("TaGSim", lambda: TaGSimSolver(num_labels=num_labels, device=device, target_mode="exp")),
        ("GPN", lambda: GPNSolver(num_labels=num_labels, device=device, target_mode="exp")),
        ("EGSC", lambda: EGSCSolver(num_labels=num_labels, device=device, target_mode="exp")),
        ("GREED", lambda: GREEDSolver(num_labels=num_labels, device=device, target_mode="exp")),
        (
            "GedGNN-Full",
            lambda: GEDGNNFullSolver(num_labels=num_labels, device=device, target_mode="exp"),
        ),
        ("Grasp", lambda: GraspSolver(num_labels=num_labels, device=device, target_mode="exp")),
    ]

    results = {}

    # --- Evaluate unsupervised ---
    print("--- Unsupervised Solvers ---")
    for name, factory in unsupervised:
        print(f"\nEvaluating {name}...")
        solver = factory()
        r = run_evaluation(solver, graphs, test_pairs, DATASET)
        if r:
            results[name] = r
            print(
                f"  MAE={r['mae']:.2f}, Acc={r['accuracy']:.4f}, "
                f"Spearman={r['spearman_rho']:.4f}, pairs/s={r['pairs_per_sec']:.1f}"
            )

    # --- Train and evaluate neural ---
    print("\n\n--- Neural Solvers (training) ---")
    for name, factory in neural_defs:
        print(f"\n{'=' * 50}")
        print(f"Training {name} for {N_EPOCHS} epochs...")
        solver = factory()

        trainer = GEDTrainer(
            model=solver,
            dataset_name=DATASET,
            num_epochs=N_EPOCHS,
            batch_size=16,
            learning_rate=0.001,
            device=device,
        )
        trainer.train()

        # K4 diagnostic: log sample target to verify target_mode consistency
        if trainer.train_pairs:
            sample_id1, sample_id2, sample_ged = trainer.train_pairs[0]
            sample_data = trainer._pack_pair(sample_id1, sample_id2, sample_ged)
            print(
                f"  [DIAG] target_mode={solver.target_mode}, "
                f"sample true_ged={sample_ged}, target={sample_data['target'].item():.6f}"
            )

        print(f"Evaluating {name}...")
        r = run_evaluation(solver, graphs, test_pairs, DATASET)
        if r:
            results[name] = r
            print(
                f"  MAE={r['mae']:.2f}, Acc={r['accuracy']:.4f}, "
                f"Spearman={r['spearman_rho']:.4f}, pairs/s={r['pairs_per_sec']:.1f}"
            )

    # --- Train and evaluate SF solvers ---
    if GEDSFSolver is not None:
        print("\n\n--- SF Solvers (fit + evaluate) ---")
        sf_defs = [
            ("GED-SF-Linear", lambda: GEDSFSolver(formulation="linear", device=device)),
            ("GED-SF-Quadratic", lambda: GEDSFSolver(formulation="quadratic", device=device)),
        ]

        for name, factory in sf_defs:
            print(f"\n{'=' * 50}")
            print(f"Fitting {name} (cross-validating mu x regressors)...")
            solver = factory()
            sf_pairs = train_pairs[:MAX_SF_TRAIN_PAIRS]
            solver.fit(graphs, sf_pairs)
            print(f"  Best config: mu={solver._best_mu}, regressor={solver._best_config}")

            print(f"Evaluating {name}...")
            r = run_evaluation(solver, graphs, test_pairs, DATASET)
            if r:
                results[name] = r
                print(
                    f"  MAE={r['mae']:.2f}, Acc={r['accuracy']:.4f}, "
                    f"Spearman={r['spearman_rho']:.4f}, pairs/s={r['pairs_per_sec']:.1f}"
                )
    else:
        print("\n\n--- SF Solvers: SKIPPED (scikit-learn not available) ---")

    # --- Print summary table ---
    print("\n\n" + "=" * 100)
    print(f"RESULTS — {DATASET}, {N_EPOCHS} epochs, {MAX_TEST_PAIRS} test pairs")
    print("=" * 100)
    header = f"{'Method':<15} {'MAE':>8} {'MSE':>8} {'Acc@1':>8} {'Spearman':>10} {'Kendall':>10} {'pairs/s':>10} {'ms/pair':>10}"
    print(header)
    print("-" * 100)

    # Sort by MAE
    for name, r in sorted(results.items(), key=lambda x: x[1]["mae"]):
        print(
            f"{name:<15} {r['mae']:>8.2f} {r['mse']:>8.2f} {r['accuracy']:>8.4f} "
            f"{r['spearman_rho']:>10.4f} {r['kendall_tau']:>10.4f} "
            f"{r['pairs_per_sec']:>10.1f} {r['mean_time_ms']:>10.2f}"
        )

    print("=" * 100)

    # Save results as JSON
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "comparison.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"dataset": DATASET, "epochs": N_EPOCHS, "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
