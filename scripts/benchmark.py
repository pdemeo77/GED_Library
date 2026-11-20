"""Benchmark script: train and evaluate all neural GED solvers on all datasets.

Usage:
    python scripts/benchmark.py --epochs 5 --max_pairs 200

Uses a 60/20/20 train/val/test split. Evaluates only on held-out test pairs
to avoid data leakage.

Produces a CSV table of results in results/benchmark.csv
"""

import argparse
import csv
import os
import sys
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np


def _build_test_pairs(graphs, gt, max_pairs, seed=42):
    """Build held-out test pairs using an 80/20 split on graph IDs.

    Returns only pairs where both graphs fall in the test (last 20%) split.
    """
    gt_dict = {}
    for id1, id2, ged in gt:
        key = (min(id1, id2), max(id1, id2))
        gt_dict[key] = ged

    graph_ids = sorted(graphs.keys())
    n = len(graph_ids)
    test_ids = set(graph_ids[int(n * 0.8) :])

    pairs = []
    for (id1, id2), ged in gt_dict.items():
        if id1 in test_ids and id2 in test_ids:
            pairs.append((id1, id2, ged))

    rng = np.random.RandomState(seed)
    rng.shuffle(pairs)
    return pairs[:max_pairs]


def run_benchmark(epochs: int, max_pairs: int, device: str):
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
        load_dataset,
        evaluate_solver,
        set_seed,
    )
    from ged_lib.solvers.neural.trainer import GEDTrainer

    set_seed(42)

    results = []
    datasets = ["AIDS"]

    classical_solvers = [
        ("hungarian", lambda: HungarianSolver()),
        ("gedgw", lambda: GEDGWSolver()),
    ]

    neural_solvers = [
        ("gediot", lambda ds: GEDIOTSolver(num_labels=29 if ds == "AIDS" else 1, device=device)),
        ("gedgnn", lambda ds: GedGNNSolver(num_labels=29 if ds == "AIDS" else 1, device=device)),
        ("simgnn", lambda ds: SimGNNSolver(num_labels=29 if ds == "AIDS" else 1, device=device)),
        ("tagsim", lambda ds: TaGSimSolver(num_labels=29 if ds == "AIDS" else 1, device=device)),
        ("gpn", lambda ds: GPNSolver(num_labels=29 if ds == "AIDS" else 1, device=device)),
        ("egsc", lambda ds: EGSCSolver(num_labels=29 if ds == "AIDS" else 1, device=device)),
        ("greed", lambda ds: GREEDSolver(num_labels=29 if ds == "AIDS" else 1, device=device)),
        (
            "gedgnn_full",
            lambda ds: GEDGNNFullSolver(num_labels=29 if ds == "AIDS" else 1, device=device),
        ),
        ("grasp", lambda ds: GraspSolver(num_labels=29 if ds == "AIDS" else 1, device=device)),
    ]

    for ds_name in datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_name}")
        print(f"{'=' * 60}")

        graphs, gt = load_dataset(ds_name)

        # Build held-out test pairs (no data leakage)
        test_pairs = _build_test_pairs(graphs, gt, max_pairs)
        print(f"  Test pairs (held-out): {len(test_pairs)}")

        # Classical solvers (no training)
        for name, factory in classical_solvers:
            print(f"\n  Evaluating {name} (no training)...")
            solver = factory()
            result = evaluate_solver(solver, graphs, test_pairs, dataset_name=ds_name)
            results.append(
                {
                    "solver": name,
                    "dataset": ds_name,
                    "trained": False,
                    "n_pairs": result.n_pairs,
                    "mae": result.mae,
                    "rmse": result.rmse,
                    "spearman_r": result.spearman_r,
                    "kendall_tau": result.kendall_tau,
                    "mean_time_s": result.mean_time_s,
                }
            )
            print(f"    MAE={result.mae:.4f}, Spearman={result.spearman_r:.4f}")

        # Neural solvers (train then evaluate on held-out test set)
        for name, factory in neural_solvers:
            print(f"\n  Training {name} for {epochs} epochs...")
            solver = factory(ds_name)
            trainer = GEDTrainer(
                model=solver,
                dataset_name=ds_name,
                num_epochs=epochs,
                batch_size=8,
                learning_rate=1e-3,
                save_dir=f"checkpoints/{ds_name}",
                device=device,
            )
            trainer.train()

            print(f"  Evaluating {name} on held-out test pairs...")
            result = evaluate_solver(solver, graphs, test_pairs, dataset_name=ds_name)
            results.append(
                {
                    "solver": name,
                    "dataset": ds_name,
                    "trained": True,
                    "n_pairs": result.n_pairs,
                    "mae": result.mae,
                    "rmse": result.rmse,
                    "spearman_r": result.spearman_r,
                    "kendall_tau": result.kendall_tau,
                    "mean_time_s": result.mean_time_s,
                }
            )
            print(f"    MAE={result.mae:.4f}, Spearman={result.spearman_r:.4f}")

    os.makedirs("results", exist_ok=True)
    csv_path = "results/benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'=' * 60}")
    print("BENCHMARK RESULTS (held-out test set)")
    print(f"{'=' * 60}")
    print(
        f"{'Solver':<12} {'Trained':<8} {'MAE':<8} {'RMSE':<8} {'Spearman':<10} {'Kendall':<10} {'Time(s)':<8}"
    )
    print("-" * 64)
    for r in results:
        print(
            f"{r['solver']:<12} {str(r['trained']):<8} "
            f"{r['mae']:<8.4f} {r['rmse']:<8.4f} "
            f"{r['spearman_r']:<10.4f} {r['kendall_tau']:<10.4f} "
            f"{r['mean_time_s']:<8.6f}"
        )
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark GED solvers")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs for neural solvers")
    parser.add_argument("--max_pairs", type=int, default=200, help="Max evaluation pairs")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    args = parser.parse_args()

    run_benchmark(args.epochs, args.max_pairs, args.device)
