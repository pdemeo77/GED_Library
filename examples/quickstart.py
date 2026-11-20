"""Quickstart example for GED_Library.

Demonstrates loading a dataset, computing GED with classical and neural solvers,
and evaluating performance.

Usage:
    python examples/quickstart.py
"""

from ged_lib import HungarianSolver, GEDGWSolver, load_dataset, evaluate_solver, set_seed

set_seed(42)

# Load the AIDS dataset
print("Loading AIDS dataset...")
graphs, ground_truth = load_dataset("AIDS")
print(f"  Graphs: {len(graphs)}")
print(f"  Ground truth pairs: {len(ground_truth)}")

# Inspect a graph
g = list(graphs.values())[0]
print(f"  Sample graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

# --- Classical solver (no training) ---
print("\n--- Hungarian Solver (unsupervised) ---")
solver = HungarianSolver()
id1, id2 = ground_truth[0][0], ground_truth[0][1]
ged = solver.compute(graphs[id1], graphs[id2])
print(f"  GED({id1}, {id2}) = {ged:.4f}  (true = {ground_truth[0][2]})")

# Evaluate on a small subset
result = evaluate_solver(solver, graphs, ground_truth[:50], dataset_name="AIDS")
print(f"  Evaluated {result.n_pairs} pairs:")
print(f"    MAE={result.mae:.4f}, Spearman rho={result.spearman_r:.4f}")

# --- GED-GW solver (unsupervised) ---
print("\n--- GED-GW Solver (unsupervised) ---")
gedgw = GEDGWSolver()
ged2 = gedgw.compute(graphs[id1], graphs[id2])
print(f"  GED({id1}, {id2}) = {ged2:.4f}  (true = {ground_truth[0][2]})")

# --- Neural solver (needs training) ---
print("\n--- GEDIOT Solver (neural, needs training) ---")
try:
    from ged_lib import GEDIOTSolver
    from ged_lib.solvers.neural.trainer import GEDTrainer

    model = GEDIOTSolver(num_labels=29, device="cpu")
    trainer = GEDTrainer(
        model=model,
        dataset_name="AIDS",
        num_epochs=1,  # Just 1 epoch for quickstart
        batch_size=16,
        learning_rate=1e-3,
        device="cpu",
    )
    trainer.train()

    ged3 = model.compute(graphs[id1], graphs[id2])
    print(f"  GED({id1}, {id2}) = {ged3:.4f}  (true = {ground_truth[0][2]})")
except ImportError:
    print("  (torch not installed — skipping neural solver)")

print("\nDone!")
