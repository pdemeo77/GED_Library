# GED_Library

A Python library for approximating **Graph Edit Distance (GED)** between two graphs, with a uniform scikit-learn-like interface to multiple GED algorithms.

## Features

- **13 solvers** spanning classical, optimal transport, ML-regression, neural, and post-processing approaches
- **Uniform interface**: every solver has `compute(g1, g2) -> float`
- **3 benchmark datasets**: AIDS (labeled), IMDB (unlabeled), Linux (unlabeled)
- **Training loop** included for neural models
- **Evaluation metrics**: MAE, MSE, RMSE, Spearman, Kendall, timing

## Installation

```bash
# Core (classical + OT solvers)
pip install -e .

# With neural solver support
pip install -e ".[ot]"
pip install -e ".[neural]"

# Development
pip install -e ".[dev]"
```

## Datasets

This repo ships a **small sample** of each benchmark so examples and tests run
out of the box. To fetch the **full** AIDS / IMDB / Linux corpora:

1. Set the source env vars (or edit `scripts/download_datasets.py`):

   ```bash
   export GED_DATA_OWNER=YOUR_GITHUB_USER
   export GED_DATA_REPO=YOUR_REPO
   ```

2. Run:

   ```bash
   python scripts/download_datasets.py
   ```

   This overwrites `Dataset/` and `True_GED/` with the complete data.

## Quick Start

### Basic Usage

```python
from ged_lib import HungarianSolver, load_dataset

# Load a dataset
graphs, ground_truth = load_dataset("AIDS")

# Pick a solver
solver = HungarianSolver()

# Compute GED between two graphs
g1 = graphs[8116]
g2 = graphs[559]
ged = solver.compute(g1, g2)
print(f"GED = {ged}")
```

### List Available Solvers

```python
from ged_lib import list_solvers
print(list_solvers())
# {'hungarian': HungarianSolver, 'gedgw': GEDGWSolver, ...}
```

### Evaluate a Solver

```python
from ged_lib import HungarianSolver, load_dataset, evaluate_solver

graphs, gt = load_dataset("AIDS")
solver = HungarianSolver()
result = evaluate_solver(solver, graphs, gt, dataset_name="AIDS", max_pairs=500)
print(result)
# MAE=7.74, RMSE=7.90, Spearman rho=0.998, Kendall tau=0.992
```

### Train a Neural Model

```python
from ged_lib import GEDIOTSolver
from ged_lib.solvers.neural.trainer import GEDTrainer

model = GEDIOTSolver(num_labels=29, device="cpu")
trainer = GEDTrainer(model, dataset_name="AIDS", num_epochs=20, batch_size=8)
trainer.train()
results = trainer.test()
```

### Run Full Benchmark

```bash
python scripts/benchmark.py --epochs 20 --max_pairs 500
```

## Solvers

| Solver | Type | Training | Method | Speed |
|--------|------|----------|--------|-------|
| `hungarian` | Classical | No | Bipartite matching (Hungarian algorithm) | <1ms |
| `gedgw` | Optimal Transport | No | Fused Gromov-Wasserstein (SIGMOD'25) | ~40ms |
| `gediot` | Neural | Yes | GIN + Sinkhorn OT (SIGMOD'25) | ~1ms |
| `gedgnn` | Neural | Yes | GIN + mapping matrix | ~1ms |
| `simgnn` | Neural | Yes | GCN + attention + histogram (Bai et al. 2019) | ~1ms |
| `tagsim` | Neural | Yes | Type-aware graph similarity | <1ms |
| `gpn` | Neural | Yes | GIN + matching module | ~1ms |
| `egsc` | Neural | Yes | Multi-level GIN + SE-attention (NeurIPS 2021) | ~1ms |
| `greed` | Neural | Yes | Siamese GNN + L2/MLP comparison (NeurIPS 2022) | ~1ms |
| `gedgnn_full` | Neural + Post-proc | Yes | GedGNN + k-best matching (PVLDB 2023) | ~10ms |
| `grasp` | Neural | Yes | 5-layer GIN + GRU pattern matching | ~2ms |
| `fgwalign` | Optimal Transport | No | FGW transport plan → permutation (FGWAlign 2024) | — |
| `gedsf` | Frank-Wolfe + ML | `fit()` | FW alignment + ML regression | — |

## Benchmark Results (AIDS, 3 epochs, held-out test set)

> Results use a 60/20/20 train/val/test split evaluated on 500 held-out test pairs.
> Code includes fixes for label encoding (§1.1) and Hungarian edge costs (§1.2).

| Method | MAE ↓ | MSE ↓ | Acc@1 ↑ | Spearman ρ ↑ | Kendall τ ↑ | pairs/s ↑ |
|--------|------:|------:|--------:|-------------:|------------:|----------:|
| Hungarian | **1.36** | **3.07** | **0.624** | **0.782** | **0.656** | 5,152.8 |
| GREED | 1.47 | 3.73 | 0.438 | 0.345 | 0.250 | 1,340.4 |
| GED-GW | 1.62 | 4.41 | 0.376 | 0.397 | 0.296 | 22.1 |
| EGSC | 2.18 | 7.09 | 0.224 | −0.150 | −0.112 | 885.6 |
| GPN | 4.55 | 25.63 | 0.040 | 0.086 | 0.061 | 1,037.8 |
| SimGNN | 5.02 | 30.41 | 0.032 | −0.176 | −0.131 | 713.9 |
| GedGNN | 5.66 | 36.44 | 0.012 | 0.046 | 0.032 | 776.5 |
| Grasp | 6.44 | 46.90 | 0.004 | 0.046 | 0.033 | 445.5 |
| GedGNN-Full | 6.49 | 52.48 | 0.056 | −0.019 | −0.015 | 346.5 |
| GEDIOT | 7.09 | 54.89 | 0.004 | 0.003 | 0.002 | 858.3 |
| TaGSim | 9.49 | 95.37 | 0.000 | −0.197 | −0.162 | 1,939.0 |

*Run `python scripts/comparison.py` to reproduce.*

## Project Structure

```
GED_Library/
├── ged_lib/
│   ├── __init__.py              # Public API
│   ├── base.py                  # GEDSolver abstract base class + __repr__
│   ├── graph.py                 # CSV → networkx.Graph + label encoding
│   ├── cost.py                  # Configurable cost models
│   ├── solvers/
│   │   ├── __init__.py          # Solver registry
│   │   ├── hungarian.py         # Bipartite matching
│   │   ├── gedgw.py             # Gromov-Wasserstein OT
│   │   └── neural/
│   │       ├── models.py        # GEDIOT, GedGNN, SimGNN, TaGSim, GPN
│   │       ├── layers.py        # Shared neural layers
│   │       ├── ged_matrix.py    # Cost/mapping matrix modules
│   │       ├── data_adapter.py  # networkx → torch bridge + label map
│   │       ├── egsc.py          # EGSC (NeurIPS 2021)
│   │       ├── greed.py         # GREED (NeurIPS 2022)
│   │       ├── gedgnn_full.py   # GEDGNN-matching (PVLDB 2023)
│   │       ├── grasp.py         # Grasp (multi-level GRU pattern matching)
│   │       ├── kbest_matching.py # K-best matching post-processing
│   │       └── trainer.py       # Training loop with graph tensor caching
│   ├── dataset/
│   │   └── loader.py            # load_dataset() for AIDS/IMDB/Linux
│   └── evaluation/
│       └── metrics.py           # MAE, MSE, Spearman, Kendall, timing
├── Dataset/                     # Graph CSV data
├── True_GED/                    # Ground truth GED values
├── scripts/
│   ├── benchmark.py             # Full benchmark (train/test split)
│   ├── comparison.py            # Fair comparison with held-out test
│   ├── make_sample.py           # Reduce datasets to a small sample
│   └── download_datasets.py     # Download full datasets from a release
├── examples/
│   └── quickstart.py            # Runnable quickstart example
├── tests/
├── pyproject.toml
└── PLAN.md
```

## Dependencies

**Core**: numpy, scipy, networkx

**Optional**:
- `[ot]`: pot, torch (for GEDGW solver)
- `[neural]`: torch, torch_geometric, pot (for neural solvers)
- `[plot]`: matplotlib (for visualization)
- `[dev]`: pytest, ruff

## References

- Cheng, Q. et al. (2025). *Computing Approximate Graph Edit Distance via Optimal Transport*. SIGMOD'25.
- Bai, Y. et al. (2019). *SimGNN: A Neural Network Approach to Fast Graph Similarity Computation*. WWW'19.
- Qin, C. et al. (2021). *Efficient Graph Similarity Computation via Knowledge Distillation*. NeurIPS 2021.
- Ranjan, R. et al. (2022). *GREED: A Neural Framework for Learning Graph Distance Functions*. NeurIPS 2022.
- Piao, C. et al. (2023). *Computing Graph Edit Distance via Neural Graph Matching*. PVLDB 2023.
- Riesen, K. & Bunke, H. (2009). *Approximate graph edit distance computation by means of bipartite graph matching*.

## License

MIT
