# GED_Library — Agent Context

## What Is This Project?

GED_Library is a Python library for approximating **Graph Edit Distance (GED)** between two graphs.
It provides a scikit-learn-like uniform interface so researchers can call different GED algorithms
without dealing with each one's unique dependencies and formats.

## Architecture

```
ged_lib/
├── __init__.py          # Public API: GEDSolver, load_dataset, list_solvers, set_seed
├── base.py              # GEDSolver abstract base class + __repr__
├── graph.py             # Graph I/O: CSV → networkx.Graph, label encoding, nx_to_tensors
├── cost.py              # CostModel: configurable edit costs (unit, symbolic, label)
├── solvers/
│   ├── __init__.py      # Solver registry
│   ├── hungarian.py     # HungarianSolver — bipartite matching heuristic
│   ├── gedgw.py         # GEDGWSolver — Gromov-Wasserstein OT (unsupervised)
│   ├── fgwalign.py      # FGWAlignSolver — FGW alignment projection (Tang et al. 2024)
│   ├── _fgwalign_core.py # Verbatim core from squareRoot3/FGWAlign (black-box, do not refactor blindly)
│   ├── sf/              # GED-SF: Frank-Wolfe alignment + ML regression (needs fit())
│   │   ├── gedsf_solver.py  # GEDSFSolver — fit() + compute()
│   │   ├── fw_solver.py     # solve_ged_fw, compute_fw_objective
│   │   └── feature_extraction.py # 14-dim structural features
│   └── neural/
│       ├── models.py    # NeuralGEDSolver base + GEDIOT, GedGNN, SimGNN, TaGSim, GPN
│       ├── layers.py    # AttentionModule, TensorNetworkModule, MatchingModule, OTLayer
│       ├── ged_matrix.py # CostMatrixModule, GedMatrixModule, fixed_mapping_loss
│       ├── data_adapter.py # nx_to_neural_data, label map loading, _build_features
│       ├── trainer.py   # GEDTrainer — training loop with graph tensor caching
│       ├── egsc.py      # EGSCSolver — SE-attention + co-attention
│       ├── greed.py     # GREEDSolver — Siamese GNN framework
│       ├── gedgnn_full.py # GEDGNNFullSolver — GedGNN + k-best matching post-processing
│       ├── grasp.py     # GraspSolver — multi-level GRU pattern matching
│       └── kbest_matching.py # KBestMSolver — k-best bipartite matching with GED LB pruning
├── dataset/
│   ├── __init__.py
│   └── loader.py        # load_dataset(name) for AIDS/IMDB/Linux, load_labels()
└── evaluation/
    ├── __init__.py
    └── metrics.py        # evaluate_solver, EvaluationResult: MAE, MSE, Spearman, Kendall, timing
```

## Key Design Decisions

- **Graph format**: `networkx.Graph` internally. Nodes may have a `label` attribute.
- **Solver interface**: `GEDSolver.compute(g1, g2) -> float` and `compute_pairwise(graphs) -> np.ndarray`
- **Cost model**: Decoupled from solvers. `CostModel` class handles node/edge edit costs.
  - Edge substitution cost is `0.0` by default (matched edges are preserved).
- **Label encoding**: String labels (e.g. `"C"`, `"N"`) are mapped to integer indices via
  `labels.json` before one-hot encoding. The mapping is loaded via `get_label_map()`.
- **Neural solvers**: Inherit from `NeuralGEDSolver` (nn.Module + GEDSolver). Trained via
  `GEDTrainer` which caches per-graph tensors across epochs.
- **Datasets**: AIDS (labeled, 29 labels), IMDB (unlabeled), Linux (unlabeled).
  Stored as CSV in `Dataset/` and `True_GED/`.
- **13 solver classes (15 method variants)**:
  - Classical/OT (no training): Hungarian, GED-GW, FGWAlign (Fast: patience=15/topk=5,
    Light: patience=1/topk=1/light)
  - ML regression (needs `fit()`): GED-SF (Linear / Quadratic formulations)
  - Neural (trained via `GEDTrainer`): GEDIOT, GedGNN, SimGNN, TaGSim, GPN, EGSC, GREED,
    GedGNN-Full (GedGNN + k-best matching), Grasp

## Data Format

### Graphs CSV (`Dataset/<name>/<name>_graphs.csv`)
```csv
graph_id,graph_edge_list,node_labels
13947,"[[7, 3], [7, 8], ...]","['C', 'C', 'N', ...]"
```
- `node_labels` column is absent for IMDB and Linux (unlabeled graphs)

### Ground Truth CSV (`True_GED/<name>/<name>_ged.csv`)
```csv
id_1,id_2,true_ged
8116,559,1
```

### Label Encoding (`Dataset/<name>/labels.json`)
Only for AIDS: maps label strings to integer indices.

## Dependencies

- **Core**: numpy, scipy, networkx
- **Optional extras**: `[ot]` pot+torch (GED-GW, FGWAlign), `[neural]` torch+torch_geometric+pot,
  `[sf]` scikit-learn+xgboost+lightgbm (GED-SF), `[plot]` matplotlib, `[dev]` pytest+ruff

## Session Persistence

- **`NEXT_STEPS.md` (in repo): authoritative backlog — read it FIRST each session**
- `SESSION_LOG.md` + `SESSION_LOG/` (local-only): running diary, one file per session
- `PLAN_session16.md` (local-only): detailed plan for the full 3-dataset evaluation (pending)
- Paper PDFs for the verification phase go in `docs/papers/` (local-only, gitignored)
- Upstream code snapshots for fidelity checks: `../ged_upstream_refs/` (outside the repo)

## Known Open Questions

- ~~`_fgwalign_core.py` code review pending~~ **RESOLTO 2026-11-20**: le 3 funzioni copiate sono
  identiche a livello di AST all'upstream `main` (differenze solo docstring/import/logging,
  tutte intenzionali). Upstream salvato in `../ged_upstream_refs/FGWAlign.py`.
- ~~GREED skip-residual (`if i & 1`) unverified~~ **RESOLTO 2026-11-20**: semantica confermata
  identica a `neuro/models.py` del repo upstream (residual ai layer dispari, reset anchor,
  ReLU dopo il residual, concat emb a ogni layer).
- Verifica sistematica dei restanti solver vs upstream: ✅ completata 2026-11-20 —
  esiti in `docs/verification_report.md` (in repo pubblico). Bug reali da correggere:
  EGSC fc0, Grasp ×3, GedGNN-Full k-best ×3, GPN docstring, GEDGW label-encoding edge case,
  CostModel float-cost TypeError. Decisioni pendenti: encoder GedGNN, testa SimGNN,
  default iperparametri, dati con node-mapping/decomposed-GED.
- Neural methods in `docs/experiment_report.md` were evaluated at only 2 epochs (underfit);
  full evaluation is planned in `PLAN_session16.md`
