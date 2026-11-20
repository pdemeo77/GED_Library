# NEXT_STEPS — authoritative backlog

> Single source of truth for what to do next. Any session (human or AI agent,
> on any device) should start here. Update this file when work is completed or
> priorities change. Detailed context lives in the referenced files.

**Last updated:** 2026-11-20 (session 16)
**Repo state:** public, CI green (pytest 3.10/3.11/3.12 + ruff report-only)

---

## How to work on this project

- Read `AGENTS.md` (project context) and this file before touching code.
- After each work session, append a log file to `SESSION_LOG/` (local) and
  update this file. Commit early, commit often; CI must stay green.
- Algorithm-fidelity findings and fix references: `docs/verification_report.md`.
- Solver ↔ paper ↔ repo mapping: `docs/solver.md`.
- Upstream code snapshots for verification: `../ged_upstream_refs/` (outside
  the repo, per-device). Paper PDFs: `docs/papers/` (local, gitignored).

---

## Priority queue

### 1. Fix wave A+B — correctness bugs + upstream alignment 🔥

Source and evidence: `docs/verification_report.md` (confirmed by two
independent reviews). Each fix needs a regression test.

**A — certain bugs:**
- [ ] `kbest_matching.py:256` — root LB init `compute([], best_matching)` →
      upstream `label_set([], [])`; makes the k-best search actually run
- [ ] `gedgnn_full.py:199` — feed `round(Softmax(dim=1)(map)·1e9 + 1)` to the
      k-best solver, not raw logits (also fixes the compounding `max_res=-1` issue)
- [ ] `kbest_matching.py:99-113` — edge term must count edges OUTSIDE the mapped
      subgraph, not total edge-count difference
- [ ] `gedgnn_full.py:206-214` — narrow the `except Exception: pass`, log failures
- [ ] `kbest_matching.py:115-125` — unlabeled-graph branch: `|n2−n1|` instead of
      the constant-feature sums (IMDB/Linux lower bound)
- [ ] `egsc.py:76-83` — remove the extra `fc0` block from the SE tensor-network
      head (net one Linear+ReLU pair too many vs upstream)
- [ ] `grasp.py` — (a) degree without self-loops, (b) degree out of final MLP,
      (c) raw conv outputs (pre-ReLU) into pattern matrices and final concat
- [ ] `grasp.py:191-203` — dropout configurable, not hardcoded 0.5

**B — design alignment to upstream (checkpoints are throwaway, zero cost):**
- [ ] GedGNN + GedGNN-Full: plain 3-layer GIN encoder (upstream GedGNN), keeping
      the multi-scale encoder only for GEDIOT
- [ ] SimGNN: single-bottleneck prediction head (Rozemberczki reference)
- [ ] GPN: default `target_mode="linear"`; upstream default dims
- [ ] GEDIOT: upstream default filters (128,64,32) / bottleneck (16,8,4)
- [ ] `layers.py`: NTN bias init `xavier_uniform_` (upstream)
- [ ] GEDGW: labeled graphs without `dataset_name` → build label encoding over
      the union of both graphs (or require `dataset_name` explicitly)

### 2. TaGED integration — richer ground truth (approved, data validated)

`TaGED.json` (chengqihao repo, all 3 datasets): `[id1, id2, ged, nc, in, ie, mappings]`.
Verified: identical to our AIDS ground truth on all 245,350 pairs.

- [ ] `scripts/download_taged.py` (fetch into `True_GED/<DS>/TaGED.json`,
      gitignored, cite source)
- [ ] Loader in `ged_lib/dataset/` → dict `(id1,id2) -> (ged, nc, in, ie, mappings)`
- [ ] `GEDTrainer`: optional mapping-supervised loss for GEDIOT / GedGNN /
      GedGNN-Full / Grasp (`fixed_mapping_loss + w·MSE`, upstream weights)
- [ ] TaGSim per-component loss (`nc/in/ie` targets)
- [ ] Tests with tiny synthetic mapping data

### 3. Property-based tests with exact GED

- [ ] networkx `graph_edit_distance` on small random graphs: identity → 0,
      symmetry, approximation ≥ exact − tolerance, sanity of unsupervised solvers

### 4. New solvers

- [ ] **HED** (row 14 of `docs/solver.md`): port from priba repo summary in
      `verification_report.md` (Hausdorff matching; no LSAP; lower-bound style)
- [ ] **FORISolver** (row 15): exact ILP from D'Ascenzo et al. PVLDB 2025
      (paper in `docs/papers/`), implement via `scipy.optimize.milp` (HiGHS),
      constraints 5a–5g; verify against their published optimal solutions

### 5. Code quality phases (P2/P3/P4)

- [ ] P2: shared `_GINBackbone` extraction review after fix wave; unified
      label-map utility; export `GEDTrainer`; `GEDSFSolver`/`FGWAlignSolver`
      in `__all__` only when the import succeeds; `requires_fit` contract
- [ ] P3: trainer shuffle + seed, LR scheduler, upstream training protocol
      (delta graphs, batch, weight decay) — decide per-solver
- [ ] P4.1: clean the 37 ruff errors, then make lint a blocking CI gate
      (remove `--exit-zero` in `ci.yml`)
- [ ] P4.2: GEDTrainer smoke test, failing-solver evaluation test, IMDB/Linux
      end-to-end tests; investigate slow tests (`--durations`)
- [ ] Verify `examples/quickstart.py` runs on a clean clone

### 6. Full evaluation (office machine, week-long, block-based)

- [ ] Make `scripts/comparison.py` CLI-driven (`argparse`: dataset, epochs,
      max pairs, seed, device) with **resumable per-block JSON output**
      (one block = solver × dataset; skip blocks whose output exists)
- [ ] 20 epochs / 2000 test pairs / 10k SF pairs per `PLAN_session16.md` (local)
- [ ] Update `docs/experiment_report.md` with converged results

### 7. Later / optional

- Naima tracker trial on this repo (after fix wave stabilizes)
- Literature scan for recent (2025–2026) GED algorithms to add
- `CITATION.cff` + README badges (CI status)

---

## Definition of Done (per item above)

Fixes A/B: regression test added, full `pytest` green, CI green, commit
message references the report section. New solvers: tests + entry in
`docs/solver.md` + experiment numbers added when the evaluation runs.
