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

### 1. Fix wave A+B — correctness bugs + upstream alignment 🔥 ✅ DONE 2026-11-21

Source and evidence: `docs/verification_report.md` (confirmed by two
independent reviews). Each fix needs a regression test.

**A — certain bugs (commits 66e0f0d, 3afd83e, 83c6371):**
- [x] `kbest_matching.py` — root LB init `compute([], [])` (unconstrained,
      upstream `label_set`); the k-best search actually runs now
- [x] `gedgnn_full.py` — weights `round(Softmax(dim=1)(map)·1e9 + 1)`;
      answer `min(neural_ged, solver.min_ged)`; failures logged, not swallowed
- [x] `kbest_matching.py` — edge term counts edges OUTSIDE the mapped subgraph
- [x] `kbest_matching.py` — unlabeled branch uses `|n2−n1|`
- [x] `egsc.py` — fc0 block removed from the SE tensor-network head
- [x] `grasp.py` — degree without self-loops; degree out of final MLP;
      raw conv outputs into patterns/concat; dropout configurable

**B — design alignment (commit f41a406; checkpoints were throwaway):**
- [x] GedGNN + GedGNN-Full: plain 3-layer GIN encoder (`_PlainGINEncoder`);
      multi-scale encoder exclusive to GEDIOT
- [x] SimGNN: single-bottleneck head (Rozemberczki reference)
- [x] GPN: `target_mode="linear"`, filters (128,64,32), bottleneck (16,)
- [x] GEDIOT/GedGNN/GedGNN-Full: upstream defaults (128,64,32)/(16,8,4)
- [x] `layers.py`: NTN bias init `xavier_uniform_`
- [x] GEDGW: union label encoding across the pair when no dataset map

18 regression tests added (TestKBestMatchingFixes, TestEGSCFixes,
TestGraspFixes, TestUpstreamAlignmentFixes).

### 2. TaGED integration — richer ground truth ✅ DONE 2026-11-21 (c339072, 7615da1)

- [x] `scripts/download_taged.py` (AIDS/IMDB/Linux, gitignored, source cited)
- [x] Loader: `load_taged` / `taged_lookup` / `taged_mapping_matrix` (union
      of up-to-10 optimal matchings; semantics validated 200/200 on real AIDS)
- [x] `GEDTrainer`: upstream losses — GedGNN/GedGNN-Full `map+10·MSE`
      (AIDS/Linux, 1.0 IMDB), GEDIOT/Grasp `2·map+8·MSE`, TaGSim
      per-component exp targets; auto-enabled when TaGED exists,
      byte-identical behaviour otherwise
- [x] Tests: 2 loader + 5 trainer (incl. real-data end-to-end, skipped in CI)
- [x] GEDTrainer `max_pairs` param for fast smoke runs

*AIDS TaGED downloaded and validated (245,350 pairs, values identical).
IMDB/Linux: download on the office machine before the evaluation.*

### 3. Property-based tests with exact GED

- [ ] networkx `graph_edit_distance` on small random graphs: identity → 0,
      symmetry, approximation ≥ exact − tolerance, sanity of unsupervised solvers

### 4. New solvers

- [x] **HED** (row 14) — DONE 2026-11-21 (`412352d`): faithful priba port,
      5 tests incl. first property test (HED ≤ networkx exact GED)
- [ ] **FORISolver** (row 15): exact ILP from D'Ascenzo et al. PVLDB 2025
      (paper in `docs/papers/`), implement via `scipy.optimize.milp` (HiGHS),
      constraints 5a–5g; verify against their published optimal solutions
- [ ] **App-BMao** (row 19, GED-EXP repo in `../ged_upstream_refs/GED-EXP/`):
      read Models/App-BMao (Python + `ged` C++ binary — assess portability),
      heuristic upper bound, beats learning methods per the paper
- [ ] **GraphEdX** (row 18, structlearning/GraphEdX), **DiffGED** (row 16,
      code in `../ged_upstream_refs/DiffGED/`), **GELATO** (row 17,
      BorgwardtLab/Gelato)
- [ ] Wave 2 (post-evaluation): SDTED, ERIC, GENN-A* (all in GED-EXP)

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
