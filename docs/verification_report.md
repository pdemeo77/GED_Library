# Algorithm Fidelity Verification Report

> Systematic comparison of each solver against its upstream reference implementation.
> Method: source-level semantic comparison (architecture, forward-pass math, losses, data
> handling). Upstream snapshots archived in `ged_upstream_refs/` (outside the repo).
> Started: 2026-11-20.

## Verdicts

| Solver | Upstream | Verdict | Behavior-changing issues |
|---|---|---|---|
| FGWAlign | squareRoot3/FGWAlign | ✅ VERIFIED | none (AST-identical, only docstring/logging/import diffs, all intended) |
| GREED | idea-iitd/greed | ✅ VERIFIED | none (residual `if i&1` semantics confirmed: odd layers, anchor reset, ReLU after residual) |
| SimGNN | benedekrozemberczki/SimGNN | ⚠️ VERIFIED w/ deviations | 1 (prediction head) |
| EGSC | canqin001/EGSC | ⚠️ VERIFIED w/ deviations | 1 (SE tensor-network head) |
| Grasp | JunkyuLee1/Grasp | ⚠️ DEVIATIONS | 4 (degree off-by-one, degree in final MLP, post-ReLU features, missing mapping loss) |
| GedGNN | ChengzhiPiao/GEDGNN | ⚠️ DEVIATIONS | 2 (encoder = GEDIOT's, missing mapping loss) |
| GedGNN-Full | ChengzhiPiao/GEDGNN | 🚨 BUGS | 3 (k-best search dead code, weights not softmax-scaled, GED formula miscounts) |
| TaGSim | jiyangbai/TaGSim | ⚠️ VERIFIED w/ deviations | 1 (training loss aggregate vs per-component) |
| Hungarian | priba/aproximated_ged | ✅ VERIFIED | none (upstream has a g1/g2 swap bug we correctly do NOT replicate) |
| GED-GW | chengqihao/GED-via-OT | ✅ VERIFIED | 1 edge case (label encoding when `dataset_name=None`) |
| GEDIOT | chengqihao/GED-via-OT | ⚠️ VERIFIED w/ deviations | defaults + missing mapping loss + training protocol |
| GPN | chengqihao/GED-via-OT | ⚠️ VERIFIED w/ deviations | defaults (target_mode, dims) |
| GED-SF | (original work — P. De Meo) | n/a | author's own algorithm |

## Detailed findings

### SimGNN — verified with one deviation

Faithful: stratified GCN encoder (3×GCNConv, ReLU+dropout placement), AttentionModule
pooling math, NTN tensor-network layer, histogram feature (`histc` 16 bins + normalize +
concat), MSE loss on `exp(-ged/(0.5(n1+n2)))` targets.

**Deviation (behavior-changing): prediction head depth.** Local: 3 hidden layers
(→128→64→32→1, `models.py:339-343`). Upstream (Rozemberczki): single bottleneck layer
(`Linear(f, bottleneck)` + `Linear(→1)`). Local follows the chengqihao lineage. Impact:
parameter count/capacity differs; checkpoints incompatible across the two conventions.
*Decision needed:* align to Rozemberczki reference or document the chengqihao head.

Benign: bias init zeros vs xavier; default filters (64,32,16) vs (128,64,32); histogram
default True vs False (all configurable).

### EGSC — verified with one deviation

Faithful: 3-level GIN backbone (Linear-ReLU-Linear-BN, train_eps=True), multi-level
attention with SE-enhanced residual (`TanhMLP(x)*x + x`), level fusion, SE modules
(reduction 4), `exp(-ged/avg_v)` targets, MSE.

**Deviation (behavior-changing): SE tensor-network head.** Local `fc_out` applies an extra
2-layer MLP before the final projection (`egsc.py:76-83`). Upstream defines that `fc0` block
but NEVER calls it — forward uses only `fc1` (`EGSC-T_layers.py:236-243`); EGSC-KD has the
call explicitly commented out. All upstream variants agree: the local port erroneously
activates the dead block.

Benign: `track_running_stats=False` on BN; self-loop epsilon shift (+1) in GIN aggregation;
default bottleneck 128 vs 16, dropout 0.5 vs 0.

### Grasp — inference path faithful, four deviations (most serious of this batch)

Faithful: 6-level pattern matrices (bilinear, no MLP — matching upstream's unused MLP),
GRU + additive-attention pooling, cross-attention cost aggregation (scaled dot-product),
map_mlp + bidirectional softmax, bias head, `sigmoid((row+col)/2 + bias)`, edge
symmetrization, target modes.

**Deviations (behavior-changing):**
1. **Degree off-by-one**: `_compute_degree` counts self-loops → systematic +1 shift
   (`grasp.py:177-180` vs upstream `models.py:137-140` which skips them).
2. **Degree in final MLP**: local concats degree into the final feature vector; upstream
   does not (k = num_labels+1+6f vs num_labels+6f).
3. **Post-ReLU intermediate features**: local feeds post-ReLU/post-dropout x1..x4 into
   pattern matrices and final concat; upstream uses raw conv outputs.
4. **Training objective**: upstream = `2*fixed_mapping_loss + 8*MSE` with ground-truth node
   mappings; local = plain MSE only (`fixed_mapping_loss` imported but never called, gt
   mapping absent from the data pipeline). NOTE: our `True_GED` CSVs contain only
   `true_ged`, no node mappings — full fidelity would require mapping data we may not have.

### Benign-notes archive

- EGSC upstream uses sum-reduction MSE (local: mean) — loss scale only.
- Grasp upstream defaults: filters 32, bottleneck 16/8/4, linear target mode.

### GedGNN — deviations: wrong encoder + missing mapping loss

Faithful: `GedMatrixModule` (k-head bilinear + MLP), regression head
(`Softmax(map)*cost`, attention→NTN→FC bias, `sigmoid(sum+bias)`, `-log(score)*avg_v`),
data semantics (bidirectional edges + self-loops, unlabeled features=2.0, `avg_v`,
`hb=max(n,m)+max(m)`), `fixed_mapping_loss` port (identical, incl. subsampling p=0.5).

**Deviations (behavior-changing):**
1. **Encoder**: local `_GINBackbone` (concat [input,h1,h2,h3] + MLP) is chengqihao's
   **GEDIOT** encoder, not GedGNN's plain 3-layer GIN pass. Upstream GedGNN checkpoints
   cannot load into the local solver.
2. **Loss**: upstream = `fixed_mapping_loss(map, gt_map) + 10*MSE` (AIDS/Linux);
   local = MSE only (no gt mappings in our True_GED CSVs — same data limitation as Grasp).

Benign: bias init zeros vs xavier; default hyperparameters (configurable).

### GedGNN-Full / k-best matching — 🚨 three behavior-changing bugs

The subspace machinery (second-best matching via minimum alternating cycle, split(),
expand-by-max-second) is a faithful port. But:

1. **k-best search never runs** (`kbest_matching.py:256`): root LB initialized with
   `self.lb.compute([], best_matching)` (degenerates to `sp.ged`) instead of upstream's
   unconstrained LB `label_set([], [])`. Since `_expand` requires `sp.lb < min_ged` and
   `min_ged <= sp.ged`, the condition is never true → `solve(k)` always returns the single
   argmax-weight matching. **The entire k-best enumeration — the point of the PVLDB
   extension — is dead code.**
2. **Weights not preprocessed** (`gedgnn_full.py:199`): upstream feeds
   `round(Softmax(dim=1)(map)*1e9 + 1)` to the k-best solver; local feeds raw logits.
   Even the single returned matching maximizes a different objective.
3. **GED-from-candidate edge term wrong** (`kbest_matching.py:99-113`): local computes
   total-edge-count difference; upstream counts only edges outside the mapped subgraph.
   For n1==n2 pairs local double-counts half the differing edges (~1.5× overcount).
4. Minor: `solve()` min over `sp.ged` only vs upstream's `min_ged` (second-best of
   unsplit subspaces); moot due to bug 1.

### TaGSim — architecture verified, training loss differs

Faithful: parameter-free GraphAggregationLayers (`adj@X` twice), three TensorNetworkModules,
three heads (tensor_neurons→bottleneck→8→4→1, ReLU+sigmoid), 01/12 concats feeding
nc/in/ie heads, sum-pooling, exact GED inversion `Σ -log(s_c)*avg_v`, adjacency handling.

**Deviation (behavior-changing, training only):** upstream supervises each component
(`MSE(exp(-nc/avg), s_nc)` etc., per-component); local trains
`MSE(Σ scores, exp(-total_ged/avg))` — additive score vs product-of-exponentials target,
no per-component supervision. Cause: our True_GED CSVs lack decomposed GEDs
(nc/in/ie split). Benign: extra "linear" target mode (default matches upstream exp).

### Hungarian — VERIFIED (with a pleasant surprise)

Verified against `priba/aproximated_ged` (AproximatedEditDistance/VanillaAED):
- (n1+n2)² matrix layout (different filling convention, provably LSAP-equivalent:
  500 random cases, max diff 4.4e-16)
- Node substitution = label mismatch + neighborhood edge term
- Our closed-form edge term `min(deg_u,deg_v)*edge_sub + |deg_u-deg_v|*edge_del` is
  **exactly the optimum** of upstream's per-pair edge LSAP under constant per-edge costs
  (verified: 0.0 diff on 500 edge cases + 300 random graph pairs, incl. non-zero edge sub)
- Node ins/del entries include incident-edge costs; same LSAP solver; same final formula
- With equalized costs our solver reproduces upstream **exactly** (0.0 over 300 pairs)

**Upstream bug NOT replicated (ours is correct):** upstream builds the insertion diagonal
from `g1` and deletion from `g2` (graphs swapped — changes results on ~3% of same-size
pairs, silently tiles wrong-length vectors otherwise). Ours uses the correct degrees.

**Library bug found incidentally:** `CostModel` docstring says substitution costs accept
floats but floats raise `TypeError` (`cost.py:77,89` — only callables work). To fix.

Benign: cost-model defaults differ by design (decoupled `CostModel` vs upstream's
euclidean-distance instantiation); dummy self-loops passed to `edge_sub_cost` are exact
while edge sub cost is constant per edge (true for all shipped cost models).

### GED-GW — VERIFIED

Cost matrices ((1−feature equality)×n2 cross-cost, adjacency scaling), FGW call
(`fused_gromov_wasserstein`, square_loss, alpha=1/3, armijo) and final estimate
`fgw_dist × 3/2`: identical to upstream (`models.py:628-696`).

**Edge case (behavior-changing):** with `dataset_name=None` and labeled graphs, local
builds a *per-graph* label encoding → the same label in g1/g2 can get different one-hot
columns → cross-cost 1 for identical labels. Upstream always uses the global 29-dim
one-hot. Passing `dataset_name="AIDS"` reproduces upstream exactly. Fix: document/require
`dataset_name` for labeled data, or build the encoding over the union of both graphs.

### GEDIOT — forward VERIFIED line-for-line; defaults/training discrepancies

GIN×3 (BN, train_eps), concat pre-activation + MLP, `tanh(costMatrix)`, OTLayer,
`soft = map ⊙ cost`, bias branch, `sigmoid(soft.sum()+bias)`, targets: all identical.

Deviations: (1) default `filters=(64,32,16)`, `bottleneck=(128,64,32)` vs upstream
`(128,64,32)` / `(16,8,4)` — checkpoints incompatible unless constructed to match;
(2) missing mapping loss (data limitation, see cross-cutting); (3) training protocol:
upstream trains on synthetic delta graphs (100 perturbations per graph) + real pairs,
batch 128 summed, weight_decay 5e-4; local real pairs only, batch 8 averaged, 1e-5.

### GPN — forward VERIFIED; identity corrected

**GPN = Graph Path Network** (the GNN of the Noah A*-beam framework), NOT "Graph Pair
Network" as our docstring claims. Citation: Yang & Zou, *Noah: Neural-optimized A* Search
Algorithm for Graph Edit Distance Computation*, ICDE 2021 (repo `pkumod/Noah-GED`;
included in chengqihao's repo). Docstring to fix.

Forward pass verified (GIN×3, cross MatchingModules, attention→NTN head). Deviations:
default `target_mode="exp"` vs upstream `"linear"` (GPN predicts `score × hb` upstream);
default dims differ.

### Shared modules

`AttentionModule`, `TensorNetworkModule` (math), `MatchingModule`, `OTLayer`/`PSinkhorn`
(5 log-domain iterations, dummy-row trick), `CostMatrixModule`, `GedMatrixModule`,
`fixed_mapping_loss`, `mapping_loss_mse`: all VERIFIED. Init: bias zeros vs upstream
xavier (affects from-scratch training only). Local adds benign n1>n2 handling where
upstream asserts.

### Cross-cutting finding: ground-truth data limitation

Three solvers (Grasp, GedGNN, TaGSim) train with a **reduced objective** because our
`True_GED` CSVs contain only scalar `true_ged`, while the upstream training loops use
(a) ground-truth **node mappings** (GEDIOT, Grasp, GedGNN, GedGNN-Full) and/or (b)
**decomposed GEDs** per component (TaGSim nc/in/ie). The mapping-augmented data exists
upstream (chengqihao's `TaGED.json`; ChengzhiPiao/GEDGNN data). Options: obtain richer
ground truth and implement the full losses, or document the library as value-supervised
only. Decision pending.



## Blind cross-verification (independent second reviewer, 2026-11-20)

An independent verifier (DeepSeek, via OpenCode) re-assessed the six most severe
findings WITHOUT access to this report. **Result: full convergence — all six
confirmed TRUE**, two of them with synthetic numerical reproduction.

Precision corrections adopted from the second reviewer:
- S3: the "~1.5x overcount" is the nested-edge-set special case, not a universal
  factor; in general the local edge term adds `|E1-E2|/2` even for full mappings
  where upstream adds 0.
- S4: EGSC `fc0` adds one net Linear+ReLU pair relative to upstream `fc1`
  (the block itself has two).
- S5(c): Grasp's final concat receives post-ReLU but pre-dropout tensors; only
  the pattern matrices receive post-dropout ones; `x5` is raw in both.

Additional findings from the second reviewer (to include in the fix wave):
1. `max_res = -1` in `_expand` assumes non-negative weights: with raw logits
   (S2) even a fixed root LB (S1) would still reject all subspaces — the two
   bugs compound.
2. `gedgnn_full.py` wraps the whole k-best call in `try/except Exception: pass`,
   silently masking failures — narrow it and log.

## Remaining work

- [x] All 13 solvers verified (2026-11-20)
- [x] Blind cross-verification by independent second reviewer (2026-11-20)
- [ ] Apply fix wave A+B (bugs + design alignment) with regression tests
- [ ] TaGED integration (richer ground truth: mappings + decomposed GEDs)
- [ ] Property-based tests with exact GED (networkx now; FORI later)
- [ ] HED port (row 14) and FORI exact solver (row 15)
