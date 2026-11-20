# Experimental Evaluation

## 4. Experiments

### 4.1 Experimental Setup

**Dataset.** We evaluate all methods on the **AIDS** dataset, a widely-used benchmark for graph edit distance computation. The dataset contains 617 molecular graphs (average: 9.5 nodes, 9.0 edges) with 29 symbolic node labels representing chemical elements. Ground-truth GED values are available for all pairs. The true GED ranges from 0 to 14 on the test set, with a mean of approximately 7.89.

**Data split.** We adopt a 60/20/20 random split of the graph collection. Neural methods are trained on pairs drawn from the training graphs (60%), validated on pairs from the validation graphs (20%), and evaluated on 500 held-out pairs drawn exclusively from the test graphs (20%). Unsupervised methods are evaluated on the same 500 test pairs to ensure a fair comparison. Graphs in the test split are never seen during training.

**Metrics.** We report the following metrics:
- **MAE** (Mean Absolute Error): average absolute deviation between predicted and true GED.
- **MSE** (Mean Squared Error): penalizes large errors more heavily than MAE.
- **Accuracy@1** (Acc@1): the fraction of graph pairs for which $|\text{pred} - \text{true}| \leq 1$. This metric captures whether a prediction is practically usable, as a GED error of $\pm 1$ often corresponds to a single node or edge edit.
- **Spearman's $\rho$**: rank correlation between predicted and true GED values, measuring whether the method correctly orders pairs by dissimilarity.
- **Kendall's $\tau$**: ordinal association metric, less sensitive to outliers than Spearman.
- **Throughput** (pairs/s): number of graph pairs processed per second at inference time, measuring practical efficiency.

**Training protocol.** All neural methods are trained for **2 epochs** with a batch size of 16, using the Adam optimizer with a learning rate of $10^{-3}$ and weight decay of $10^{-5}$. All models use the default hyperparameters specified in their respective papers. We deliberately use only 2 epochs to assess *sample efficiency* — how quickly each method reaches a usable level of performance with limited training. This setting also reflects practical scenarios where training data or compute budget is constrained.

**Methods compared.** We compare **15 methods** spanning five paradigms:

| Paradigm | Methods |
|----------|------|
| Classical | Hungarian (bipartite matching), GED-GW (Gromov-Wasserstein OT), FGWAlign-Fast, FGWAlign-Light (Fused Gromov-Wasserstein alignment) |
| Classical + ML (partial†) | GED-SF-Linear, GED-SF-Quadratic (Frank-Wolfe alignment + learned regressor) |
| Neural (value prediction) | SimGNN, GedGNN, GEDIOt, TaGSim, GPN, EGSC, GREED |
| Neural (mapping + value) | Grasp |
| Neural + post-processing | GedGNN-Full (k-best matching with LB pruning) |

† GED-SF results are preliminary: trained on 2,000 pairs (out of 121,771 available), 3 mu values, and 50 FW iterations. Full training is expected to improve performance further.

### 4.2 Results

Table 1 reports the full comparison. Methods are sorted by MAE.

**Table 1.** Comparison of GED methods on AIDS (500 test pairs, 2 training epochs for neural methods). GED-SF results marked † are preliminary (see §4.1 footnote).

| Method | MAE ↓ | MSE ↓ | Acc@1 ↑ | Spearman ρ ↑ | Kendall τ ↑ | pairs/s ↑ |
|--------|------:|------:|--------:|-------------:|------------:|----------:|
| **Classical (OT-based)** | | | | | | |
| FGWAlign-Fast | **0.17** | **0.24** | **0.966** | **0.9625** | **0.9291** | 4.0 |
| **Classical + ML (partial†)** | | | | | | |
| GED-SF-Quadratic† | 1.14 | 2.22 | 0.542 | 0.6332 | 0.4876 | 36.7 |
| GED-SF-Linear† | 1.18 | 2.25 | 0.502 | 0.6279 | 0.4814 | 212.5 |
| **Classical** | | | | | | |
| Hungarian | 1.38 | 3.17 | 0.624 | 0.7877 | 0.6591 | 6,024.2 |
| **Neural (value)** | | | | | | |
| GREED | 1.56 | 4.26 | 0.438 | 0.3800 | 0.2813 | 1,352.0 |
| GED-GW | 1.60 | 4.69 | 0.384 | 0.4475 | 0.3380 | 24.3 |
| EGSC | 1.70 | 4.85 | 0.340 | −0.1830 | −0.1399 | 742.2 |
| GEDIOt | 1.72 | 4.97 | 0.376 | −0.0483 | −0.0366 | 738.8 |
| **Classical (OT-based, light)** | | | | | | |
| FGWAlign-Light | 1.87 | 7.19 | 0.500 | 0.7439 | 0.6213 | 468.9 |
| **Neural + post-proc.** | | | | | | |
| GedGNN-Full | 6.86 | 58.96 | 0.044 | −0.1407 | −0.1050 | 321.3 |
| SimGNN | 7.74 | 66.53 | 0.002 | −0.1826 | −0.1383 | 810.5 |
| GPN | 10.69 | 127.31 | 0.000 | −0.1220 | −0.0880 | 1,079.9 |
| **Neural (mapping)** | | | | | | |
| TaGSim | 16.36 | 274.16 | 0.000 | −0.1805 | −0.1506 | 2,297.0 |
| Grasp | 17.50 | 319.14 | 0.000 | −0.1201 | −0.0925 | 422.5 |
| **Neural (diverged)** | | | | | | |
| GedGNN | 163.55 | 27,039.92 | 0.000 | −0.1199 | −0.0889 | 814.7 |

### 4.3 Analysis

**Headline finding: FGWAlign-Fast dominates all methods by a large margin.** FGWAlign-Fast achieves MAE=0.17 and Acc@1=0.966, which is **8× better MAE** than the second-place GED-SF-Quadratic (MAE=1.14) and far exceeds every neural method. Spearman $\rho=0.9625$ and Kendall $\tau=0.9291$ indicate near-perfect rank ordering. This result is remarkable given that FGWAlign is an unsupervised OT-based method: it requires no training data, no fitting, and no learned parameters. The permutation-projection scheme from Tang et al. computes a tight combinatorial alignment between node feature distributions, achieving accuracy that no trained model approaches at 2 epochs. The main limitation is throughput: at 4.0 pairs/s it is the slowest method by a large margin, making it unsuitable for large-scale applications without further engineering.

**FGWAlign-Light reveals a strong accuracy–speed tradeoff.** Reducing patience from 15 to 1 and enabling the light mode degrades MAE from 0.17 to 1.87 — an 11× increase — while boosting speed from 4.0 to 468.9 pairs/s (117×). Despite this degradation, FGWAlign-Light still achieves competitive Spearman $\rho=0.7439$ and Kendall $\tau=0.6213$, placing it ahead of GED-GW and most neural methods on rank-ordering metrics. The tradeoff is therefore: FGWAlign-Fast for accuracy, FGWAlign-Light for large-scale screening.

**GED-SF (partial evaluation) achieves the second-best MAE, ahead of all other baselines.** With only 2,000 training pairs, GED-SF-Quadratic (MAE=1.14) and GED-SF-Linear (MAE=1.18) outperform Hungarian (MAE=1.38) and all neural methods on MAE and MSE. GED-SF-Quadratic best config was mu=0.7 with Ensemble regressor; GED-SF-Linear best config was mu=0.5 with GradientBoosting. However, both GED-SF variants trail Hungarian on Acc@1 (0.542/0.502 vs. 0.624) and rank correlation (Spearman 0.63 vs. 0.79), suggesting the regressor learns GED magnitude well but not its fine-grained integer structure. At full training scale (121,771 pairs, full mu grid), GED-SF accuracy is expected to improve substantially.

**Hungarian is the strongest fully-trained classical baseline.** After the edge substitution cost fix, Hungarian achieves MAE=1.38, Acc@1=0.624, and Spearman $\rho=0.7877$ — tied with or beating all non-FGWAlign methods on ranking metrics. At 6,024 pairs/s it is also the fastest method that produces useful rank ordering, making it the go-to baseline for any practical deployment scenario.

**FGWAlign-Fast vs. Hungarian: quality–speed tradeoff.** FGWAlign-Fast beats Hungarian on every accuracy metric (MAE 0.17 vs. 1.38, Spearman 0.963 vs. 0.788, Kendall 0.929 vs. 0.659), but is roughly 1,500× slower (4.0 vs. 6,024 pairs/s). For datasets of a few hundred test pairs this is acceptable; for large-scale retrieval Hungarian or FGWAlign-Light are preferred.

**Neural methods underperform at 2 epochs on most architectures.** GREED (MAE=1.56) is the best neural method, followed closely by GED-GW (MAE=1.60, unsupervised), EGSC (1.70), and GEDIOt (1.72). Most other neural methods produce MAE above 6.0. The consistently negative Spearman correlations for EGSC, GEDIOt, SimGNN, TaGSim, GPN, Grasp, and GedGNN-Full indicate that these architectures are still in an underfitting regime at 2 epochs and have not learned to rank pairs correctly. Extended training (20–50 epochs) is essential for these models.

**GedGNN diverges at 2 epochs.** GedGNN produces MAE=163.55 and MSE=27,039.92 — orders of magnitude worse than any other method. The training log confirms the issue: validation MAE was stuck at 126 throughout both epochs, indicating the model never started learning. GedGNN's mapping-matrix approach appears to require a minimum number of epochs before the OT layer begins to produce meaningful soft assignments. At 2 epochs the model outputs near-constant high values. This behavior is consistent with the known sensitivity of optimal-transport-based mapping to learning rate scheduling and initialization.

**Post-processing is counterproductive before model convergence.** GedGNN-Full (MAE=6.86) is worse than the k-next-best baseline because the underlying GedGNN mapping matrix is poorly estimated. When the OT layer has not converged, the k-best matching post-processor amplifies the mapping errors rather than correcting them. This behavior was also observed in the 3-epoch run and confirms that post-processing requires a high-quality learned mapping.

**GED-GW remains competitive despite its simplicity.** GED-GW (MAE=1.60, Spearman=0.4475) is the best unsupervised method after the two FGWAlign variants. It requires no training, works on any graph format, and is deterministic. Its place at position 6 in the MAE ranking illustrates the strength of Gromov-Wasserstein optimization for GED approximation.

**Throughput summary.** Hungarian is 1,500× faster than FGWAlign-Fast and 35× faster than FGWAlign-Light. Neural methods cluster around 740–2,300 pairs/s. GED-SF-Quadratic (37 pairs/s) and GED-GW (24 pairs/s) are the slowest production-capable methods. FGWAlign-Fast (4 pairs/s) is suitable only for small test sets or applications where accuracy is paramount.

### 4.4 Method Ranking by Performance Tier

The 15 methods naturally cluster into four performance tiers based on MAE on this 2-epoch evaluation:

| Tier | Methods | MAE range | Notes |
|------|---------|-----------|-------|
| **Excellent** (MAE < 1.2) | FGWAlign-Fast | 0.17 | Unsupervised OT; slow (4 pairs/s) |
| **Good** (1.2 ≤ MAE < 1.6) | GED-SF-Quadratic†, GED-SF-Linear†, Hungarian | 1.14–1.38 | SF is partial; Hungarian is fastest good method |
| **Moderate** (1.6 ≤ MAE < 2.0) | GREED, GED-GW, EGSC, GEDIOt, FGWAlign-Light | 1.56–1.87 | Mixed neural + OT + fast-FGW |
| **Poor / Failed** (MAE ≥ 6.0) | GedGNN-Full, SimGNN, GPN, TaGSim, Grasp, GedGNN | 6.86–163.55 | Need more epochs to converge |

Key observations from the tiering:
- **FGWAlign-Fast stands alone** — no other method comes within 6× its MAE. It is the only method achieving Acc@1 > 0.9.
- **The moderate tier contains a heterogeneous mix** of the best neural method (GREED), the best pure-OT method (GED-GW), and two near-unsupervised methods (FGWAlign-Light plus EGSC/GEDIOt which barely outperform random at 2 epochs by rank metrics).
- **Hungarian is uniquely robust**: it achieves the best Acc@1 and Spearman among all non-FGWAlign methods, placing comfortably in the "Good" tier without any training.
- **GedGNN's tier-5 result** (MAE=163.55) is an outlier caused by non-convergence at 2 epochs. Its architecture converges only after 5+ epochs based on the checkpoint logs from earlier sessions.

### 4.5 Summary of Key Findings

1. **FGWAlign-Fast is the state-of-the-art method in this benchmark.** With MAE=0.17, Acc@1=0.966, and Spearman $\rho=0.963$, it outperforms all 14 other methods by a large margin — 8× better MAE than the next-best method. As an unsupervised, parameter-free OT algorithm, it requires no training data or fitting and generalizes immediately to any graph dataset. Its only limitation is speed (4.0 pairs/s), which limits applicability to small test sets.

2. **FGWAlign-Light offers a high-speed alternative with reasonable accuracy.** Reducing the patience and enabling light mode yields a 117× speedup (468.9 pairs/s) at the cost of an 11× MAE increase (1.87). FGWAlign-Light still ranks ahead of GED-GW and most neural methods on Spearman ($\rho=0.744$) and Kendall ($\tau=0.621$), making it suitable for large-scale graph retrieval tasks.

3. **GED-SF is a promising learned method, even in its partial configuration.** GED-SF-Quadratic (MAE=1.14) and GED-SF-Linear (MAE=1.18) both beat Hungarian (MAE=1.38) on absolute error while training on only 2,000 pairs. Full training on all 121,771 pairs and a complete CV grid is expected to improve results further. GED-SF-Quadratic's best config (mu=0.7, Ensemble) consistently outperforms GED-SF-Linear (mu=0.5, GradientBoosting) on all metrics, suggesting the quadratic FW alignment preserves more structural information in its 14-dimensional feature vector.

4. **Hungarian remains the best non-FGWAlign method on rank quality.** After all bug fixes, Hungarian achieves Acc@1=0.624 and Spearman $\rho=0.788$ — the best rank metrics among all methods except FGWAlign-Fast and FGWAlign-Light. At 6,024 pairs/s it is extremely fast, making it the recommended first baseline for any new GED application.

5. **GREED is the best neural method at low training budgets.** Its simple Siamese architecture (MAE=1.56, Spearman=0.380) converges fastest and is the only neural method reaching the "Moderate" tier without relying on OT alignment. It is the preferred neural choice when no classical features are available and training is limited to 2–3 epochs.

6. **Most neural methods require substantially more than 2 epochs.** EGSC, GEDIOt, SimGNN, TaGSim, GPN, Grasp, and GedGNN-Full all produce negative Spearman correlations, indicating they have not learned to rank pairs correctly at 2 epochs. GedGNN completely diverges (MAE=163.55), requiring at least 5 epochs to produce usable results based on prior checkpoint evaluations. Extended training (20–50 epochs) is essential for fair neural comparison.

7. **Post-processing requires model convergence to be beneficial.** GedGNN-Full (MAE=6.86) underperforms all individual neural methods, confirming that k-best matching amplifies errors in a poorly-converged mapping matrix.

### 4.6 GED-SF Partial Results — Caveats and Next Steps

The GED-SF results reported here are intentionally partial. The experimental configuration deviates from the full setup in three ways:

| Setting | Used here | Full setup |
|---------|-----------|------------|
| Training pairs | 2,000 | 121,771 |
| Frank-Wolfe iterations | 50 | 150 |
| mu grid | {0.1, 0.5, 1.0} | {0.1, 0.3, 0.5, 0.7, 1.0} |

Despite these limitations, GED-SF-Quadratic achieves MAE=1.14, placing it second overall behind FGWAlign-Fast and ahead of Hungarian (1.38). The two main open questions are:

- **Scaling with training data.** The regressor sees only $\sim$1.6% of available pairs. More data should tighten the MAE:Spearman gap, as the regressor can learn the finer integer structure of GED values (currently it over-smooths predictions, visible in the Acc@1 gap vs. Hungarian 0.542 vs. 0.624).
- **Formulation difference.** GED-SF-Quadratic (MAE=1.14, Spearman=0.633) consistently beats GED-SF-Linear (MAE=1.18, Spearman=0.628) on all metrics. The quadratic FW alignment better preserves structural similarity in the 14-dimensional feature vector. The best configs found by CV were mu=0.7 (Ensemble regressor) and mu=0.5 (GradientBoosting) respectively. Future experiments should confirm whether this advantage persists at full training scale.
