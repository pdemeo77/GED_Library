| # | Solver (nome interno) | Varianti | Metodo | Paper (autore, anno, titolo/venue) | Repo GitHub |
| --- | --- | --- | --- | --- | --- |
| 1 | hungarian — HungarianSolver | — | classico | Riesen & Bunke (2009), Approximate GED computation by means of bipartite graph matching, Image and Vision Computing 27(7) |  https://github.com/priba/aproximated_ged (il repo contiene anche HED → riga 14) |
| 2 | gedgw — GEDGWSolver | — | OT unsupervised | Cheng et al. (2025), SIGMOD ⬜ titolo esatto (arXiv:2412.18857) | chengqihao/GED-via-Optimal-Transport |
| 3 | fgwalign — FGWAlignSolver | Fast / Light | OT unsupervised | Tang et al. (2024), Fast Graph Edit Distance via Optimal Transport ⬜ venue | squareRoot3/FGWAlign |
| 4 | gedsf — GEDSFSolver | Linear / Quadratic | FW + regressione ML | Algoritmo originale in sviluppo (P. De Meo) — paper in preparazione | ⬜ nostro repo, link in arrivo |
| 5 | gediot — GEDIOTSolver | — | neurale | Cheng et al. (2025), SIGMOD ⬜ titolo esatto (arXiv:2412.18857) | chengqihao/GED-via-Optimal-Transport |
| 6 | gedgnn — GedGNNSolver | — | neurale | Piao et al. (2023) ⬜ titolo | ChengzhiPiao/GEDGNN |
| 7 | simgnn — SimGNNSolver | — | neurale | Bai et al. (2019), SimGNN: A Neural Network Approach to Fast Graph Similarity Computation, AAAI, arXiv:1808.05689 |  https://github.com/benedekrozemberczki/SimGNN |
| 8 | tagsim — TaGSimSolver | — | neurale | Bai et al., TaGSim: Type-aware Graph Similarity Learning ⬜ venue | jiyangbai/TaGSim |
| 9 | gpn — GPNSolver | — | neurale | **Graph Path Network** — Yang & Zou (2021), Noah: Neural-optimized A* Search Algorithm for Graph Edit Distance Computation, ICDE 2021 | pkumod/Noah-GED (via chengqihao/GED-via-Optimal-Transport) |
| 10 | egsc — EGSCSolver | — | neurale | Qin et al. (2021), NeurIPS ⬜ titolo | canqin001/Efficient_Graph_Similarity_Computation |
| 11 | greed — GREEDSolver | — | neurale | Ranjan et al. (2022), GREED: A Neural Framework for Learning Graph Distance Functions, NeurIPS | idea-iitd/greed |
| 12 | gedgnnfull — GEDGNNFullSolver | — | neurale + k-best | Piao et al. (2023), PVLDB ⬜ titolo | ChengzhiPiao/GEDGNN |
| 13 | grasp — GraspSolver | — | neurale | Lee et al. ⬜ titolo/venue | JunkyuLee1/Grasp |
| 14 | hed — HEDSolver (da aggiungere) | — | classico | Fischer et al. (2015), Approximation of graph edit distance based on Hausdorff matching ⬜ venue | https://github.com/priba/aproximated_ged |
| 15 | fori — FORISolver (da aggiungere, esatto) | — | **exact ILP** | D'Ascenzo, Meffert, Mutzel, Rossi (2025), Enhancing GED Computation: Stronger and Orientation-based ILP Formulations, PVLDB 18(11) — implementazione nostra via scipy.optimize.milp (paper in docs/papers) | 
https://github.com/meffertj/FORI-GED (solo come riferimento; implementazione indipendente) |
| 16 | DiffGED | — | diffusion | Wei Huang, Hanchen Wang, Dong Wen, Wenjie Zhang, Ying Zhang, Xuemin Lin:
DiffGED: Computing Graph Edit Distance via Diffusion-based Graph Matching. CoRR abs/2503.18245 (2025) | Disponibile localmente|
| 17 | GELATO | — | autoregressivo | P Pellizzoni, T Schulz, K Borgwardt. Gelato: Graph edit distance via autoregressive neural combinatorial optimization International Conference on Learning Representations 2026, 14972-14999 | https://github.com/BorgwardtLab/Gelato|
| 18 | GraphEDX | — | neurale | Eeshaan Jain, Indradyumna Roy, Saswat Meher, Soumen Chakrabarti, Abir De. Graph Edit Distance with General Costs Using Neural Set Divergence. NeurIPS 2024  | https://github.com/structlearning/GraphEdX|
| 19 | AppBMao | — | euristico | Mouyi Xu, Lijun Chang: Graph Edit Distance Estimation: A New Heuristic and A Holistic Evaluation of Learning-based Methods. Proc. ACM Manag. Data 3(3): 167:1-167:24 (2025)  | https://github.com/XMYMouyiXu/GED-EXP |

