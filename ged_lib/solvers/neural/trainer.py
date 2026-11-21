"""Training loop for neural GED solvers.

Adapted from:
    https://github.com/chengqihao/GED-via-Optimal-Transport/blob/main/src/trainer.py

Provides a Trainer class that trains any neural GED model on a dataset,
saving checkpoints and logging metrics.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr, kendalltau

from ged_lib.dataset.loader import load_dataset
from ged_lib.solvers.neural.data_adapter import nx_to_neural_data, get_num_labels, get_label_map


class GEDTrainer:
    """Trainer for neural GED solvers.

    Handles data loading, batching, training loop, checkpointing,
    and evaluation against ground truth.

    When the mapping-augmented TaGED ground truth is available
    (``python scripts/download_taged.py``), the trainer applies the exact
    upstream losses (chengqihao/GED-via-Optimal-Transport, Grasp repo):

    - GedGNN / GedGNN-Full: ``fixed_mapping_loss(map, gt) + w * MSE``,
      w = 10.0 (AIDS/Linux) or 1.0 (IMDB)
    - GEDIOT / Grasp: ``(10 - 8) * fixed_mapping_loss(map, gt) + 8 * MSE``
    - TaGSim: per-component MSE against
      ``[exp(-nc/avg_v), exp(-in/avg_v), exp(-ie/avg_v)]``
    - GPN / SimGNN and everything else: plain MSE (upstream behaviour)

    Without TaGED the trainer behaves exactly as before (value-supervised
    MSE only).

    Args:
        model: A NeuralGEDSolver instance to train.
        dataset_name: One of "AIDS", "IMDB", "Linux".
        num_epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Adam optimizer learning rate.
        weight_decay: Adam weight decay.
        loss_weight: Fallback weight kept for backward compatibility.
        save_dir: Directory to save model checkpoints.
        device: Torch device string.
        mapping_loss: Use the TaGED mapping/component losses when the data
            is available. None (default) = auto-enable when TaGED.json
            exists for the dataset.
        max_pairs: Cap on the number of pairs per split (train/val/test).
        taged_root: Root containing True_GED/<DS>/TaGED.json (default: project).
    """

    def __init__(
        self,
        model,
        dataset_name: str = "AIDS",
        num_epochs: int = 20,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        loss_weight: float = 8.0,
        save_dir: str = "checkpoints",
        device: str = "cpu",
        mapping_loss: Optional[bool] = None,
        max_pairs: int = 5000,
        taged_root: Optional[str] = None,
    ) -> None:
        self.model = model
        self.dataset_name = dataset_name.upper()
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_weight = loss_weight
        self.device = torch.device(device)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_pairs = max_pairs
        self.taged_root = taged_root
        self._mapping_loss_requested = mapping_loss

        self.used_mapping_loss = False
        self.used_component_loss = False

        self.num_labels = 1
        self._graph_cache: dict = {}  # gid -> {edge_index, features, n, m}
        self._load_data()

    def _load_data(self) -> None:
        """Load dataset and prepare graph pairs."""
        graphs, ground_truth = load_dataset(self.dataset_name)

        if self.dataset_name == "AIDS":
            self.num_labels = 29
        else:
            self.num_labels = get_num_labels(graphs, self.dataset_name)

        self.label_map = get_label_map(self.dataset_name)

        self.graphs = graphs
        self.ground_truth = ground_truth
        self.graph_ids = sorted(graphs.keys())

        n = len(self.graph_ids)
        train_split = int(n * 0.6)
        val_split = int(n * 0.8)

        self.train_ids = self.graph_ids[:train_split]
        self.val_ids = self.graph_ids[train_split:val_split]
        self.test_ids = self.graph_ids[val_split:]

        gt_dict = {}
        for id1, id2, ged in self.ground_truth:
            key = (min(id1, id2), max(id1, id2))
            gt_dict[key] = ged
        self.gt_dict = gt_dict

        self.train_pairs = self._make_pairs(self.train_ids)
        self.val_pairs = self._make_pairs(self.val_ids)
        self.test_pairs = self._make_pairs(self.test_ids)

        self._taged = self._load_taged()
        if self._mapping_loss_requested is None:
            self.mapping_loss = self._taged is not None
        else:
            self.mapping_loss = self._mapping_loss_requested and self._taged is not None
            if self._mapping_loss_requested and self._taged is None:
                print(
                    "WARNING: mapping_loss=True but TaGED.json is not available "
                    "(python scripts/download_taged.py); falling back to MSE only."
                )
        if self.mapping_loss:
            print(
                "TaGED ground truth loaded: mapping/component losses ENABLED "
                f"({len(self._taged)} pairs)"
            )

        print(
            f"Loaded {self.dataset_name}: {n} graphs, "
            f"{len(self.train_pairs)} train pairs, "
            f"{len(self.val_pairs)} val pairs, "
            f"{len(self.test_pairs)} test pairs"
        )

    def _load_taged(self):
        """Load the TaGED mapping ground truth if available (None otherwise)."""
        from ged_lib.dataset.loader import load_taged

        try:
            return load_taged(self.dataset_name, root=self.taged_root)
        except FileNotFoundError:
            return None

    def _make_pairs(self, graph_ids: list, max_pairs: Optional[int] = None) -> list:
        """Create graph pairs from a list of graph IDs using ground truth."""
        if max_pairs is None:
            max_pairs = self.max_pairs
        pairs = []
        id_set = set(graph_ids)
        for (id1, id2), ged in self.gt_dict.items():
            if id1 in id_set and id2 in id_set:
                pairs.append((id1, id2, ged))
                if len(pairs) >= max_pairs:
                    break
        return pairs

    def _get_graph_tensors(self, gid: int) -> dict:
        """Get cached tensor representation for a graph, computing if needed.

        Caches edge_index and features per graph to avoid redundant
        conversion across epochs and pairs.

        Args:
            gid: Graph ID.

        Returns:
            Dict with edge_index, features, n, m tensors/values.
        """
        if gid in self._graph_cache:
            return self._graph_cache[gid]

        from ged_lib.solvers.neural.data_adapter import _build_edge_index, _build_features

        g = self.graphs[gid]
        nodes = sorted(g.nodes())
        n = len(nodes)
        m = g.number_of_edges()

        edge_index = _build_edge_index(g, nodes, self.device)
        features = _build_features(g, nodes, self.num_labels, self.device, label_map=self.label_map)

        cached = {"edge_index": edge_index, "features": features, "n": n, "m": m}
        self._graph_cache[gid] = cached
        return cached

    def _pack_pair(self, id1: int, id2: int, true_ged: float) -> dict:
        """Prepare a single graph pair for the model.

        Uses cached per-graph tensors (edge_index, features) and only
        computes pair-specific fields (target, hb, avg_v) per call.
        """
        t1 = self._get_graph_tensors(id1)
        t2 = self._get_graph_tensors(id2)

        n1, n2 = t1["n"], t2["n"]
        m1, m2 = t1["m"], t2["m"]
        avg_v = (n1 + n2) / 2.0
        hb = max(n1, n2) + max(m1, m2)

        data = {
            "edge_index_1": t1["edge_index"],
            "edge_index_2": t2["edge_index"],
            "features_1": t1["features"],
            "features_2": t2["features"],
            "n1": n1,
            "n2": n2,
            "avg_v": avg_v,
            "hb": hb,
        }

        if true_ged is not None:
            if self.model.target_mode == "exp":
                data["target"] = (
                    torch.exp(torch.tensor([-true_ged / avg_v])).float().to(self.device)
                )
            elif self.model.target_mode == "linear":
                data["target"] = torch.tensor([true_ged / hb]).float().to(self.device)

        if self._taged is not None:
            from ged_lib.dataset.loader import taged_lookup, taged_mapping_matrix

            entry = taged_lookup(self._taged, id1, id2)
            if entry is not None and entry[4]:
                try:
                    matrix = taged_mapping_matrix(entry[4], n1, n2)
                    data["gt_map"] = torch.from_numpy(matrix).to(self.device)
                    data["ta_components"] = (entry[1], entry[2], entry[3])
                except ValueError:
                    pass  # orientation mismatch for this pair; MSE only

        return data

    def train(self) -> dict:
        """Run the training loop.

        Returns:
            Dictionary with training history (losses and metrics per epoch).
        """
        from ged_lib.solvers.neural.ged_matrix import fixed_mapping_loss

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        history = {"train_loss": [], "val_mae": [], "val_spearman": []}
        is_tagsim = hasattr(self.model, "tensor_nc")

        # Upstream per-model loss regimes (chengqihao trainer.setup_model /
        # process_batch; Grasp trainer.py).
        model_name = type(self.model).__name__
        if model_name in ("GedGNNSolver", "GEDGNNFullSolver"):
            # fixed_mapping_loss + weight*MSE, weight 10.0 (AIDS/Linux) / 1.0 (IMDB)
            if self.dataset_name in ("AIDS", "LINUX"):
                mse_weight, map_coef = 10.0, 1.0
            else:
                mse_weight, map_coef = 1.0, 1.0
        elif model_name in ("GEDIOTSolver", "GraspSolver"):
            # (10 - 8) * fixed_mapping_loss + 8 * MSE
            mse_weight, map_coef = 8.0, 2.0
        else:
            mse_weight, map_coef = None, None

        for epoch in range(self.num_epochs):
            self.model.train()
            np.random.shuffle(self.train_pairs)
            epoch_loss = 0.0
            n_batches = 0

            for batch_start in range(0, len(self.train_pairs), self.batch_size):
                batch = self.train_pairs[batch_start : batch_start + self.batch_size]
                optimizer.zero_grad()
                batch_loss = torch.tensor(0.0, device=self.device)

                for id1, id2, true_ged in batch:
                    data = self._pack_pair(id1, id2, true_ged)
                    target = data["target"]

                    if is_tagsim:
                        prediction, _ = self.model(data)
                        components = data.get("ta_components")
                        if (
                            self.mapping_loss
                            and components is not None
                            and self.model.target_mode == "exp"
                            and prediction.shape[0] == 3
                        ):
                            # Upstream: MSE([exp(-nc/avg), exp(-in/avg),
                            # exp(-ie/avg)], [s_nc, s_in, s_ie])
                            nc, in_, ie = components
                            ta_target = torch.exp(
                                torch.tensor(
                                    [-nc / data["avg_v"], -in_ / data["avg_v"], -ie / data["avg_v"]],
                                    device=self.device,
                                )
                            ).float()
                            batch_loss = batch_loss + F.mse_loss(
                                ta_target, prediction
                            )
                            self.used_component_loss = True
                        elif prediction.shape[0] == 3:
                            # Fallback (no TaGED): loss on the summed score
                            batch_loss = batch_loss + F.mse_loss(
                                prediction.sum().unsqueeze(0), target
                            )
                        else:
                            batch_loss = batch_loss + F.mse_loss(target, prediction)
                    else:
                        output = self.model(data)
                        prediction = output[0]
                        gt_map = data.get("gt_map")
                        if (
                            self.mapping_loss
                            and gt_map is not None
                            and map_coef is not None
                            and len(output) >= 3
                        ):
                            mapping = output[2]
                            if mapping.shape == gt_map.shape:
                                batch_loss = batch_loss + (
                                    map_coef * fixed_mapping_loss(mapping, gt_map)
                                    + mse_weight * F.mse_loss(target, prediction)
                                )
                                self.used_mapping_loss = True
                            else:
                                batch_loss = batch_loss + F.mse_loss(target, prediction)
                        else:
                            batch_loss = batch_loss + F.mse_loss(target, prediction)

                batch_loss = batch_loss / len(batch)
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            val_mae, val_rho = self._evaluate(self.val_pairs[:500])

            history["train_loss"].append(avg_loss)
            history["val_mae"].append(val_mae)
            history["val_spearman"].append(val_rho)

            print(
                f"Epoch {epoch + 1}/{self.num_epochs}: "
                f"loss={avg_loss:.6f}, val_mae={val_mae:.4f}, val_rho={val_rho:.4f}"
            )

            if (epoch + 1) % 5 == 0 or epoch == self.num_epochs - 1:
                self._save_checkpoint(epoch + 1)

        return history

    def _evaluate(self, pairs: list) -> tuple:
        """Evaluate model on a set of graph pairs.

        Returns:
            Tuple of (MAE, Spearman rho).
        """
        self.model.eval()
        preds = []
        gts = []

        with torch.no_grad():
            for id1, id2, true_ged in pairs:
                data = self._pack_pair(id1, id2, true_ged)
                output = self.model(data)
                pre_ged = output[1]
                preds.append(pre_ged)
                gts.append(true_ged)

        if len(preds) < 2:
            return 0.0, 0.0

        preds_arr = np.array(preds)
        gts_arr = np.array(gts)
        mae = float(np.mean(np.abs(preds_arr - gts_arr)))
        rho = spearmanr(preds_arr, gts_arr)[0] if np.std(preds_arr) > 0 else 0.0
        if np.isnan(rho):
            rho = 0.0

        return mae, rho

    def test(self) -> dict:
        """Evaluate on the test set.

        Returns:
            Dictionary with test metrics.
        """
        self.model.eval()
        preds = []
        gts = []
        times = []

        with torch.no_grad():
            for id1, id2, true_ged in self.test_pairs:
                data = self._pack_pair(id1, id2, true_ged)
                t0 = time.perf_counter()
                output = self.model(data)
                t1 = time.perf_counter()
                preds.append(output[1])
                gts.append(true_ged)
                times.append(t1 - t0)

        preds_arr = np.array(preds)
        gts_arr = np.array(gts)
        times_arr = np.array(times)

        mae = float(np.mean(np.abs(preds_arr - gts_arr)))
        mse = float(np.mean((preds_arr - gts_arr) ** 2))
        rho = spearmanr(preds_arr, gts_arr)[0] if np.std(preds_arr) > 0 else 0.0
        tau = kendalltau(preds_arr, gts_arr)[0] if np.std(preds_arr) > 0 else 0.0
        if np.isnan(rho):
            rho = 0.0
        if np.isnan(tau):
            tau = 0.0

        results = {
            "model": self.model.__class__.__name__,
            "dataset": self.dataset_name,
            "n_pairs": len(preds),
            "mae": mae,
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "spearman_r": float(rho),
            "kendall_tau": float(tau),
            "mean_time_s": float(np.mean(times_arr)),
        }

        print(f"\n=== Test Results ({results['model']} on {results['dataset']}) ===")
        for k, v in results.items():
            if k not in ("model", "dataset"):
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        return results

    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint."""
        path = (
            self.save_dir / f"{self.model.__class__.__name__}_{self.dataset_name}_epoch{epoch}.pt"
        )
        torch.save(self.model.state_dict(), path)
        print(f"  Saved checkpoint: {path}")
