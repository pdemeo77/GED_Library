"""GEDGNN-matching: GedGNN with k-best matching post-processing.

Reference:
    Piao, C. et al. (2023).
    Computing Graph Edit Distance via Neural Graph Matching. PVLDB 2023.
    https://github.com/ChengzhiPiao/GEDGNN

Combines the GedGNN neural model with k-best bipartite matching post-processing.
The neural model predicts a mapping matrix, then the k-best matching solver
finds the top-k bipartite matchings, computes actual GED for each using
adjacency/feature comparison, and prunes with lower bounds.

This produces the "GEDGNN-matching" predictions from the paper, which are
generally more accurate than the neural score alone ("GEDGNN-value").
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GINConv

import networkx as nx

from ged_lib.solvers import register_solver
from ged_lib.solvers.neural.models import NeuralGEDSolver
from ged_lib.solvers.neural.ged_matrix import GedMatrixModule
from ged_lib.solvers.neural.layers import AttentionModule, TensorNetworkModule
from ged_lib.solvers.neural.data_adapter import nx_to_neural_data
from ged_lib.solvers.neural.kbest_matching import KBestMSolver


class _GINBackbone(nn.Module):
    """3-layer GIN backbone with multi-scale feature concatenation."""

    def __init__(self, num_labels: int, filters: tuple = (64, 32, 16)):
        super().__init__()
        f1, f2, f3 = filters
        self.num_labels = num_labels

        nn1 = nn.Sequential(
            nn.Linear(num_labels, f1),
            nn.ReLU(),
            nn.Linear(f1, f1),
            nn.BatchNorm1d(f1, track_running_stats=False),
        )
        nn2 = nn.Sequential(
            nn.Linear(f1, f2),
            nn.ReLU(),
            nn.Linear(f2, f2),
            nn.BatchNorm1d(f2, track_running_stats=False),
        )
        nn3 = nn.Sequential(
            nn.Linear(f2, f3),
            nn.ReLU(),
            nn.Linear(f3, f3),
            nn.BatchNorm1d(f3, track_running_stats=False),
        )
        self.conv1 = GINConv(nn1, train_eps=True)
        self.conv2 = GINConv(nn2, train_eps=True)
        self.conv3 = GINConv(nn3, train_eps=True)
        self.f3 = f3

        k = num_labels + f1 + f2 + f3
        self.mlp = nn.Sequential(
            nn.Linear(k, k * 2),
            nn.ReLU(),
            nn.Linear(k * 2, k),
            nn.ReLU(),
            nn.Linear(k, f3),
        )

    def forward(self, edge_index: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x_list = features
        x = self.conv1(features, edge_index)
        x_list = torch.cat([x_list, x], dim=1)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x_list = torch.cat([x_list, x], dim=1)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x_list = torch.cat([x_list, x], dim=1)
        return self.mlp(x_list)


def _build_adj_dense(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    A = torch.zeros(n, n)
    src, dst = edge_index[0].cpu(), edge_index[1].cpu()
    A[src, dst] = 1.0
    return A


@register_solver("gedgnn_full")
class GEDGNNFullSolver(NeuralGEDSolver):
    """GEDGNN-matching: GedGNN with k-best matching post-processing.

    PVLDB 2023. Uses the same neural architecture as GedGNN (GIN + mapping/cost
    matrices + bias), then applies k-best bipartite matching with GED lower bound
    pruning to find the actual edit path with minimum GED.

    The post-processing converts the predicted mapping matrix into discrete
    matchings and computes exact GED from adjacency/feature comparison.

    Args:
        num_labels: Number of one-hot label dimensions.
        filters: Tuple of GIN filter sizes.
        hidden_dim: Hidden dimension for GedMatrixModule.
        tensor_neurons: Number of tensor network neurons.
        bottleneck: Bottleneck layer sizes for bias MLP.
        target_mode: "exp" or "linear" normalization.
        k_best: Number of best matchings to explore in post-processing.
        device: Torch device string.
    """

    def __init__(
        self,
        num_labels: int = 29,
        filters: tuple = (64, 32, 16),
        hidden_dim: int = 16,
        tensor_neurons: int = 16,
        bottleneck: tuple = (128, 64, 32),
        target_mode: str = "linear",
        k_best: int = 10,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.num_labels = max(num_labels, 1)
        self.target_mode = target_mode
        self.device = torch.device(device)
        self.k_best = k_best
        f1, f2, f3 = filters
        self.filters_3 = f3

        self.gin = _GINBackbone(self.num_labels, filters)
        self.map_matrix = GedMatrixModule(f3, hidden_dim)
        self.cost_matrix_mod = GedMatrixModule(f3, hidden_dim)

        self.attention = AttentionModule(f3)
        self.tensor_network = TensorNetworkModule(f3, tensor_neurons)
        b1, b2, b3 = bottleneck
        self.fc1 = nn.Linear(tensor_neurons, b1)
        self.fc2 = nn.Linear(b1, b2)
        self.fc3 = nn.Linear(b2, b3)
        self.scoring = nn.Linear(b3, 1)

        self.to(self.device)
        self.eval()

    def _bias_value(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        p1 = self.attention(emb1)
        p2 = self.attention(emb2)
        s = self.tensor_network(p1, p2)
        s = torch.t(s)
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        s = F.relu(self.fc3(s))
        return self.scoring(s).view(-1)

    def forward(self, data: dict):
        emb1 = self.gin(data["edge_index_1"], data["features_1"])
        emb2 = self.gin(data["edge_index_2"], data["features_2"])

        cost_mat = self.cost_matrix_mod(emb1, emb2)
        map_mat = self.map_matrix(emb1, emb2)

        m = nn.Softmax(dim=1)
        soft_matrix = m(map_mat) * cost_mat
        bias = self._bias_value(emb1, emb2)
        score = torch.sigmoid(soft_matrix.sum() + bias)

        if self.target_mode == "exp":
            pre_ged = -torch.log(score + 1e-8) * data["avg_v"]
        else:
            pre_ged = score * data["hb"]

        return score, pre_ged.item(), map_mat

    def compute(self, g1: nx.Graph, g2: nx.Graph) -> float:
        if self._both_empty(g1, g2):
            return 0.0
        data = nx_to_neural_data(
            g1,
            g2,
            num_labels=self.num_labels,
            target_mode=self.target_mode,
            device=self.device,
            label_map=self.label_map,
        )
        with torch.no_grad():
            _, neural_ged, map_mat = self.forward(data)

        n1, n2 = data["n1"], data["n2"]
        A1 = _build_adj_dense(data["edge_index_1"], n1)
        A2 = _build_adj_dense(data["edge_index_2"], n2)
        f1 = data["features_1"].cpu()
        f2 = data["features_2"].cpu()

        weight = map_mat.cpu()
        if n1 > n2:
            weight = weight.t()
            A1, A2 = A2, A1
            f1, f2 = f2, f1
            n1, n2 = n2, n1

        try:
            solver = KBestMSolver(weight, A1, A2, f1, f2, pre_ged=neural_ged)
            best_ged, _ = solver.solve(k=self.k_best)
            if best_ged < float("inf"):
                return float(best_ged)
        except Exception:
            pass

        return float(neural_ged)
