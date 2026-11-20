"""Grasp: Graph Edit Distance via Multi-Level Pattern Matching with GRU.

Reference:
    Lee, J. et al.
    Grasp: Graph Edit Distance via Multi-Level Pattern Matching.
    https://github.com/JunkyuLee1/Grasp

Key innovations over GedGNN:
- 5-layer GIN with degree features appended to initial node features
- Multi-level pattern matching: computes pairwise matrices at 6 GNN levels
- GRU aggregates the 6 pattern matrices across the "layer sequence"
- Additive attention pools GRU outputs into a single pattern matrix
- Cross-attention cost aggregation (Q from pattern, K/V from raw patterns)
- Bidirectional softmax (row + column) on the mapping matrix
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GINConv
from torch_geometric.utils import degree as pyg_degree

import networkx as nx

from ged_lib.solvers import register_solver
from ged_lib.solvers.neural.models import NeuralGEDSolver
from ged_lib.solvers.neural.layers import AttentionModule, TensorNetworkModule
from ged_lib.solvers.neural.data_adapter import nx_to_neural_data


class _BilinearMatrix(nn.Module):
    """Bilinear matrix module that returns k-channel pairwise features.

    Unlike GedMatrixModule which collapses to (n1, n2) via MLP,
    this returns (n1, n2, k) for use in the Grasp GRU sequence.
    """

    def __init__(self, d: int, k: int):
        super().__init__()
        self.d = d
        self.k = k
        self.weight_matrix = nn.Parameter(torch.Tensor(k, d, d))
        nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        matrix = torch.matmul(emb1, self.weight_matrix)
        matrix = torch.matmul(matrix, emb2.t())
        return matrix.permute(1, 2, 0)


@register_solver("grasp")
class GraspSolver(NeuralGEDSolver):
    """Grasp: Multi-level pattern matching with GRU for GED computation.

    Uses 5-layer GIN with degree features, computes GedMatrixModules at each
    of 6 levels, aggregates via GRU + attention, then derives cost and mapping
    matrices with bidirectional softmax scoring.

    Args:
        num_labels: Number of one-hot label dimensions.
        filters: GIN filter size (same for all 5 layers).
        bilinear_dim: GedMatrixModule hidden dim / GRU hidden dim.
        pattern_att_dim: Additive attention dimension for GRU pooling.
        cost_att_dim: Cross-attention dimension for cost aggregation.
        tensor_neurons: Tensor network neurons for bias.
        bottleneck: Bottleneck layer sizes for bias MLP.
        target_mode: "exp" or "linear" normalization.
        device: Torch device string.
    """

    def __init__(
        self,
        num_labels: int = 29,
        filters: int = 16,
        bilinear_dim: int = 32,
        pattern_att_dim: int = 32,
        cost_att_dim: int = 32,
        tensor_neurons: int = 16,
        bottleneck: tuple = (128, 64, 32),
        target_mode: str = "exp",
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.num_labels = max(num_labels, 1)
        self.target_mode = target_mode
        self.device = torch.device(device)
        self.filters = filters
        self.bilinear_dim = bilinear_dim

        self._build_gin(filters)
        self._build_pattern_matrices(filters, bilinear_dim)
        self._build_aggregation(bilinear_dim, pattern_att_dim, cost_att_dim)
        self._build_bias(filters, tensor_neurons, bottleneck)

        self.to(self.device)
        self.eval()

    def _make_gin_layer(self, dim: int) -> GINConv:
        return GINConv(
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
                nn.BatchNorm1d(dim, track_running_stats=False),
            ),
            train_eps=True,
        )

    def _build_gin(self, filters: int) -> None:
        self.layer0_emb = nn.Sequential(
            nn.Linear(self.num_labels + 1, filters),
            nn.ReLU(),
            nn.Linear(filters, filters),
        )
        self.conv1 = self._make_gin_layer(filters)
        self.conv2 = self._make_gin_layer(filters)
        self.conv3 = self._make_gin_layer(filters)
        self.conv4 = self._make_gin_layer(filters)
        self.conv5 = self._make_gin_layer(filters)

        k = self.num_labels + 1 + filters * 6
        self.mlp = nn.Sequential(
            nn.Linear(k, k * 2),
            nn.ReLU(),
            nn.Linear(k * 2, k),
            nn.ReLU(),
            nn.Linear(k, filters),
        )

    def _build_pattern_matrices(self, filters: int, bilinear_dim: int) -> None:
        self.h0_pattern = _BilinearMatrix(filters, bilinear_dim)
        self.h1_pattern = _BilinearMatrix(filters, bilinear_dim)
        self.h2_pattern = _BilinearMatrix(filters, bilinear_dim)
        self.h3_pattern = _BilinearMatrix(filters, bilinear_dim)
        self.h4_pattern = _BilinearMatrix(filters, bilinear_dim)
        self.h5_pattern = _BilinearMatrix(filters, bilinear_dim)

    def _build_aggregation(
        self, bilinear_dim: int, pattern_att_dim: int, cost_att_dim: int
    ) -> None:
        self.gru = nn.GRU(bilinear_dim, bilinear_dim, batch_first=True)

        self.W_attn = nn.Linear(bilinear_dim, pattern_att_dim, bias=True)
        self.v_attn = nn.Parameter(torch.randn(pattern_att_dim))

        self.W_q1 = nn.Linear(bilinear_dim, cost_att_dim, bias=True)
        self.W_q2 = nn.Linear(cost_att_dim, cost_att_dim, bias=True)
        self.W_k = nn.Linear(bilinear_dim, cost_att_dim, bias=True)
        self.W_v = nn.Linear(bilinear_dim, cost_att_dim, bias=True)

        self.cost_mlp = nn.Sequential(
            nn.Linear(cost_att_dim, cost_att_dim * 2),
            nn.ReLU(),
            nn.Linear(cost_att_dim * 2, cost_att_dim),
            nn.ReLU(),
            nn.Linear(cost_att_dim, 1),
        )
        self.map_mlp = nn.Sequential(
            nn.Linear(bilinear_dim, bilinear_dim * 2),
            nn.ReLU(),
            nn.Linear(bilinear_dim * 2, bilinear_dim),
            nn.ReLU(),
            nn.Linear(bilinear_dim, 1),
        )

    def _build_bias(self, filters: int, tensor_neurons: int, bottleneck: tuple) -> None:
        self.attention = AttentionModule(filters)
        self.tensor_network = TensorNetworkModule(filters, tensor_neurons)
        b1, b2, b3 = bottleneck
        self.fc1 = nn.Linear(tensor_neurons, b1)
        self.fc2 = nn.Linear(b1, b2)
        self.fc3 = nn.Linear(b2, b3)
        self.scoring = nn.Linear(b3, 1)

    def _compute_degree(self, edge_index: torch.Tensor, n: int) -> torch.Tensor:
        """Compute node degrees efficiently using torch_geometric.utils.degree."""
        deg = pyg_degree(edge_index[0], num_nodes=n).unsqueeze(1)
        return deg

    def _conv_pass(self, edge_index: torch.Tensor, features: torch.Tensor):
        n = features.size(0)
        degree = self._compute_degree(edge_index, n)
        x0 = self.layer0_emb(torch.cat((features, degree), dim=1))

        feat_list = torch.cat([features, degree, x0], dim=1)

        x1 = F.relu(self.conv1(x0, edge_index))
        feat_list = torch.cat([feat_list, x1], dim=1)
        x1 = F.dropout(x1, p=0.5, training=self.training)

        x2 = F.relu(self.conv2(x1, edge_index))
        feat_list = torch.cat([feat_list, x2], dim=1)
        x2 = F.dropout(x2, p=0.5, training=self.training)

        x3 = F.relu(self.conv3(x2, edge_index))
        feat_list = torch.cat([feat_list, x3], dim=1)
        x3 = F.dropout(x3, p=0.5, training=self.training)

        x4 = F.relu(self.conv4(x3, edge_index))
        feat_list = torch.cat([feat_list, x4], dim=1)
        x4 = F.dropout(x4, p=0.5, training=self.training)

        x5 = self.conv5(x4, edge_index)
        feat_list = torch.cat([feat_list, x5], dim=1)

        final = self.mlp(feat_list)
        return final, x0, x1, x2, x3, x4, x5

    def _get_bias(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        p1 = self.attention(emb1)
        p2 = self.attention(emb2)
        s = self.tensor_network(p1, p2)
        s = torch.t(s)
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        s = F.relu(self.fc3(s))
        return self.scoring(s).view(-1)

    def forward(self, data: dict):
        emb1, g1_h0, g1_h1, g1_h2, g1_h3, g1_h4, g1_h5 = self._conv_pass(
            data["edge_index_1"], data["features_1"]
        )
        emb2, g2_h0, g2_h1, g2_h2, g2_h3, g2_h4, g2_h5 = self._conv_pass(
            data["edge_index_2"], data["features_2"]
        )

        patterns = torch.stack(
            [
                self.h0_pattern(g1_h0, g2_h0),
                self.h1_pattern(g1_h1, g2_h1),
                self.h2_pattern(g1_h2, g2_h2),
                self.h3_pattern(g1_h3, g2_h3),
                self.h4_pattern(g1_h4, g2_h4),
                self.h5_pattern(g1_h5, g2_h5),
            ],
            dim=0,
        )

        seq_len, n, m, feat_dim = patterns.shape
        gru_in = patterns.permute(1, 2, 0, 3).reshape(n * m, seq_len, feat_dim)
        gru_out, _ = self.gru(gru_in)

        attn_scores = torch.matmul(torch.tanh(self.W_attn(gru_out)), self.v_attn)
        a_t = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        context = (a_t * gru_out).sum(dim=1)
        pattern = context.view(n, m, self.bilinear_dim)

        q = self.W_q2(F.relu(self.W_q1(pattern)))
        k = self.W_k(patterns)
        v = self.W_v(patterns)
        s = (q * k).sum(dim=-1) / math.sqrt(self.bilinear_dim)
        a = torch.softmax(s, dim=0)
        cost_context = (a.unsqueeze(-1) * v).sum(dim=0)

        cost_matrix = self.cost_mlp(cost_context).squeeze(-1)
        map_matrix = self.map_mlp(pattern).squeeze(-1)

        row_agg = F.softmax(map_matrix, dim=1) * cost_matrix
        col_agg = F.softmax(map_matrix, dim=0) * cost_matrix

        bias = self._get_bias(emb1, emb2)
        score = torch.sigmoid((row_agg.sum() + col_agg.sum()) / 2.0 + bias)

        if self.target_mode == "exp":
            pre_ged = -torch.log(score + 1e-8) * data["avg_v"]
        else:
            pre_ged = score * data["hb"]

        return score, pre_ged.item(), map_matrix

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
            _, pre_ged, _ = self.forward(data)
        return float(pre_ged)
