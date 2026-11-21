"""EGSC-T: Efficient Graph Similarity Computation (Teacher Model).

Reference:
    Qin, C. et al. (2021).
    Slow Learning and Fast Inference: Efficient Graph Similarity Computation
    via Knowledge Distillation. NeurIPS 2021.
    https://github.com/canqin001/Efficient_Graph_Similarity_Computation

The model uses a multi-level GNN backbone with SE-attention and
co-attention across 3 feature levels for graph comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

import networkx as nx

from ged_lib.solvers import register_solver
from ged_lib.solvers.neural.models import NeuralGEDSolver
from ged_lib.solvers.neural.data_adapter import nx_to_neural_data


class _SEModule(nn.Module):
    """Squeeze-and-Excitation attention module."""

    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _MultiLevelAttention(nn.Module):
    """Attention pooling with SE-enhanced features."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight_matrix = nn.Parameter(torch.Tensor(dim, dim))
        nn.init.xavier_uniform_(self.weight_matrix)
        self.fc_enhance = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enhanced = self.fc_enhance(x) * x + x
        mean = enhanced.mean(dim=0, keepdim=True)
        transformed = torch.tanh(torch.mm(mean, self.weight_matrix))
        coefs = torch.sigmoid((enhanced * transformed).sum(dim=1, keepdim=True))
        weighted = coefs * enhanced
        return weighted.sum(dim=0, keepdim=True).t()


class _SETensorNetwork(nn.Module):
    """SE-enhanced tensor network for graph-level comparison."""

    def __init__(self, dim: int):
        super().__init__()
        channel = dim * 2
        self.fc_se = nn.Sequential(
            nn.Linear(channel, channel // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channel // 4, channel),
            nn.Sigmoid(),
        )
        # Upstream SETensorNetworkModule.forward uses ONLY its fc1
        # (Linear(c,c) -> ReLU -> Linear(c, dim//2) -> ReLU). Its fc0 block
        # is defined but never called (commented out in EGSC-KD as well) —
        # do not activate that dead block here.
        self.fc_out = nn.Sequential(
            nn.Linear(channel, channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, dim // 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((emb1.t(), emb2.t()), dim=1)
        se_coefs = self.fc_se(combined)
        se_feat = se_coefs * combined + combined
        return self.fc_out(se_feat)


@register_solver("egsc")
class EGSCSolver(NeuralGEDSolver):
    """EGSC-T: Efficient Graph Similarity Computation (Teacher Model).

    NeurIPS 2021. Uses multi-level GIN backbone with SE-attention
    and co-attention across 3 feature levels.

    Args:
        num_labels: Number of one-hot label dimensions.
        filters: Tuple of GIN filter sizes for 3 levels.
        bottleneck: Bottleneck layer size.
        target_mode: "exp" or "linear" normalization.
        device: Torch device string.
    """

    def __init__(
        self,
        num_labels: int = 29,
        filters: tuple = (64, 32, 16),
        bottleneck: int = 128,
        target_mode: str = "exp",
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.num_labels = max(num_labels, 1)
        self.target_mode = target_mode
        self.device = torch.device(device)
        f1, f2, f3 = filters
        self.filters_3 = f3

        self._build_gin_backbone(num_labels, filters)
        self._build_attention_modules(filters)
        self._build_tensor_networks(filters)
        self._build_scoring_head(filters, bottleneck)

        self.to(self.device)
        self.eval()

    def _build_gin_backbone(self, num_labels: int, filters: tuple) -> None:
        f1, f2, f3 = filters
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

    def _build_attention_modules(self, filters: tuple) -> None:
        f1, f2, f3 = filters
        self.att_level1 = _MultiLevelAttention(f1)
        self.att_level2 = _MultiLevelAttention(f2)
        self.att_level3 = _MultiLevelAttention(f3)

    def _build_tensor_networks(self, filters: tuple) -> None:
        f1, f2, f3 = filters
        self.tn_level1 = _SETensorNetwork(f1)
        self.tn_level2 = _SETensorNetwork(f2)
        self.tn_level3 = _SETensorNetwork(f3)

    def _build_scoring_head(self, filters: tuple, bottleneck: int) -> None:
        f1, f2, f3 = filters
        feature_count = (f1 + f2 + f3) // 2
        self.score_attention = _SEModule(feature_count)
        self.fc1 = nn.Linear(feature_count, bottleneck)
        self.scoring = nn.Linear(bottleneck, 1)

    def _conv_pass(self, edge_index: torch.Tensor, features: torch.Tensor) -> tuple:
        """3-level GIN forward pass, returning all intermediate features."""
        h1 = F.relu(self.conv1(features, edge_index))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = F.relu(self.conv2(h1, edge_index))
        h2 = F.dropout(h2, p=0.5, training=self.training)
        h3 = F.relu(self.conv3(h2, edge_index))
        h3 = F.dropout(h3, p=0.5, training=self.training)
        return h1, h2, h3

    def forward(self, data: dict):
        """Full forward pass returning score and predicted GED."""
        h1_1, h2_1, h3_1 = self._conv_pass(data["edge_index_1"], data["features_1"])
        h1_2, h2_2, h3_2 = self._conv_pass(data["edge_index_2"], data["features_2"])

        p1_1 = self.att_level1(h1_1)
        p1_2 = self.att_level1(h1_2)
        s1 = self.tn_level1(p1_1, p1_2)

        p2_1 = self.att_level2(h2_1)
        p2_2 = self.att_level2(h2_2)
        s2 = self.tn_level2(p2_1, p2_2)

        p3_1 = self.att_level3(h3_1)
        p3_2 = self.att_level3(h3_2)
        s3 = self.tn_level3(p3_1, p3_2)

        scores = torch.cat((s3, s2, s1), dim=1)
        scores = F.relu(self.fc1(self.score_attention(scores) * scores + scores))
        score = torch.sigmoid(self.scoring(scores)).view(-1)

        if self.target_mode == "exp":
            pre_ged = -torch.log(score + 1e-8) * data["avg_v"]
        else:
            pre_ged = score * data["hb"]

        return score, pre_ged.item()

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
            _, pre_ged = self.forward(data)
        return float(pre_ged)
