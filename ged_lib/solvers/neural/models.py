"""Neural GED solver models: GEDIOT, GedGNN, SimGNN.

Ported and adapted from:
    https://github.com/chengqihao/GED-via-Optimal-Transport/blob/main/src/models.py

All models inherit from GEDSolver and can be used with compute(g1, g2) -> float.
They require pre-trained weights for inference (use load_weights()).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv, GINConv

import networkx as nx

from ged_lib.base import GEDSolver
from ged_lib.solvers import register_solver
from ged_lib.solvers.neural.layers import (
    AttentionModule,
    TensorNetworkModule,
    MatchingModule,
    OTLayer,
)
from ged_lib.solvers.neural.ged_matrix import CostMatrixModule, GedMatrixModule
from ged_lib.solvers.neural.data_adapter import nx_to_neural_data, get_label_map


class NeuralGEDSolver(nn.Module, GEDSolver):
    """Base class for neural GED solvers that are also nn.Module.

    Uses multiple inheritance so that:
    - nn.Module provides .to(), .eval(), .parameters(), etc.
    - GEDSolver provides the compute() interface.

    Neural solvers do not use CostModel, so GEDSolver.__init__ is skipped
    to avoid creating an unnecessary CostModel instance.
    """

    def __init__(self):
        nn.Module.__init__(self)
        # Skip GEDSolver.__init__() — neural solvers don't use CostModel
        self.cost_model = None
        self.label_map = None

    def _load_label_map(self, dataset_name: str = "") -> None:
        """Load and cache label mapping for the given dataset.

        Should be called from subclass __init__ when num_labels is set.
        """
        self.label_map = get_label_map(dataset_name)

    @staticmethod
    def _both_empty(g1: nx.Graph, g2: nx.Graph) -> bool:
        """True when both graphs have no nodes — GED is trivially 0.

        Guards the neural forward pass, which cannot run on empty node tensors
        (e.g. BatchNorm1d on 0 samples) and would otherwise hit a division by
        zero on the ``avg_v`` / ``hb`` normalisation terms.
        """
        return g1.number_of_nodes() == 0 and g2.number_of_nodes() == 0

    def compute(self, g1: nx.Graph, g2: nx.Graph) -> float:
        raise NotImplementedError

    def load_weights(self, path: str) -> None:
        state = torch.load(path, map_location=next(self.parameters()).device, weights_only=True)
        self.load_state_dict(state)
        self.eval()


class _GINBackbone(nn.Module):
    """Shared GIN (Graph Isomorphism Network) backbone for node embedding.

    3-layer GIN with batch normalization, producing d-dimensional node embeddings.
    Includes an optional MLP that combines multi-scale features.
    """

    def __init__(self, num_labels: int, filters: tuple = (64, 32, 16)):
        super().__init__()
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


@register_solver("gediot")
class GEDIOTSolver(NeuralGEDSolver):
    """GED via Inverse Optimal Transport (GEDIOT).

    Main contribution of the SIGMOD'25 paper.
    Uses GIN embeddings + learned cost matrix + differentiable Sinkhorn OT.
    Requires pre-trained weights for inference.

    Default hyperparameters:
        filters = (64, 32, 16), hidden_dim = 16, tensor_neurons = 16
        bottleneck = (128, 64, 32), target_mode = "linear"
    """

    def __init__(
        self,
        num_labels: int = 29,
        filters: tuple = (64, 32, 16),
        tensor_neurons: int = 16,
        bottleneck: tuple = (128, 64, 32),
        target_mode: str = "linear",
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.num_labels = max(num_labels, 1)
        self.target_mode = target_mode
        self.device = torch.device(device)
        f1, f2, f3 = filters
        self.filters_3 = f3

        self.gin = _GINBackbone(self.num_labels, filters)
        self.cost_matrix = CostMatrixModule(f3)
        self.ot_layer = OTLayer(max_iter=5)

        self.attention = AttentionModule(f3)
        self.tensor_network = TensorNetworkModule(f3, tensor_neurons)
        b1, b2, b3 = bottleneck
        self.fc1 = nn.Linear(tensor_neurons, b1)
        self.fc2 = nn.Linear(b1, b2)
        self.fc3 = nn.Linear(b2, b3)
        self.scoring = nn.Linear(b3, 1)

        self.to(self.device)
        self.eval()

    def _unmatching_part(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        pooled1 = self.attention(emb1)
        pooled2 = self.attention(emb2)
        scores = self.tensor_network(pooled1, pooled2)
        scores = torch.t(scores)
        scores = F.relu(self.fc1(scores))
        scores = F.relu(self.fc2(scores))
        scores = F.relu(self.fc3(scores))
        return self.scoring(scores).view(-1)

    def forward(self, data: dict):
        """Full forward pass returning score, predicted GED, and mapping matrix."""
        emb1 = self.gin(data["edge_index_1"], data["features_1"])
        emb2 = self.gin(data["edge_index_2"], data["features_2"])

        cost_mat = torch.tanh(self.cost_matrix(emb1, emb2))
        map_mat = self.ot_layer(cost_mat)
        soft_matrix = map_mat * cost_mat
        bias = self._unmatching_part(emb1, emb2)
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
            _, pre_ged, _ = self.forward(data)
        return float(pre_ged)


@register_solver("gedgnn")
class GedGNNSolver(NeuralGEDSolver):
    """GED Graph Neural Network (GedGNN) solver.

    Uses GIN embeddings + learned mapping/cost matrices.
    Requires pre-trained weights.
    """

    def __init__(
        self,
        num_labels: int = 29,
        filters: tuple = (64, 32, 16),
        hidden_dim: int = 16,
        tensor_neurons: int = 16,
        bottleneck: tuple = (128, 64, 32),
        target_mode: str = "linear",
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.num_labels = max(num_labels, 1)
        self.target_mode = target_mode
        self.device = torch.device(device)
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
            _, pre_ged, _ = self.forward(data)
        return float(pre_ged)


@register_solver("simgnn")
class SimGNNSolver(NeuralGEDSolver):
    """SimGNN: A Neural Network Approach to Fast Graph Similarity Computation.

    Reference: Bai et al. (2019), https://arxiv.org/abs/1808.05689
    Uses GCN + attention + tensor network for graph similarity scoring.
    Requires pre-trained weights.
    """

    def __init__(
        self,
        num_labels: int = 29,
        filters: tuple = (64, 32, 16),
        tensor_neurons: int = 16,
        bottleneck: tuple = (128, 64, 32),
        histogram: bool = True,
        bins: int = 16,
        target_mode: str = "exp",
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.num_labels = max(num_labels, 1)
        self.target_mode = target_mode
        self.device = torch.device(device)
        self.histogram = histogram
        self.bins = bins
        f1, f2, f3 = filters

        self.conv1 = GCNConv(self.num_labels, f1)
        self.conv2 = GCNConv(f1, f2)
        self.conv3 = GCNConv(f2, f3)

        feature_count = tensor_neurons + (bins if histogram else 0)

        self.attention = AttentionModule(f3)
        self.tensor_network = TensorNetworkModule(f3, tensor_neurons)
        b1, b2, b3 = bottleneck
        self.fc1 = nn.Linear(feature_count, b1)
        self.fc2 = nn.Linear(b1, b2)
        self.fc3 = nn.Linear(b2, b3)
        self.scoring = nn.Linear(b3, 1)

        self.to(self.device)
        self.eval()

    def _conv_pass(self, edge_index: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(features, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return x

    def forward(self, data: dict):
        emb1 = self._conv_pass(data["edge_index_1"], data["features_1"])
        emb2 = self._conv_pass(data["edge_index_2"], data["features_2"])

        p1 = self.attention(emb1)
        p2 = self.attention(emb2)
        scores = self.tensor_network(p1, p2)
        scores = torch.t(scores)

        if self.histogram:
            sim = torch.mm(emb1, emb2.t()).detach()
            hist = torch.histc(sim, bins=self.bins)
            hist = hist / (hist.sum() + 1e-8)
            scores = torch.cat((scores, hist.view(1, -1)), dim=1)

        scores = F.relu(self.fc1(scores))
        scores = F.relu(self.fc2(scores))
        scores = F.relu(self.fc3(scores))
        score = torch.sigmoid(self.scoring(scores).view(-1))

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


@register_solver("tagsim")
class TaGSimSolver(NeuralGEDSolver):
    """TaGSim: Type-aware Graph Similarity Learning and Computation.

    Reference: Bai et al., https://github.com/jiyangbai/TaGSim
    Decomposes GED into node relabeling (nc), node insertions (in),
    and edge insertions (ie), predicting each separately.
    """

    def __init__(
        self,
        num_labels: int = 29,
        tensor_neurons: int = 16,
        bottleneck: tuple = (128, 8, 4),
        target_mode: str = "exp",
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.num_labels = max(num_labels, 1)
        self.target_mode = target_mode
        self.device = torch.device(device)
        self.feature_count = tensor_neurons

        self.tensor_nc = TensorNetworkModule(
            self.feature_count, tensor_neurons, 2 * self.num_labels
        )
        self.tensor_in = TensorNetworkModule(
            self.feature_count, tensor_neurons, 2 * self.num_labels
        )
        self.tensor_ie = TensorNetworkModule(
            self.feature_count, tensor_neurons, 2 * self.num_labels
        )
        b1, b2, b3 = bottleneck

        self.fc_nc = nn.Sequential(
            nn.Linear(self.feature_count, b1),
            nn.ReLU(),
            nn.Linear(b1, b2),
            nn.ReLU(),
            nn.Linear(b2, b3),
            nn.ReLU(),
            nn.Linear(b3, 1),
        )
        self.fc_in = nn.Sequential(
            nn.Linear(self.feature_count, b1),
            nn.ReLU(),
            nn.Linear(b1, b2),
            nn.ReLU(),
            nn.Linear(b2, b3),
            nn.ReLU(),
            nn.Linear(b3, 1),
        )
        self.fc_ie = nn.Sequential(
            nn.Linear(self.feature_count, b1),
            nn.ReLU(),
            nn.Linear(b1, b2),
            nn.ReLU(),
            nn.Linear(b2, b3),
            nn.ReLU(),
            nn.Linear(b3, 1),
        )
        self.to(self.device)
        self.eval()

    def _gal_pass(self, adj: torch.Tensor, features: torch.Tensor) -> tuple:
        h1 = torch.mm(adj, features)
        h2 = torch.mm(adj, h1)
        return h1, h2

    def forward(self, data: dict):
        n1, n2 = data["n1"], data["n2"]
        f1, f2 = data["features_1"], data["features_2"]
        ei1, ei2 = data["edge_index_1"], data["edge_index_2"]

        adj1 = torch.sparse_coo_tensor(ei1, torch.ones(ei1.shape[1]), (n1, n1)).to_dense()
        adj2 = torch.sparse_coo_tensor(ei2, torch.ones(ei2.shape[1]), (n2, n2)).to_dense()
        adj1 = adj1 * (1.0 - torch.eye(n1, device=adj1.device))
        adj2 = adj2 * (1.0 - torch.eye(n2, device=adj2.device))

        h1_1, h1_2 = self._gal_pass(adj1, f1)
        h2_1, h2_2 = self._gal_pass(adj2, f2)

        cat1_01 = torch.cat([f1, h1_1], dim=1)
        cat2_01 = torch.cat([f2, h2_1], dim=1)
        cat1_12 = torch.cat([h1_1, h1_2], dim=1)
        cat2_12 = torch.cat([h2_1, h2_2], dim=1)

        p1_01 = torch.sum(cat1_01, dim=0).unsqueeze(1)
        p2_01 = torch.sum(cat2_01, dim=0).unsqueeze(1)
        p1_12 = torch.sum(cat1_12, dim=0).unsqueeze(1)
        p2_12 = torch.sum(cat2_12, dim=0).unsqueeze(1)

        s_nc = torch.sigmoid(self.fc_nc(torch.t(self.tensor_nc(p1_01, p2_01))))
        s_in = torch.sigmoid(self.fc_in(torch.t(self.tensor_in(p1_01, p2_01))))
        s_ie = torch.sigmoid(self.fc_ie(torch.t(self.tensor_ie(p1_12, p2_12))))

        score = torch.cat([s_nc.view(-1), s_in.view(-1), s_ie.view(-1)])
        if self.target_mode == "exp":
            pre_ged = -torch.log(score + 1e-8) * data["avg_v"]
        else:
            pre_ged = score * data["hb"]
        return score, pre_ged.sum().item()

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


@register_solver("gpn")
class GPNSolver(NeuralGEDSolver):
    """Graph Path Network (GPN) solver.

    The GNN of the Noah framework: GIN embeddings + matching modules for
    cross-graph comparison. Included here via the GED-via-Optimal-Transport
    codebase (Cheng et al., SIGMOD 2025).

    Reference:
        Yang, L., & Zou, L. (2021). Noah: Neural-optimized A* Search Algorithm
        for Graph Edit Distance Computation. ICDE 2021, pp. 576-587.
        https://github.com/pkumod/Noah-GED
    """

    def __init__(
        self,
        num_labels: int = 29,
        filters: tuple = (64, 32, 16),
        tensor_neurons: int = 16,
        bottleneck: tuple = (128,),
        target_mode: str = "exp",
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.num_labels = max(num_labels, 1)
        self.target_mode = target_mode
        self.device = torch.device(device)
        f1, f2, f3 = filters

        nn1 = nn.Sequential(
            nn.Linear(self.num_labels, f1),
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

        self.matching1 = MatchingModule(f3)
        self.matching2 = MatchingModule(f3)
        self.attention = AttentionModule(f3)
        self.tensor_network = TensorNetworkModule(f3, tensor_neurons)
        self.fc1 = nn.Linear(tensor_neurons, bottleneck[0])
        self.scoring = nn.Linear(bottleneck[0], 1)

        self.to(self.device)
        self.eval()

    def _conv_pass(self, edge_index: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(features, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return x

    def forward(self, data: dict):
        emb1 = self._conv_pass(data["edge_index_1"], data["features_1"])
        emb2 = self._conv_pass(data["edge_index_2"], data["features_2"])

        diff1 = torch.abs(emb1 - self.matching2(emb2))
        diff2 = torch.abs(emb2 - self.matching1(emb1))

        p1 = self.attention(diff1)
        p2 = self.attention(diff2)
        scores = torch.t(self.tensor_network(p1, p2))
        scores = F.relu(self.fc1(scores))
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
