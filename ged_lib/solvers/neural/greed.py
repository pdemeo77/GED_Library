"""GREED: A Neural Framework for Learning Graph Distance Functions.

Reference:
    Ranjan, R. et al. (2022).
    GREED: A Neural Framework for Learning Graph Distance Functions.
    NeurIPS 2022.
    https://github.com/idea-iitd/greed

Implements Siamese GNN architectures with multiple comparison modes:
- "neural": MLP-based comparison after concatenation
- "norm_ged": L2 distance between embeddings (symmetric, for GED)
- "norm_sed": One-sided L2 distance (asymmetric, for SED)

The model uses multi-scale GIN/GCN with residual connections and
flexible pooling (add, mean, max, attention).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, SAGEConv, GATConv
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,
    GlobalAttention,
)

import networkx as nx

from ged_lib.solvers import register_solver
from ged_lib.solvers.neural.models import NeuralGEDSolver
from ged_lib.solvers.neural.data_adapter import nx_to_neural_data


_CONV_REGISTRY = {
    "gin": lambda d: GINConv(nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))),
    "gcn": lambda d: GCNConv(d, d),
    "sage": lambda d: SAGEConv(d, d),
    "gat": lambda d: GATConv(d, d),
}


class _EmbedModel(nn.Module):
    """Multi-scale GNN encoder with residual connections.

    Concatenates features from all layers, then pools and projects
    to a fixed-size graph embedding.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 3,
        conv: str = "gin",
        pool: str = "add",
        max_degree: int = 0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.max_degree = max_degree

        feat_dim = input_dim + (max_degree + 1) if max_degree > 0 else input_dim

        self.pre = nn.Linear(feat_dim, hidden_dim)

        make_conv = _CONV_REGISTRY.get(conv)
        if make_conv is None:
            raise ValueError(f"Unknown conv type: {conv}")
        self.convs = nn.ModuleList([make_conv(hidden_dim) for _ in range(n_layers)])

        cat_dim = hidden_dim * (n_layers + 1)
        if pool == "att":
            self.pool_fn = GlobalAttention(
                nn.Sequential(
                    nn.Linear(cat_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
            )
        else:
            self.pool_fn = None
        self.pool_str = pool

        self.post = nn.Sequential(
            nn.Linear(cat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _append_degree_features(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.max_degree <= 0:
            return x
        from torch_geometric.utils import degree

        deg = degree(edge_index[0], x.size(0)).long()
        deg = deg.clamp(max=self.max_degree)
        deg_onehot = F.one_hot(deg, num_classes=self.max_degree + 1).float()
        return torch.cat((x, deg_onehot), dim=-1)

    def _pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pool_str == "add":
            return global_add_pool(x, batch)
        elif self.pool_str == "mean":
            return global_mean_pool(x, batch)
        elif self.pool_str == "max":
            return global_max_pool(x, batch)
        elif self.pool_str == "att":
            return self.pool_fn(x, batch)
        return global_add_pool(x, batch)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        x = self._append_degree_features(x, edge_index)
        x = self.pre(x)
        emb = x
        xres = x
        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index)
            if i & 1:
                x = x + xres
                xres = x
            x = F.relu(x)
            emb = torch.cat((emb, x), dim=1)
        x = self._pool(emb, batch)
        x = self.post(x)
        return x


@register_solver("greed")
class GREEDSolver(NeuralGEDSolver):
    """GREED: A Neural Framework for Learning Graph Distance Functions.

    NeurIPS 2022. Siamese GNN with multi-scale encoder and configurable
    comparison mode (MLP, L2-GED, or L2-SED).

    Args:
        num_labels: Number of one-hot label dimensions.
        hidden_dim: Hidden dimension of the GNN encoder.
        output_dim: Output graph embedding dimension.
        n_layers: Number of GNN layers.
        conv: GNN type — "gin", "gcn", "sage", or "gat".
        pool: Pooling method — "add", "mean", "max", or "att".
        compare: Comparison mode — "neural", "norm_ged", or "norm_sed".
        max_degree: If >0, appends one-hot degree features.
        target_mode: "exp" or "linear" normalization.
        device: Torch device string.
    """

    def __init__(
        self,
        num_labels: int = 29,
        hidden_dim: int = 64,
        output_dim: int = 32,
        n_layers: int = 3,
        conv: str = "gin",
        pool: str = "add",
        compare: str = "neural",
        max_degree: int = 0,
        target_mode: str = "exp",
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.num_labels = max(num_labels, 1)
        self.target_mode = target_mode
        self.device = torch.device(device)
        self.compare = compare
        self.output_dim = output_dim

        self.embed_model = _EmbedModel(
            input_dim=self.num_labels,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            conv=conv,
            pool=pool,
            max_degree=max_degree,
        )

        if compare == "neural":
            self.mlp_model = nn.Sequential(
                nn.Linear(2 * output_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, 1),
            )
        else:
            self.mlp_model = None

        self.to(self.device)
        self.eval()

    def _make_batch(
        self, edge_index: torch.Tensor, features: torch.Tensor, n_nodes: int
    ) -> torch.Tensor:
        return torch.zeros(n_nodes, dtype=torch.long, device=self.device)

    def forward(self, data: dict):
        batch1 = self._make_batch(data["edge_index_1"], data["features_1"], data["n1"])
        batch2 = self._make_batch(data["edge_index_2"], data["features_2"], data["n2"])

        gx = self.embed_model(data["features_1"], data["edge_index_1"], batch1)
        hx = self.embed_model(data["features_2"], data["edge_index_2"], batch2)

        if self.compare == "neural":
            score = self.mlp_model(torch.cat((gx, hx), dim=-1)).view(-1)
            score = torch.sigmoid(score)
        elif self.compare == "norm_ged":
            dist = torch.norm(gx - hx, dim=-1)
            gs = torch.tensor(data["n1"] + data["n2"], dtype=torch.float, device=self.device) / 2.0
            score = torch.exp(-dist / (gs + 1e-8))
        elif self.compare == "norm_sed":
            dist = torch.norm(F.relu(gx - hx), dim=-1)
            gs = torch.tensor(float(data["n1"]), dtype=torch.float, device=self.device)
            score = torch.exp(-dist / (gs + 1e-8))
        else:
            raise ValueError(f"Unknown compare mode: {self.compare}")

        if self.target_mode == "exp":
            avg_v = data["avg_v"]
            pre_ged = -torch.log(score + 1e-8) * avg_v
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
