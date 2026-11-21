"""Shared neural network layers for GED solvers.

Ported from:
    https://github.com/chengqihao/GED-via-Optimal-Transport/blob/main/src/layers.py

Contains attention, tensor network, matching, OT (Sinkhorn), and other
modules used across multiple GED neural models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionModule(torch.nn.Module):
    """Attention-based graph pooling module.

    Computes a graph-level representation by attending over node embeddings.
    """

    def __init__(self, filters_3: int):
        super().__init__()
        self.filters_3 = filters_3
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(filters_3, filters_3))
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Pool node embeddings into a graph-level vector.

        Args:
            embedding: Node embeddings of shape (n, d).

        Returns:
            Graph representation of shape (d, 1).
        """
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=0)
        transformed_global = torch.tanh(global_context)
        sigmoid_scores = torch.sigmoid(torch.mm(embedding, transformed_global.view(-1, 1)))
        representation = torch.mm(torch.t(embedding), sigmoid_scores)
        return representation


class TensorNetworkModule(torch.nn.Module):
    """Tensor network module for computing similarity between two graphs.

    Produces a similarity vector from two graph-level representations.
    """

    def __init__(self, filters_3: int, tensor_neurons: int, input_dim: int = None):
        super().__init__()
        self.filters_3 = filters_3
        self.input_dim = input_dim if input_dim is not None else filters_3
        self.tensor_neurons = tensor_neurons

        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(self.input_dim, self.input_dim, tensor_neurons)
        )
        self.weight_matrix_block = torch.nn.Parameter(
            torch.Tensor(tensor_neurons, 2 * self.input_dim)
        )
        self.bias = torch.nn.Parameter(torch.Tensor(tensor_neurons, 1))

        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        # Upstream initializes the bias with xavier_uniform_ as well
        # (chengqihao and rozemberczki layers.py).
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1: torch.Tensor, embedding_2: torch.Tensor) -> torch.Tensor:
        """Compute similarity vector from two graph representations.

        Args:
            embedding_1: Graph 1 representation of shape (d, 1).
            embedding_2: Graph 2 representation of shape (d, 1).

        Returns:
            Similarity vector of shape (tensor_neurons, 1).
        """
        scoring = torch.mm(torch.t(embedding_1), self.weight_matrix.view(self.input_dim, -1))
        scoring = scoring.view(self.input_dim, self.tensor_neurons)
        scoring = torch.mm(torch.t(scoring), embedding_2)
        combined_representation = torch.cat((embedding_1, embedding_2))
        block_scoring = torch.mm(self.weight_matrix_block, combined_representation)
        scores = torch.nn.functional.relu(scoring + block_scoring + self.bias)
        return scores


class MatchingModule(torch.nn.Module):
    """Cross-graph matching module for gathering inter-graph information."""

    def __init__(self, filters_3: int):
        super().__init__()
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(filters_3, filters_3))
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Compute a transformed global context from node embeddings.

        Args:
            embedding: Node embeddings of shape (n, d).

        Returns:
            Transformed global context of shape (d,).
        """
        global_context = torch.sum(torch.matmul(embedding, self.weight_matrix), dim=0)
        transformed_global = torch.tanh(global_context)
        return transformed_global


class PSinkhorn(torch.nn.Module):
    """Sinkhorn layer for computing optimal transport plans.

    Implements the Sinkhorn-Knopp algorithm for differentiable
    doubly stochastic matrix computation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, corr: torch.Tensor, max_iter: int, eps: float = 1e-16) -> torch.Tensor:
        """Compute transport plan via Sinkhorn iterations.

        Args:
            corr: Cost/similarity matrix of shape (n1, n2).
            max_iter: Number of Sinkhorn iterations.
            eps: Numerical stability epsilon.

        Returns:
            Transport plan matrix of shape (n1, n2).
        """
        n1, n2 = corr.shape
        transpose = False
        if n1 > n2:
            corr = corr.t()
            n1, n2 = n2, n1
            transpose = True

        dummy_mar = torch.ones(1, requires_grad=False, device=corr.device)
        dummy_flag = False

        if n1 < n2:
            dummy_shape = list(corr.shape)
            dummy_shape[0] = 1
            corr = torch.cat((corr, torch.full(dummy_shape, 0.0, device=corr.device)), dim=0)
            dummy_mar = dummy_mar + (n2 - n1 - 1)
            dummy_flag = True

        n11, n22 = corr.shape
        log_prob1 = torch.zeros(n11, device=corr.device)
        log_prob2 = torch.zeros(n22, device=corr.device)
        log_prob1[-1] = log_prob1[-1] + torch.log(dummy_mar)
        log_u = torch.zeros_like(log_prob1)

        for _ in range(max_iter):
            log_v = log_prob2 - torch.logsumexp(corr + log_u.unsqueeze(-1), -2)
            log_u = log_prob1 - torch.logsumexp(corr + log_v.unsqueeze(-2), -1)

        T = torch.exp(log_u.unsqueeze(-1) + corr + log_v.unsqueeze(-2))

        if dummy_flag:
            T = T[:-1, :]

        if transpose:
            T = T.t()

        return T


class OTLayer(nn.Module):
    """Optimal Transport layer using learnable Sinkhorn.

    Produces a soft assignment matrix from a cost matrix via
    differentiable Sinkhorn iterations.
    """

    def __init__(self, max_iter: int = 5):
        super().__init__()
        self.max_iter = max_iter
        self.epsilon = torch.nn.Parameter(torch.zeros(1))
        self.sinkhorn = PSinkhorn()

    def forward(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        """Compute soft assignment from cost matrix.

        Args:
            cost_matrix: Cost matrix of shape (n1, n2).

        Returns:
            Soft assignment matrix of shape (n1, n2).
        """
        K = -cost_matrix / (self.epsilon + 0.05)
        match = self.sinkhorn(K, self.max_iter)
        return match


class TenSetModule(torch.nn.Module):
    """MLP module for scalar prediction from node embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        layers = []
        layers.append(torch.nn.Linear(dim, dim * 2))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(dim * 2, dim))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(dim, 1))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)
