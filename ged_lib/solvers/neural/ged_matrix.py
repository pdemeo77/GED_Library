"""GED matrix modules for neural GED solvers.

Ported from:
    https://github.com/chengqihao/GED-via-Optimal-Transport/blob/main/src/GedMatrix.py

Contains modules that learn cost matrices and mapping matrices
from node embeddings, plus loss functions for training.
"""

import torch
import torch.nn.functional as F


class CostMatrixModule(torch.nn.Module):
    """Learns a cost/similarity matrix from two sets of node embeddings.

    Uses a learnable weight matrix to transform embeddings into
    a pairwise cost matrix of shape (n1, n2).
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(d, d))
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding_1: torch.Tensor, embedding_2: torch.Tensor) -> torch.Tensor:
        """Compute cost matrix from two sets of embeddings.

        Args:
            embedding_1: Node embeddings of graph 1, shape (n1, d).
            embedding_2: Node embeddings of graph 2, shape (n2, d).

        Returns:
            Cost matrix of shape (n1, n2).
        """
        matrix = torch.matmul(embedding_1, self.weight_matrix)
        matrix = torch.matmul(matrix, embedding_2.t())
        return matrix


class GedMatrixModule(torch.nn.Module):
    """GED mapping matrix module with multi-head projection + MLP.

    Uses k weight matrices to produce k candidate matrices,
    then an MLP to collapse them into a single (n1, n2) mapping.
    """

    def __init__(self, d: int, k: int):
        super().__init__()
        self.d = d
        self.k = k
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(k, d, d))
        torch.nn.init.xavier_uniform_(self.weight_matrix)

        layers = []
        layers.append(torch.nn.Linear(k, k * 2))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(k * 2, k))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(k, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, embedding_1: torch.Tensor, embedding_2: torch.Tensor) -> torch.Tensor:
        """Compute mapping matrix from two sets of embeddings.

        Args:
            embedding_1: Node embeddings of graph 1, shape (n1, d).
            embedding_2: Node embeddings of graph 2, shape (n2, d).

        Returns:
            Mapping matrix of shape (n1, n2).
        """
        n1 = embedding_1.shape[0]
        matrix = torch.matmul(embedding_1, self.weight_matrix)
        matrix = torch.matmul(matrix, embedding_2.t())
        matrix = matrix.reshape(self.k, -1).t()
        matrix = self.mlp(matrix)
        return matrix.reshape(n1, -1)


def fixed_mapping_loss(mapping: torch.Tensor, gt_mapping: torch.Tensor) -> torch.Tensor:
    """BCE loss with adaptive subsampling for sparse ground-truth mappings.

    Args:
        mapping: Predicted mapping logits, shape (n1, n2).
        gt_mapping: Ground truth binary mapping, shape (n1, n2).

    Returns:
        Scalar loss value.
    """
    mapping_loss = torch.nn.BCEWithLogitsLoss()
    n1, n2 = mapping.shape

    num_1 = gt_mapping.sum().item()
    num_0 = n1 * n2 - num_1
    if num_1 >= num_0:
        return mapping_loss(mapping, gt_mapping)

    p_base = num_1 / num_0
    p = 1.0 - (p_base + 0.5 * (1 - p_base))

    mask = (torch.rand([n1, n2], device=mapping.device) + gt_mapping) > p
    return mapping_loss(mapping[mask], gt_mapping[mask])


def mapping_loss_mse(mapping: torch.Tensor, gt_mapping: torch.Tensor) -> torch.Tensor:
    """MSE loss between softmax of predicted and ground-truth mapping.

    Args:
        mapping: Predicted mapping logits, shape (n1, n2).
        gt_mapping: Ground truth mapping, shape (n1, n2).

    Returns:
        Scalar MSE loss.
    """
    m = torch.nn.Softmax(dim=1)
    return torch.nn.functional.mse_loss(m(mapping), m(gt_mapping))
