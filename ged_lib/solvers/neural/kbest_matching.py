"""GEDGNN post-processing: k-best matching with GED lower bound pruning.

Reference:
    Piao, C. et al. (2023).
    Computing Graph Edit Distance via Neural Graph Matching. PVLDB 2023.
    https://github.com/ChengzhiPiao/GEDGNN

Provides the KBestMSolver that takes a predicted mapping matrix from GedGNN
and finds the k-best bipartite matchings, computing actual GED for each
using adjacency and feature matrix comparison, pruning with lower bounds.

This enables the "GEDGNN-matching" mode from the paper, which produces
actual edit paths and generally more accurate GED predictions than the
neural score alone ("GEDGNN-value").
"""

import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms.shortest_paths import dense, weighted

import torch
import numpy as np


def ged_from_mapping(
    matrix: torch.Tensor, A1: torch.Tensor, A2: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor
) -> torch.Tensor:
    """Compute GED from a soft or hard mapping matrix.

    Args:
        matrix: Mapping matrix of shape (n1, n2).
        A1: Adjacency matrix of graph 1, shape (n1, n1).
        A2: Adjacency matrix of graph 2, shape (n2, n2).
        f1: Node features of graph 1, shape (n1, d).
        f2: Node features of graph 2, shape (n2, d).

    Returns:
        Scalar tensor with the GED value.
    """
    A_loss = torch.mm(torch.mm(matrix.t(), A1), matrix) - A2
    F_loss = torch.mm(matrix.t(), f1) - f2
    mapping_ged = ((A_loss * A_loss).sum() + (F_loss * F_loss).sum()) / 2.0
    return mapping_ged.view(-1)


def _build_adj_matrix(edge_index: torch.Tensor, n: int) -> torch.Tensor:
    """Build dense adjacency matrix from edge_index (with self-loops)."""
    A = torch.zeros(n, n, device=edge_index.device)
    src, dst = edge_index[0], edge_index[1]
    A[src, dst] = 1.0
    return A


def _build_onehot_mapping(n1: int, n2: int, mapping: list) -> torch.Tensor:
    """Build a (n1, n2) one-hot mapping matrix from a list assignment."""
    M = torch.zeros(n1, n2)
    for i, j in enumerate(mapping):
        if j is not None and 0 <= j < n2:
            M[i, j] = 1.0
    return M


class _GedLowerBound:
    """Compute GED lower bounds for partial matchings."""

    def __init__(self, A1: torch.Tensor, A2: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor):
        self.A1 = A1
        self.A2 = A2
        self.f1 = f1
        self.f2 = f2
        self.n1 = A1.size(0)
        self.n2 = A2.size(0)

    def _matching_cost(self, left_nodes: list, right_nodes: list) -> float:
        if len(left_nodes) == 0:
            return 0.0
        idx_left = torch.tensor(left_nodes)
        idx_right = torch.tensor(right_nodes)
        sub_A1 = self.A1[idx_left][:, idx_left]
        sub_A2 = self.A2[idx_right][:, idx_right]
        A_ged = ((sub_A1 - sub_A2) ** 2).sum().item()
        sub_f1 = self.f1[left_nodes]
        sub_f2 = self.f2[right_nodes]
        F_ged = ((sub_f1 - sub_f2) ** 2).sum().item()
        return (A_ged + F_ged) / 2.0

    def compute(self, left_nodes: list, right_nodes: list) -> float:
        partial_n = len(left_nodes)

        if partial_n == 0 and right_nodes is not None and len(right_nodes) == self.n1:
            left_nodes = list(range(self.n1))
            partial_n = self.n1

        if right_nodes is None:
            return float("inf")

        lb = self._matching_cost(left_nodes, right_nodes)

        diag1 = torch.diag(self.A1)
        diag2 = torch.diag(self.A2)
        m1 = (
            diag1.sum().item() - diag1[left_nodes].sum().item()
            if len(left_nodes) > 0
            else diag1.sum().item()
        )
        m2 = (
            diag2.sum().item() - diag2[right_nodes].sum().item()
            if len(right_nodes) > 0
            else diag2.sum().item()
        )
        m1_real = (self.A1.sum().item() - m1) / 2.0
        m2_real = (self.A2.sum().item() - m2) / 2.0
        lb += abs(m1_real - m2_real) / 2.0

        if partial_n == self.n1:
            lb += self.n2 - self.n1
        else:
            mask1 = torch.ones(self.n1, dtype=torch.bool)
            mask1[left_nodes] = False
            mask2 = torch.ones(self.n2, dtype=torch.bool)
            mask2[right_nodes] = False
            remaining_f1 = self.f1[mask1].sum(dim=0)
            remaining_f2 = self.f2[mask2].sum(dim=0)
            intersect = torch.min(remaining_f1, remaining_f2)
            lb += max(remaining_f1.sum().item(), remaining_f2.sum().item()) - intersect.sum().item()

        return lb


class _Subspace:
    """A solution subspace in the k-best matching search."""

    def __init__(self, G, matching, res, I=None, O=None):
        self.G = G
        self.best_matching = matching
        self.best_res = res
        self.I = set() if I is None else I
        self.O = [] if O is None else O
        self.second_matching = None
        self.second_res = None
        self.branch_edge = None
        self.lb = None
        self.ged = None
        self.ged2 = None
        self._get_second_matching()

    def _get_second_matching(self):
        G = self.G.copy()
        matching = self.best_matching.copy()
        n1 = len(matching)
        n = G.number_of_nodes()
        n2 = n - n1

        for u, v in self.O:
            G[u][v]["weight"] = float("inf")

        matched = [False] * n2
        for u in range(n1):
            v = matching[u]
            matched[v] = True
            v += n1
            w = -G[u][v]["weight"]
            if u in self.I:
                w = float("inf")
            G.remove_edge(u, v)
            G.add_edge(v, u, weight=w)

        n = G.number_of_nodes()
        G.add_node(n, bipartite=0)
        for v in range(n2):
            if matched[v]:
                G.add_edge(n, n1 + v, weight=0.0)
            else:
                G.add_edge(n1 + v, n, weight=0.0)

        dis = dense.floyd_warshall(G)
        cycle_min_weight = float("inf")
        cycle_min_uv = None
        for u in range(n1):
            if u in self.I:
                continue
            v = matching[u] + n1
            res = dis[u][v] + G[v][u]["weight"]
            if res < cycle_min_weight:
                cycle_min_weight = res
                cycle_min_uv = (u, v)

        if cycle_min_uv is None:
            return

        u, v = cycle_min_uv
        try:
            length, path = weighted.single_source_bellman_ford(G, source=u, target=v)
        except Exception:
            return

        self.branch_edge = (u, v)
        for i in range(0, len(path) - 1, 2):
            u_p, v_p = path[i], path[i + 1] - n1
            if u_p < n1:
                matching[u_p] = v_p
        self.second_matching = matching
        self.second_res = self.best_res - cycle_min_weight

    def split(self):
        u, v = self.branch_edge
        I = self.I.copy()
        self.I.add(u)
        O = self.O.copy()
        O.append((u, v))

        G = self.G
        second_matching = self.second_matching
        self.second_matching = None
        second_res = self.second_res
        self.second_res = None

        self._get_second_matching()
        return _Subspace(G, second_matching, second_res, I, O)


class KBestMSolver:
    """K-best bipartite matching solver with GED lower bound pruning.

    Takes a predicted mapping weight matrix and finds the k-best matchings,
    computing actual GED for each using adjacency/feature comparison.

    Args:
        weight_matrix: Predicted mapping weights, shape (n1, n2), n1 <= n2.
        A1: Adjacency matrix of graph 1.
        A2: Adjacency matrix of graph 2.
        f1: Node features of graph 1.
        f2: Node features of graph 2.
        pre_ged: Optional neural GED prediction for additional pruning.
    """

    def __init__(
        self,
        weight_matrix: torch.Tensor,
        A1: torch.Tensor,
        A2: torch.Tensor,
        f1: torch.Tensor,
        f2: torch.Tensor,
        pre_ged: float = None,
    ):
        self.A1 = A1
        self.A2 = A2
        self.f1 = f1
        self.f2 = f2
        self.pre_ged = pre_ged

        G, best_matching, res = self._tensor_to_nx(weight_matrix)
        sp = _Subspace(G, best_matching, res)

        self.lb = _GedLowerBound(A1, A2, f1, f2)
        sp.lb = self.lb.compute([], best_matching)
        sp.ged = self.lb.compute([], best_matching)
        self.min_ged = sp.ged
        if sp.second_matching is not None:
            sp.ged2 = self.lb.compute([], sp.second_matching)
            self._update_min_ged(sp.ged2)

        self.subspaces = [sp]
        self.k = 1
        self.expandable = True

    def _update_min_ged(self, ged):
        if ged is not None and ged < self.min_ged:
            self.min_ged = ged

    @staticmethod
    def _tensor_to_nx(A: torch.Tensor):
        n1, n2 = A.shape
        top_nodes = range(n1)
        bottom_nodes = range(n1, n1 + n2)

        G = nx.DiGraph()
        G.add_nodes_from(top_nodes, bipartite=0)
        G.add_nodes_from(bottom_nodes, bipartite=1)
        A_list = A.tolist()
        for u in top_nodes:
            for v in bottom_nodes:
                G.add_edge(u, v, weight=-A_list[u][v - n1])

        matching = bipartite.matching.minimum_weight_full_matching(G, top_nodes)
        matching = [matching[u] - n1 for u in top_nodes]
        res = sum(A_list[u][matching[u]] for u in top_nodes)

        return G, matching, res

    def _expand(self):
        max_res = -1
        max_spid = None

        for spid, sp in enumerate(self.subspaces):
            if sp.lb < self.min_ged and sp.second_res is not None and sp.second_res > max_res:
                max_res = sp.second_res
                max_spid = spid

        if max_spid is None:
            self.expandable = False
            return

        sp = self.subspaces[max_spid]
        sp_new = sp.split()
        self.subspaces.append(sp_new)
        self.k += 1

        sp_new.lb = sp.lb
        sp_new.ged = sp.ged2
        if sp_new.second_matching is not None:
            sp_new.ged2 = self.lb.compute([], sp_new.second_matching)
            self._update_min_ged(sp_new.ged2)

        left_nodes = list(sp.I)
        right_nodes = [sp.best_matching[u] for u in left_nodes]
        sp.lb = self.lb.compute(left_nodes, right_nodes)
        if sp.second_matching is not None:
            sp.ged2 = self.lb.compute([], sp.second_matching)
            self._update_min_ged(sp.ged2)

    def solve(self, k: int = 10):
        """Find the best matching among k-best candidates.

        Args:
            k: Maximum number of matchings to explore.

        Returns:
            Tuple of (best_ged, best_matching) where best_matching is a list
            mapping graph1 node indices to graph2 node indices.
        """
        while self.k < k and self.expandable:
            self._expand()

        best_ged = float("inf")
        best_matching = None
        for sp in self.subspaces:
            if sp.ged is not None and sp.ged < best_ged:
                best_ged = sp.ged
                best_matching = sp.best_matching
        return best_ged, best_matching
