"""Hungarian (Bipartite Matching) solver for Graph Edit Distance.

Reference:
    Riesen, K., & Bunke, H. (2009).
    Approximate graph edit distance computation by means of bipartite graph matching.
    Image and Vision Computing, 27(7), 950-959.

This implementation uses a cost-matrix approach where node substitution costs
are derived from local neighborhood structure, then the Hungarian algorithm
(scipy.optimize.linear_sum_assignment) finds the optimal assignment.
The total GED is estimated from the assignment cost plus edge edit costs.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

import networkx as nx

from ged_lib.base import GEDSolver
from ged_lib.cost import CostModel
from ged_lib.solvers import register_solver


@register_solver("hungarian")
class HungarianSolver(GEDSolver):
    """Bipartite matching heuristic for Graph Edit Distance.

    Builds a cost matrix where entry (i, j) represents the cost of
    substituting node i from g1 with node j from g2, using local
    neighborhood information. Then solves the assignment problem
    with the Hungarian algorithm to find the minimum-cost matching.

    Args:
        cost_model: CostModel instance defining edit costs.
    """

    def __init__(self, cost_model: CostModel = None) -> None:
        super().__init__(cost_model)

    def compute(self, g1: nx.Graph, g2: nx.Graph) -> float:
        """Compute approximate GED using bipartite matching.

        Args:
            g1: First graph.
            g2: Second graph.

        Returns:
            Approximate Graph Edit Distance.
        """
        n1 = g1.number_of_nodes()
        n2 = g2.number_of_nodes()

        if n1 == 0 and n2 == 0:
            return 0.0

        nodes1 = sorted(g1.nodes())
        nodes2 = sorted(g2.nodes())

        node_map1 = {node: i for i, node in enumerate(nodes1)}
        node_map2 = {node: i for i, node in enumerate(nodes2)}

        cost_matrix = self._build_cost_matrix(g1, g2, nodes1, nodes2, node_map1, node_map2)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        total_cost = cost_matrix[row_ind, col_ind].sum()

        return float(total_cost)

    def _build_cost_matrix(
        self,
        g1: nx.Graph,
        g2: nx.Graph,
        nodes1: list,
        nodes2: list,
        node_map1: dict,
        node_map2: dict,
    ) -> np.ndarray:
        """Build the cost matrix for the assignment problem.

        The matrix has shape (n1 + n2, n1 + n2):
        - Top-left (n1 x n2): substitution costs
        - Top-right (n1 x n1): deletion costs (node i -> dummy)
        - Bottom-left (n2 x n2): insertion costs (dummy -> node j)
        - Bottom-right (n2 x n1): zero (dummy -> dummy)

        Args:
            g1, g2: Input graphs.
            nodes1, nodes2: Sorted node lists.
            node_map1, node_map2: Node-to-index mappings.

        Returns:
            Cost matrix of shape (n1+n2, n1+n2).
        """
        n1 = len(nodes1)
        n2 = len(nodes2)
        dim = n1 + n2

        C = np.zeros((dim, dim), dtype=np.float64)

        ins_cost = self.cost_model.node_ins_cost(g2, 0)
        del_cost = self.cost_model.node_del_cost(g1, 0)

        for i, u in enumerate(nodes1):
            for j, v in enumerate(nodes2):
                C[i, j] = self._substitution_cost(g1, g2, u, v, node_map1, node_map2)

        for i in range(n1):
            for j in range(n2, dim):
                C[i, j] = del_cost + self._edge_deletion_cost(g1, nodes1[i])

        for i in range(n1, dim):
            for j in range(n2):
                C[i, j] = ins_cost + self._edge_insertion_cost(g2, nodes2[j])

        return C

    def _substitution_cost(
        self,
        g1: nx.Graph,
        g2: nx.Graph,
        u,
        v,
        node_map1: dict,
        node_map2: dict,
    ) -> float:
        """Compute the cost of substituting node u (in g1) with node v (in g2).

        Includes node label substitution cost plus edge edit cost
        from aligning their neighborhoods.

        Args:
            g1, g2: Input graphs.
            u: Node in g1.
            v: Node in g2.
            node_map1, node_map2: Node-to-index mappings.

        Returns:
            Substitution cost as a float.
        """
        node_cost = self.cost_model.node_sub_cost(g1, g2, u, v)

        neighbors_u = set(g1.neighbors(u))
        neighbors_v = set(g2.neighbors(v))

        edge_sub = min(len(neighbors_u), len(neighbors_v))
        edge_del = max(0, len(neighbors_u) - len(neighbors_v))
        edge_ins = max(0, len(neighbors_v) - len(neighbors_u))

        edge_cost = (
            edge_sub * self.cost_model.edge_sub_cost(g1, g2, (u, u), (v, v))
            + edge_del * self.cost_model.edge_del_cost(g1, (u, u))
            + edge_ins * self.cost_model.edge_ins_cost(g2, (v, v))
        )

        return node_cost + edge_cost

    def _edge_deletion_cost(self, g: nx.Graph, u) -> float:
        """Cost of deleting all edges incident to node u."""
        return g.degree(u) * self.cost_model.edge_del_cost(g, (u, u))

    def _edge_insertion_cost(self, g: nx.Graph, v) -> float:
        """Cost of inserting all edges incident to node v."""
        return g.degree(v) * self.cost_model.edge_ins_cost(g, (v, v))
