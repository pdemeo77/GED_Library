"""Hausdorff Edit Distance (HED) solver.

Reference:
    Fischer, A., Serrat, A., Riesen, K., & Bunke, H. (2015).
    Approximation of graph edit distance based on Hausdorff matching.
    Pattern Recognition, 48(2), 331-343.

Faithful port of the reference implementation (priba/aproximated_ged,
HausdorffEditDistance.py + VanillaHED.py, networkx-1.x) using the
library's CostModel for the edit costs.

The algorithm has no LSAP step: every node independently takes the
cheapest between (a) being deleted/inserted with half of its incident
edge costs charged at each endpoint, and (b) the best Hausdorff-style
substitution with a node of the other graph. The result is a
lower-bound style approximation (<= exact GED under consistent costs)
and carries the node-count lower bound L_graph as a final floor.
"""

from typing import Optional

import numpy as np

import networkx as nx

from ged_lib.base import GEDSolver
from ged_lib.cost import CostModel
from ged_lib.solvers import register_solver


@register_solver("hed")
class HEDSolver(GEDSolver):
    """Hausdorff Edit Distance approximation (Fischer et al., 2015).

    Args:
        cost_model: CostModel providing node/edge edit costs. Defaults to
            the symbolic-label unit-cost model, like the other solvers.
    """

    def __init__(self, cost_model: Optional[CostModel] = None) -> None:
        super().__init__(cost_model)

    def _hausdorff_edge_cost(self, g1: nx.Graph, g2: nx.Graph, u, v) -> float:
        """Hausdorff-style matching of the edges incident to u (in g1) and v (in g2).

        Mirrors upstream ``hec``: deletion/insertion entries improved by the
        cheapest substitution (halved), summed over both sides.
        """
        edges_u = list(g1.adj[u].keys())
        edges_v = list(g2.adj[v].keys())

        c1 = [self.cost_model.edge_del_cost(g1, (u, nbr)) for nbr in edges_u]
        c2 = [self.cost_model.edge_ins_cost(g2, (v, nbr)) for nbr in edges_v]

        for i, nbr_u in enumerate(edges_u):
            e1 = (u, nbr_u)
            for j, nbr_v in enumerate(edges_v):
                c = self.cost_model.edge_sub_cost(g1, g2, e1, (v, nbr_v)) / 2.0
                if c < c1[i]:
                    c1[i] = c
                if c < c2[j]:
                    c2[j] = c

        return float(np.sum(c1) + np.sum(c2))

    def _l_edges(self, g1: nx.Graph, g2: nx.Graph, u, v) -> float:
        """Degree-difference lower bound on the edge cost of substituting u with v."""
        deg_u = g1.degree(u)
        deg_v = g2.degree(v)
        if deg_u == deg_v:
            return 0.0
        if deg_u > deg_v:
            excess = range(deg_u - deg_v)
            side_costs = [self.cost_model.edge_del_cost(g1, (u, nbr)) for nbr in g1.adj[u]]
        else:
            excess = range(deg_v - deg_u)
            side_costs = [self.cost_model.edge_ins_cost(g2, (v, nbr)) for nbr in g2.adj[v]]
        return len(excess) * float(np.min(side_costs)) if side_costs else 0.0

    def compute(self, g1: nx.Graph, g2: nx.Graph) -> float:
        """Compute the Hausdorff Edit Distance between two graphs."""
        nodes1 = sorted(g1.nodes())
        nodes2 = sorted(g2.nodes())
        n1, n2 = len(nodes1), len(nodes2)

        if n1 == 0 and n2 == 0:
            return 0.0

        # Deletion: node cost + half of the incident edge costs (each edge is
        # charged at both endpoints).
        d1 = []
        for u in nodes1:
            edge_part = sum(self.cost_model.edge_del_cost(g1, (u, nbr)) for nbr in g1.adj[u])
            d1.append(self.cost_model.node_del_cost(g1, u) + edge_part / 2.0)

        # Insertion
        d2 = []
        for v in nodes2:
            edge_part = sum(self.cost_model.edge_ins_cost(g2, (v, nbr)) for nbr in g2.adj[v])
            d2.append(self.cost_model.node_ins_cost(g2, v) + edge_part / 2.0)

        # Substitution: every node independently takes its best option.
        for i, u in enumerate(nodes1):
            for j, v in enumerate(nodes2):
                hec = self._hausdorff_edge_cost(g1, g2, u, v)
                ce = max(self._l_edges(g1, g2, u, v), hec)
                ce = (self.cost_model.node_sub_cost(g1, g2, u, v) + ce / 2.0) / 2.0
                if ce < d1[i]:
                    d1[i] = ce
                if ce < d2[j]:
                    d2[j] = ce

        dist = float(np.sum(d1) + np.sum(d2))

        # Node-count lower bound floor (upstream L_graph).
        if n1 > n2:
            floor = (n1 - n2) * min(
                (self.cost_model.node_del_cost(g1, u) for u in nodes1), default=0.0
            )
        else:
            floor = (n2 - n1) * min(
                (self.cost_model.node_ins_cost(g2, v) for v in nodes2), default=0.0
            )
        return float(max(floor, dist))
