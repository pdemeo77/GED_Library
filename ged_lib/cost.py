"""CostModel: configurable edit cost functions for GED computation."""

from typing import Callable, Optional

import networkx as nx


class CostModel:
    """Defines the cost of graph edit operations.

    The cost model determines the penalty for inserting, deleting,
    and substituting nodes and edges. This is decoupled from solvers
    so that different cost schemes can be applied to any algorithm.

    Args:
        node_ins_cost: Cost of inserting a node.
        node_del_cost: Cost of deleting a node.
        node_sub_cost: Cost of substituting node u with node v.
            Can be a float or a callable (g1, g2, u, v) -> float.
        edge_ins_cost: Cost of inserting an edge.
        edge_del_cost: Cost of deleting an edge.
        edge_sub_cost: Cost of substituting edge e1 with edge e2.
            Can be a float or a callable (g1, g2, e1, e2) -> float.
    """

    def __init__(
        self,
        node_ins_cost: float = 1.0,
        node_del_cost: float = 1.0,
        node_sub_cost: Optional[Callable] = None,
        edge_ins_cost: float = 1.0,
        edge_del_cost: float = 1.0,
        edge_sub_cost: Optional[Callable] = None,
    ) -> None:
        self._node_ins_cost = node_ins_cost
        self._node_del_cost = node_del_cost
        self._edge_ins_cost = edge_ins_cost
        self._edge_del_cost = edge_del_cost

        if node_sub_cost is None:
            self._node_sub_cost = self._default_node_sub_cost
        else:
            self._node_sub_cost = node_sub_cost

        if edge_sub_cost is None:
            self._edge_sub_cost = self._default_edge_sub_cost
        else:
            self._edge_sub_cost = edge_sub_cost

    @staticmethod
    def _default_node_sub_cost(g1: nx.Graph, g2: nx.Graph, u: int, v: int) -> float:
        """Default node substitution cost: 0 if same label, 1 otherwise."""
        label_u = g1.nodes[u].get("label", None)
        label_v = g2.nodes[v].get("label", None)
        return 0.0 if label_u == label_v else 1.0

    @staticmethod
    def _default_edge_sub_cost(g1: nx.Graph, g2: nx.Graph, e1: tuple, e2: tuple) -> float:
        """Default edge substitution cost: 0.0 for unlabeled edges.

        In the Riesen-Bunke bipartite matching algorithm, matched (substituted)
        edges are preserved under the node assignment — they should incur no cost.
        Only label mismatches on edges (if edge labels exist) should cost > 0.
        """
        return 0.0

    def node_ins_cost(self, g: nx.Graph, u: int) -> float:
        """Return the cost of inserting node u into graph g."""
        return self._node_ins_cost

    def node_del_cost(self, g: nx.Graph, u: int) -> float:
        """Return the cost of deleting node u from graph g."""
        return self._node_del_cost

    def node_sub_cost(self, g1: nx.Graph, g2: nx.Graph, u: int, v: int) -> float:
        """Return the cost of substituting node u (in g1) with node v (in g2)."""
        return self._node_sub_cost(g1, g2, u, v)

    def edge_ins_cost(self, g: nx.Graph, e: tuple) -> float:
        """Return the cost of inserting edge e into graph g."""
        return self._edge_ins_cost

    def edge_del_cost(self, g: nx.Graph, e: tuple) -> float:
        """Return the cost of deleting edge e from graph g."""
        return self._edge_del_cost

    def edge_sub_cost(self, g1: nx.Graph, g2: nx.Graph, e1: tuple, e2: tuple) -> float:
        """Return the cost of substituting edge e1 (in g1) with edge e2 (in g2)."""
        return self._edge_sub_cost(g1, g2, e1, e2)


class UnitCostModel(CostModel):
    """All edit operations cost exactly 1.0."""

    def __init__(self) -> None:
        super().__init__(
            node_ins_cost=1.0,
            node_del_cost=1.0,
            node_sub_cost=lambda g1, g2, u, v: 1.0,
            edge_ins_cost=1.0,
            edge_del_cost=1.0,
            edge_sub_cost=lambda g1, g2, e1, e2: 1.0,
        )


class LabelCostModel(CostModel):
    """Node substitution cost is 0 if labels match, 1 otherwise.

    Suitable for labeled graphs like AIDS where node labels (chemical elements)
    carry important semantic information.
    """

    def __init__(self) -> None:
        super().__init__(
            node_ins_cost=1.0,
            node_del_cost=1.0,
            edge_ins_cost=1.0,
            edge_del_cost=1.0,
        )
