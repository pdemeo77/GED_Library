"""Tests for GED_Library core modules."""

import networkx as nx
import numpy as np
import pytest


class TestCostModel:
    def test_default_sub_same_label(self):
        from ged_lib.cost import CostModel

        cm = CostModel()
        g = nx.Graph()
        g.add_node(0, label="C")
        g.add_node(1, label="C")
        assert cm.node_sub_cost(g, g, 0, 1) == 0.0

    def test_default_sub_diff_label(self):
        from ged_lib.cost import CostModel

        cm = CostModel()
        g1 = nx.Graph()
        g1.add_node(0, label="C")
        g2 = nx.Graph()
        g2.add_node(0, label="N")
        assert cm.node_sub_cost(g1, g2, 0, 0) == 1.0

    def test_float_sub_costs_accepted(self):
        """Float substitution costs must work as documented (regression: raised TypeError)."""
        from ged_lib.cost import CostModel

        cm = CostModel(node_sub_cost=0.5, edge_sub_cost=0.25)
        g1 = nx.Graph()
        g1.add_node(0, label="C")
        g1.add_node(1, label="C")
        g1.add_edge(0, 1)
        g2 = nx.Graph()
        g2.add_node(0, label="N")
        g2.add_node(1, label="N")
        g2.add_edge(0, 1)
        assert cm.node_sub_cost(g1, g2, 0, 0) == 0.5
        assert cm.edge_sub_cost(g1, g2, (0, 1), (0, 1)) == 0.25

    def test_unit_cost(self):
        from ged_lib.cost import UnitCostModel

        cm = UnitCostModel()
        g1 = nx.Graph()
        g1.add_node(0, label="C")
        g2 = nx.Graph()
        g2.add_node(0, label="C")
        assert cm.node_sub_cost(g1, g2, 0, 0) == 1.0


class TestGraphIO:
    def test_csv_row_to_graph(self):
        from ged_lib.graph import csv_row_to_graph

        g = csv_row_to_graph("[[0, 1], [1, 2]]", ["C", "N", "O"])
        assert g.number_of_nodes() == 3
        assert g.number_of_edges() == 2
        assert g.nodes[0]["label"] == "C"

    def test_csv_row_unlabeled(self):
        from ged_lib.graph import csv_row_to_graph

        g = csv_row_to_graph("[[0, 1], [1, 2]]")
        assert g.number_of_edges() == 2

    def test_nx_to_tensors(self):
        from ged_lib.graph import nx_to_tensors

        g = nx.Graph()
        g.add_node(0, label="C")
        g.add_node(1, label="N")
        g.add_edge(0, 1)
        t = nx_to_tensors(g)
        assert t["n"] == 2
        assert t["m"] == 1
        assert t["adj"].shape == (2, 2)
        assert t["has_labels"] is True


class TestDatasetLoader:
    def test_load_aids(self):
        from ged_lib import load_dataset

        graphs, gt = load_dataset("AIDS")
        assert len(graphs) > 0
        assert len(gt) > 0
        g = list(graphs.values())[0]
        assert isinstance(g, nx.Graph)

    def test_load_linux(self):
        from ged_lib import load_dataset

        graphs, gt = load_dataset("Linux")
        assert len(graphs) > 0

    def test_unknown_dataset(self):
        from ged_lib import load_dataset

        with pytest.raises(ValueError):
            load_dataset("UNKNOWN")


class TestHungarianSolver:
    def test_identical_graphs(self):
        from ged_lib import HungarianSolver

        g = nx.Graph()
        g.add_node(0)
        g.add_node(1)
        solver = HungarianSolver()
        assert solver.compute(g, g) == 0.0

    def test_empty_graphs(self):
        from ged_lib import HungarianSolver

        g1 = nx.Graph()
        g2 = nx.Graph()
        solver = HungarianSolver()
        assert solver.compute(g1, g2) == 0.0

    def test_labeled_graphs(self):
        from ged_lib import HungarianSolver

        g1 = nx.Graph()
        g1.add_node(0, label="C")
        g1.add_node(1, label="N")
        g1.add_edge(0, 1)
        g2 = nx.Graph()
        g2.add_node(0, label="C")
        g2.add_node(1, label="O")
        g2.add_edge(0, 1)
        solver = HungarianSolver()
        ged = solver.compute(g1, g2)
        assert ged > 0.0

    def test_pairwise(self):
        from ged_lib import HungarianSolver

        g1 = nx.Graph()
        g1.add_node(0)
        g1.add_node(1)
        g2 = nx.Graph()
        g2.add_node(0)
        g2.add_node(1)
        g3 = nx.Graph()
        g3.add_node(0)
        solver = HungarianSolver()
        D = solver.compute_pairwise([g1, g2, g3])
        assert D.shape == (3, 3)
        assert D[0, 1] == 0.0
        assert D[0, 0] == 0.0
        assert D[0, 2] > 0.0


class TestGEDGWSolver:
    def test_identical_graphs(self):
        from ged_lib import GEDGWSolver

        g = nx.Graph()
        g.add_edge(0, 1)
        solver = GEDGWSolver()
        assert solver.compute(g, g) == pytest.approx(0.0, abs=1e-6)

    def test_different_structure(self):
        from ged_lib import GEDGWSolver

        g1 = nx.Graph()
        g1.add_edge(0, 1)
        g2 = nx.Graph()
        g2.add_node(0)
        g2.add_node(1)
        solver = GEDGWSolver()
        ged = solver.compute(g1, g2)
        assert ged > 0.0


class TestEvaluation:
    def test_evaluation_result(self):
        from ged_lib.evaluation.metrics import EvaluationResult

        r = EvaluationResult(solver_name="test", dataset_name="AIDS", mae=0.5)
        d = r.to_dict()
        assert d["mae"] == 0.5
        assert "solver=" in str(r)


class TestSolverRegistry:
    def test_list_solvers(self):
        from ged_lib import list_solvers

        solvers = list_solvers()
        assert "hungarian" in solvers
        assert "gedgw" in solvers

    def test_solver_names(self):
        from ged_lib import list_solvers

        solvers = list_solvers()
        for name, cls in solvers.items():
            assert hasattr(cls, "compute")


class TestLabelEncoding:
    """Tests that verify label features are correctly encoded (§1.1 regression)."""

    def test_aids_label_features_not_all_zeros(self):
        """One-hot features for AIDS graphs must not be all-zeros (§1.1)."""
        import torch

        from ged_lib import load_dataset
        from ged_lib.solvers.neural.data_adapter import (
            nx_to_neural_data,
            get_num_labels,
            get_label_map,
        )

        graphs, _ = load_dataset("AIDS")
        num_labels = get_num_labels(graphs, "AIDS")
        label_map = get_label_map("AIDS")

        g = list(graphs.values())[0]
        data = nx_to_neural_data(
            g, g, num_labels=num_labels, device=torch.device("cpu"), label_map=label_map
        )
        features = data["features_1"]
        assert features.shape[1] == 29
        assert features.sum().item() > 0, "One-hot features should not be all-zeros"
        # Each node should have exactly one one-hot bit set
        assert torch.allclose(features.sum(dim=1), torch.ones(features.shape[0]))

    def test_label_map_loads_correctly(self):
        from ged_lib.solvers.neural.data_adapter import get_label_map

        lm = get_label_map("AIDS")
        assert lm is not None
        assert len(lm) == 29
        assert lm["C"] == 4
        assert lm["N"] == 14
        assert lm["O"] == 16

    def test_resolve_label_with_map(self):
        from ged_lib.solvers.neural.data_adapter import resolve_label

        lm = {"C": 4, "N": 14}
        assert resolve_label("C", 29, lm) == 4
        assert resolve_label("N", 29, lm) == 14
        assert resolve_label("X", 29, lm) == -1

    def test_resolve_label_without_map(self):
        from ged_lib.solvers.neural.data_adapter import resolve_label

        assert resolve_label("C", 29, None) == -1
        assert resolve_label(4, 29, None) == 4
        assert resolve_label(99, 29, None) == -1

    def test_nx_to_tensors_with_label_map(self):
        from ged_lib.graph import nx_to_tensors

        g = nx.Graph()
        g.add_node(0, label="C")
        g.add_node(1, label="N")
        g.add_edge(0, 1)
        label_map = {"C": 4, "N": 14}
        t = nx_to_tensors(g, num_labels=29, label_map=label_map)
        assert t["features"].shape == (2, 29)
        assert t["features"][0, 4] == 1.0  # C -> index 4
        assert t["features"][1, 14] == 1.0  # N -> index 14
        assert t["features"].sum() == 2.0


class TestNeuralSolvers:
    """Smoke tests: forward pass on tiny graphs for all neural solvers."""

    @staticmethod
    def _make_pair(solver_class, num_labels=2, device="cpu", **kwargs):
        """Build a tiny 2-3 node graph pair and run compute()."""
        import torch

        g1 = nx.Graph()
        g1.add_node(0, label=0)
        g1.add_node(1, label=1)
        g1.add_edge(0, 1)

        g2 = nx.Graph()
        g2.add_node(0, label=0)
        g2.add_node(1, label=1)
        g2.add_node(2, label=0)
        g2.add_edge(0, 1)
        g2.add_edge(1, 2)

        solver = solver_class(num_labels=num_labels, device=device, **kwargs)
        ged = solver.compute(g1, g2)
        return ged

    def test_gediot_smoke(self):
        from ged_lib import GEDIOTSolver

        ged = self._make_pair(GEDIOTSolver)
        assert isinstance(ged, float)
        assert ged >= 0

    def test_gedgnn_smoke(self):
        from ged_lib import GedGNNSolver

        ged = self._make_pair(GedGNNSolver)
        assert isinstance(ged, float)

    def test_simgnn_smoke(self):
        from ged_lib import SimGNNSolver

        ged = self._make_pair(SimGNNSolver)
        assert isinstance(ged, float)

    def test_tagsim_smoke(self):
        from ged_lib import TaGSimSolver

        ged = self._make_pair(TaGSimSolver)
        assert isinstance(ged, float)

    def test_gpn_smoke(self):
        from ged_lib import GPNSolver

        ged = self._make_pair(GPNSolver)
        assert isinstance(ged, float)

    def test_egsc_smoke(self):
        from ged_lib import EGSCSolver

        ged = self._make_pair(EGSCSolver)
        assert isinstance(ged, float)

    def test_greed_smoke(self):
        from ged_lib import GREEDSolver

        ged = self._make_pair(GREEDSolver)
        assert isinstance(ged, float)

    def test_gedgnn_full_smoke(self):
        from ged_lib import GEDGNNFullSolver

        ged = self._make_pair(GEDGNNFullSolver)
        assert isinstance(ged, float)

    def test_grasp_smoke(self):
        from ged_lib import GraspSolver

        ged = self._make_pair(GraspSolver)
        assert isinstance(ged, float)


class TestKBestMatchingFixes:
    """Regression tests for the GedGNN-Full k-best post-processing fixes.

    Covers verification_report.md statements S1-S3 plus the reviewer's
    additional findings (unlabeled node term, weight-matrix scaling).
    """

    @staticmethod
    def _path_vs_triangle():
        """g1: path 0-1-2 (2 edges), g2: triangle (3 edges); distinct labels.

        Adjacencies include self-loops exactly like production edge_index.
        True GED of the identity mapping = 1 (one edge insertion).
        """
        import torch

        A1 = torch.eye(3)
        A1[0, 1] = A1[1, 0] = 1.0
        A1[1, 2] = A1[2, 1] = 1.0
        A2 = torch.ones(3, 3)
        f = torch.eye(3)
        return A1, A2, f, f

    def test_root_lb_is_unconstrained(self):
        """Root lower bound must be the full-space bound (|E1-E2| + label term),
        not the best matching's own GED (which disabled the whole search)."""
        import torch

        from ged_lib.solvers.neural.kbest_matching import _GedLowerBound

        A1, A2, f1, f2 = self._path_vs_triangle()
        lb = _GedLowerBound(A1, A2, f1, f2)
        # |E1 - E2| = 1; label sums identical -> 0; unconstrained space
        assert lb.compute([], []) == pytest.approx(1.0)

    def test_full_mapping_edge_term(self):
        """GED of the identity mapping must be 1.0 (one edge insertion).

        The old total-edge-count formula returned 1.5 (the reviewer's
        synthetic case): it added |E1-E2|/2 even for full mappings.
        """
        from ged_lib.solvers.neural.kbest_matching import _GedLowerBound

        A1, A2, f1, f2 = self._path_vs_triangle()
        lb = _GedLowerBound(A1, A2, f1, f2)
        assert lb.compute([], [0, 1, 2]) == pytest.approx(1.0)

    def test_unlabeled_node_term(self):
        """Unlabeled graphs: node term must be |n2-n1| (upstream branch),
        not the constant-feature intersection that gave 2*(n2-n1)."""
        import torch

        from ged_lib.solvers.neural.kbest_matching import _GedLowerBound

        A1 = torch.ones(2, 2)  # edge 0-1 + self-loops
        A2 = torch.zeros(3, 3)
        A2[0, 0] = A2[1, 1] = A2[2, 2] = 1.0
        A2[0, 1] = A2[1, 0] = 1.0  # edge 0-1, node 2 isolated
        f1 = torch.full((2, 1), 2.0)
        f2 = torch.full((3, 1), 2.0)
        lb = _GedLowerBound(A1, A2, f1, f2)
        assert lb.compute([0], [1]) == pytest.approx(1.0)

    def test_kbest_expands_and_finds_optimum(self):
        """With weights forcing a bad first matching, the search must expand
        (k > 1) and discover the identity matching with GED 1."""
        import torch

        from ged_lib.solvers.neural.kbest_matching import KBestMSolver

        A1 = torch.ones(2, 2)  # g1: edge 0-1 + loops
        A2 = torch.zeros(3, 3)
        for i in range(3):
            A2[i, i] = 1.0
        A2[0, 1] = A2[1, 0] = 1.0  # g2: edge 0-1 + isolated node 2

        f1 = torch.zeros(2, 3)
        f1[0, 0] = f1[1, 1] = 1.0  # labels a, b
        f2 = torch.eye(3)  # labels a, b, c

        # Best full matching (max weight) = {0->2, 1->0} (GED 5); the optimum
        # identity {0->0, 1->1} (GED 1) must be reached by k-best expansion.
        W = torch.tensor([[1.0, 1.0, 10.0], [10.0, 1.0, 1.0]])
        solver = KBestMSolver(W, A1, A2, f1, f2)
        solver.solve(k=10)
        assert solver.k > 1, "k-best search never expanded (dead code regression)"
        assert solver.min_ged == pytest.approx(1.0)

    def test_weight_matrix_softmax_scaled(self, monkeypatch):
        """The matrix handed to KBestMSolver must be round(softmax*1e9+1):
        positive integers (raw logits break the max-weight search)."""
        import torch

        import ged_lib.solvers.neural.gedgnn_full as gedgnn_full_mod

        captured = {}

        class _Capture:
            def __init__(self, weight, *args, **kwargs):
                captured["weight"] = weight
                raise RuntimeError("capture and stop")

        monkeypatch.setattr(gedgnn_full_mod, "KBestMSolver", _Capture)
        solver = gedgnn_full_mod.GEDGNNFullSolver(num_labels=2)
        g1 = nx.Graph()
        g1.add_node(0, label=0)
        g1.add_node(1, label=1)
        g1.add_edge(0, 1)
        g2 = nx.Graph()
        g2.add_node(0, label=0)
        g2.add_node(1, label=1)
        g2.add_node(2, label=0)
        g2.add_edge(0, 1)
        result = solver.compute(g1, g2)  # falls back to the neural GED
        assert isinstance(result, float) and result >= 0.0
        w = captured["weight"]
        assert w.min().item() >= 1.0
        assert torch.equal(w, w.round())


class TestGraspFixes:
    """Regression tests for the Grasp fidelity fixes (verification_report.md S5)."""

    @staticmethod
    def _path_edge_index():
        """Path 0-1-2, direction-duplicated + self-loops (production format)."""
        import torch

        return torch.tensor(
            [[0, 1, 1, 2, 0, 1, 2], [1, 0, 2, 1, 0, 1, 2]], dtype=torch.long
        )

    def test_degree_excludes_self_loops(self):
        """Degrees must equal the undirected degree (self-loops skipped, one
        count per direction pair); the old version added the self-loop (+1)."""
        import torch

        from ged_lib import GraspSolver

        s = GraspSolver(num_labels=2, filters=8)
        deg = s._compute_degree(self._path_edge_index(), 3)
        assert torch.equal(deg, torch.tensor([[1.0], [2.0], [1.0]]))

    def test_final_mlp_input_width(self):
        """Final MLP input = raw features + 6 level outputs (degree NOT in)."""
        from ged_lib import GraspSolver

        s = GraspSolver(num_labels=4, filters=8)
        assert s.mlp[0].in_features == 4 + 6 * 8

    def test_pattern_features_are_raw_conv_outputs(self):
        """Level features feeding pattern matrices must be raw conv outputs
        (the old code applied ReLU, and dropout, before both concat and patterns)."""
        import torch

        from ged_lib import GraspSolver

        torch.manual_seed(0)
        s = GraspSolver(num_labels=2, filters=8)
        s.train()
        features = torch.eye(2)
        edge_index = torch.tensor([[0, 1, 0, 1], [1, 0, 0, 1]], dtype=torch.long)
        _, _, x1, *_ = s._conv_pass(edge_index, features)
        assert (x1 < 0).any(), "level features must be raw conv outputs, not post-ReLU"

    def test_dropout_zero_deterministic_training(self):
        """dropout=0.0 must make the conv pass deterministic in train mode."""
        import torch

        from ged_lib import GraspSolver

        torch.manual_seed(0)
        s = GraspSolver(num_labels=2, filters=8, dropout=0.0)
        s.train()
        features = torch.eye(2)
        edge_index = torch.tensor([[0, 1, 0, 1], [1, 0, 0, 1]], dtype=torch.long)
        out_a = s._conv_pass(edge_index, features)[0]
        out_b = s._conv_pass(edge_index, features)[0]
        assert torch.equal(out_a, out_b)


class TestEGSCFixes:
    """Regression tests for the EGSC fidelity fix (verification_report.md S4)."""

    def test_se_tensor_network_head_depth(self):
        """The SE tensor-network head must match upstream fc1: exactly two
        Linear layers (the fc0 block is dead code upstream)."""
        import torch

        from ged_lib.solvers.neural.egsc import _SETensorNetwork

        dim = 16
        mod = _SETensorNetwork(dim)
        linears = [m for m in mod.fc_out if isinstance(m, torch.nn.Linear)]
        assert len(linears) == 2
        assert linears[0].in_features == 2 * dim
        assert linears[0].out_features == 2 * dim
        assert linears[1].out_features == dim // 2
        emb1 = torch.randn(dim, 1)  # graph-level embeddings (dim, 1)
        emb2 = torch.randn(dim, 1)
        out = mod(emb1, emb2)
        assert out.shape == (1, dim // 2)


class TestNeuralEmptyGraphGuard:
    """Regression test for the P1 fix: neural solvers on two empty graphs.

    Previously this hit a division-by-zero (avg_v == 0) and/or a BatchNorm
    error on 0 samples. It must now return 0.0.
    """

    def test_empty_graphs_gediot(self):
        import networkx as nx

        from ged_lib import GEDIOTSolver

        g = nx.Graph()
        solver = GEDIOTSolver(num_labels=2)
        assert solver.compute(g, g) == 0.0

    def test_empty_graphs_greed_exp_mode(self):
        import networkx as nx

        from ged_lib import GREEDSolver

        g = nx.Graph()
        solver = GREEDSolver(num_labels=2)  # default target_mode="exp"
        assert solver.compute(g, g) == 0.0

    def test_empty_graphs_grasp(self):
        import networkx as nx

        from ged_lib import GraspSolver

        g = nx.Graph()
        solver = GraspSolver(num_labels=2)
        assert solver.compute(g, g) == 0.0


class TestSetSeed:
    def test_set_seed_reproducibility(self):
        from ged_lib import set_seed

        set_seed(123)
        import numpy as np

        a = np.random.rand(5)
        set_seed(123)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)


class TestSolverRepr:
    def test_hungarian_repr(self):
        from ged_lib import HungarianSolver

        solver = HungarianSolver()
        r = repr(solver)
        assert "HungarianSolver" in r

    def test_gediot_repr(self):
        from ged_lib import GEDIOTSolver

        solver = GEDIOTSolver(num_labels=2)
        r = repr(solver)
        assert "GEDIOTSolver" in r


class TestIsolatedNodes:
    """Regression test for §3.1 — isolated nodes preserved in unlabeled graphs."""

    def test_unlabeled_graph_preserves_isolated_nodes(self):
        from ged_lib.graph import csv_row_to_graph

        # Edge list references nodes 0, 1, 2; node 3 is isolated (max_id=2, no issue)
        # Edge list references 0, 2 but not 1; node 1 is isolated
        g = csv_row_to_graph("[[0, 2]]")
        # With the fix, all nodes in [0, max_id=2] should be present
        assert g.number_of_nodes() == 3
        assert 0 in g.nodes()
        assert 1 in g.nodes()
        assert 2 in g.nodes()


class TestHungarianEdgeCost:
    """Regression test for §1.2 — edge substitution cost should be 0.0."""

    def test_edge_sub_cost_is_zero(self):
        from ged_lib.cost import CostModel

        cm = CostModel()
        g = nx.Graph()
        g.add_edge(0, 1)
        assert cm._default_edge_sub_cost(g, g, (0, 1), (0, 1)) == 0.0

    def test_identical_unlabeled_graphs_zero_ged(self):
        from ged_lib import HungarianSolver

        g = nx.Graph()
        g.add_node(0)
        g.add_node(1)
        g.add_edge(0, 1)
        solver = HungarianSolver()
        assert solver.compute(g, g) == 0.0


class TestGEDSFSolver:
    """Smoke tests for GED-SF solvers (Frank-Wolfe + ML regression)."""

    @staticmethod
    def _make_training_data(n_pairs=25):
        """Create tiny labeled graph pairs for training."""
        rng = np.random.RandomState(42)
        graphs = {}
        pairs = []

        # Create 10 small graphs
        for gid in range(10):
            g = nx.Graph()
            n_nodes = rng.randint(3, 6)
            labels = ["C", "N", "O", "S"]
            for i in range(n_nodes):
                g.add_node(i, label=labels[rng.randint(0, len(labels))])
            # Add a few edges
            for i in range(n_nodes - 1):
                if rng.rand() > 0.3:
                    g.add_edge(i, i + 1)
            graphs[gid] = g

        # Create pairs with synthetic GED (edge difference)
        for _ in range(n_pairs):
            id1, id2 = rng.choice(10, size=2, replace=False)
            g1, g2 = graphs[int(id1)], graphs[int(id2)]
            # Synthetic GED: |E1| + |E2| - 2*|intersection|
            e1 = set(g1.edges())
            e2 = set(g2.edges())
            ged = float(len(e1.symmetric_difference(e2)))
            pairs.append((int(id1), int(id2), ged))

        return graphs, pairs

    def test_gedsf_linear_smoke(self):
        """GEDSFSolver(linear) can fit and predict on tiny graphs."""
        from ged_lib import GEDSFSolver

        if GEDSFSolver is None:
            pytest.skip("GEDSFSolver requires scikit-learn")

        graphs, pairs = self._make_training_data()
        solver = GEDSFSolver(
            formulation="linear",
            mu_range=(0.1, 0.5),
            max_iter_fw=10,
            max_graph_size=20,
        )
        solver.fit(graphs, pairs)

        g1, g2 = graphs[0], graphs[1]
        result = solver.compute(g1, g2)
        assert isinstance(result, float)
        assert result >= 0

    def test_gedsf_quadratic_smoke(self):
        """GEDSFSolver(quadratic) can fit and predict on tiny graphs."""
        from ged_lib import GEDSFSolver

        if GEDSFSolver is None:
            pytest.skip("GEDSFSolver requires scikit-learn")

        graphs, pairs = self._make_training_data()
        solver = GEDSFSolver(
            formulation="quadratic",
            mu_range=(0.1, 0.5),
            max_iter_fw=10,
            max_graph_size=20,
        )
        solver.fit(graphs, pairs)

        g1, g2 = graphs[0], graphs[1]
        result = solver.compute(g1, g2)
        assert isinstance(result, float)
        assert result >= 0

    def test_gedsf_repr(self):
        from ged_lib import GEDSFSolver

        if GEDSFSolver is None:
            pytest.skip("GEDSFSolver requires scikit-learn")

        solver = GEDSFSolver(formulation="linear")
        r = repr(solver)
        assert "GEDSFSolver" in r
        assert "unfitted" in r


class TestSession14Fixes:
    """Regression tests covering session-14 bug fixes."""

    # ------------------------------------------------------------------ #
    # 4.1  metrics.py — .statistic fix                                    #
    # ------------------------------------------------------------------ #

    def test_evaluate_solver_returns_spearman(self):
        """evaluate_solver must return numeric spearman_r/kendall_tau without AttributeError."""
        from ged_lib.base import GEDSolver
        from ged_lib.evaluation.metrics import evaluate_solver

        class _ConstantSolver(GEDSolver):
            name = "constant"

            def compute(self, g1, g2):
                return 1.0

        graphs = {}
        gt = []
        for i in range(6):
            g = nx.Graph()
            g.add_node(0)
            g.add_node(1)
            graphs[i] = g
        for i in range(5):
            gt.append((i, i + 1, float(i + 1)))

        solver = _ConstantSolver()
        result = evaluate_solver(solver, graphs, gt, dataset_name="test")
        assert isinstance(result.spearman_r, float)
        assert isinstance(result.kendall_tau, float)

    # ------------------------------------------------------------------ #
    # 4.2  GED-SF smoke + pre-fit guard                                   #
    # ------------------------------------------------------------------ #

    def test_gedsf_compute_before_fit_raises(self):
        """compute() on an unfitted GEDSFSolver must raise RuntimeError."""
        from ged_lib import GEDSFSolver

        solver = GEDSFSolver()
        g = nx.path_graph(3)
        with pytest.raises(RuntimeError):
            solver.compute(g, g)

    def test_gedsf_fit_and_compute_tiny(self):
        """fit() + compute() on 4 tiny synthetic graphs returns a non-negative float."""
        from ged_lib import GEDSFSolver

        rng = np.random.RandomState(0)
        graphs = {}
        for gid in range(4):
            g = nx.Graph()
            for i in range(3):
                g.add_node(i)
            g.add_edge(0, 1)
            if gid % 2 == 0:
                g.add_edge(1, 2)
            graphs[gid] = g

        pairs = [(0, 1, 1.0), (0, 2, 0.0), (1, 3, 1.0),
                 (2, 3, 1.0), (0, 3, 0.0), (1, 2, 1.0),
                 (0, 1, 1.0), (2, 3, 1.0), (1, 3, 1.0),
                 (0, 2, 0.0), (0, 3, 0.0), (1, 2, 1.0),
                 (0, 1, 1.0), (2, 3, 1.0), (0, 3, 0.0),
                 (1, 2, 1.0), (0, 2, 0.0), (1, 3, 1.0),
                 (0, 1, 1.0), (2, 3, 1.0)]

        solver = GEDSFSolver(
            formulation="linear",
            mu_range=(0.1, 0.5),
            max_iter_fw=10,
            max_graph_size=20,
        )
        solver.fit(graphs, pairs)
        result = solver.compute(graphs[0], graphs[1])
        assert isinstance(result, float)
        assert result >= 0.0

    # ------------------------------------------------------------------ #
    # 4.3  GEDGWSolver label regression                                   #
    # ------------------------------------------------------------------ #

    def test_gedgw_labeled_consistent_features(self):
        """GEDGWSolver(dataset_name='AIDS') produces stable results for labeled graphs."""
        from ged_lib import GEDGWSolver

        g1 = nx.Graph()
        g1.add_node(0, label="C")
        g1.add_node(1, label="N")
        g1.add_edge(0, 1)

        g2 = nx.Graph()
        g2.add_node(0, label="C")
        g2.add_node(1, label="O")
        g2.add_node(2, label="N")
        g2.add_edge(0, 1)
        g2.add_edge(1, 2)

        solver = GEDGWSolver(dataset_name="AIDS")
        r1 = solver.compute(g1, g2)
        r2 = solver.compute(g1, g2)
        assert isinstance(r1, float)
        assert r1 == pytest.approx(r2, rel=1e-6), "Result should be deterministic"


class TestFGWAlignSolver:
    """Tests for FGWAlignSolver (Phase 3 — Session 15)."""

    def test_identical_graphs(self):
        """FGWAlign on identical small graphs should return ~0."""
        from ged_lib.solvers.fgwalign import FGWAlignSolver

        g = nx.Graph()
        g.add_node(0)
        g.add_node(1)
        g.add_edge(0, 1)
        solver = FGWAlignSolver(patience=1, topk=1)
        result = solver.compute(g, g)
        assert result == pytest.approx(0.0, abs=1.0)

    def test_different_structure(self):
        """FGWAlign should return > 0 for structurally different graphs."""
        from ged_lib.solvers.fgwalign import FGWAlignSolver

        g1 = nx.Graph()
        g1.add_node(0)
        g1.add_node(1)
        g1.add_edge(0, 1)

        g2 = nx.Graph()
        g2.add_node(0)
        g2.add_node(1)
        # No edge in g2

        solver = FGWAlignSolver(patience=1, topk=1)
        result = solver.compute(g1, g2)
        assert result > 0.0

    def test_labeled_aids_deterministic(self):
        """FGWAlignSolver with AIDS label map should be deterministic at patience=1."""
        from ged_lib.solvers.fgwalign import FGWAlignSolver

        g1 = nx.Graph()
        g1.add_node(0, label="C")
        g1.add_node(1, label="N")
        g1.add_edge(0, 1)

        g2 = nx.Graph()
        g2.add_node(0, label="C")
        g2.add_node(1, label="O")
        g2.add_node(2, label="N")
        g2.add_edge(0, 1)
        g2.add_edge(1, 2)

        solver = FGWAlignSolver(dataset_name="AIDS", patience=1, topk=1)
        r1 = solver.compute(g1, g2)
        r2 = solver.compute(g1, g2)
        assert r1 == r2, "Single-trial result should be fully deterministic"

    def test_solver_registered(self):
        """FGWAlignSolver must be registered under the key 'fgwalign'."""
        from ged_lib import list_solvers

        assert "fgwalign" in list_solvers()
