"""
Unit and Integration Tests for Physical Network Topology, Explicit Peer Roles,
VRF Leader Election, Freshness Guards, and Traffic Accounting.
"""

import pytest
import torch

from semantic_dfl.aggregation.pull_aggregator import PullAggregator, compute_alpha_from_psi
from semantic_dfl.overlay.descriptors import PeerRole
from semantic_dfl.overlay.son import SemanticOverlayNetwork
from semantic_dfl.overlay.topology import (
    build_network_topology,
    compute_average_degree,
    compute_r_hop_neighborhood,
    generate_grid_topology,
    generate_random_geometric_topology,
    generate_ring_topology,
    generate_slices_wan_topology,
    generate_small_world_topology,
)
from semantic_dfl.overlay.zone import ZoneManager, compute_vrf_score
from semantic_dfl.search.engine import SimilaritySearchEngine


class TestTopologyAndNeighborConstraints:
    def test_ring_topology(self):
        adj = generate_ring_topology(num_nodes=10, degree=4)
        assert len(adj) == 10
        for node_id, neighbors in adj.items():
            assert len(neighbors) == 4
            assert node_id not in neighbors

    def test_small_world_topology(self):
        adj = generate_small_world_topology(num_nodes=20, k_neighbors=4, p_rewire=0.1, seed=42)
        assert len(adj) == 20
        avg_deg = compute_average_degree(adj)
        assert avg_deg >= 3.5

    def test_random_geometric_topology(self):
        adj = generate_random_geometric_topology(num_nodes=15, radius=0.4, seed=42)
        assert len(adj) == 15
        for node_id, neighbors in adj.items():
            assert node_id not in neighbors

    def test_grid_topology(self):
        adj = generate_grid_topology(num_nodes=16)
        assert len(adj) == 16
        for node_id, neighbors in adj.items():
            assert len(neighbors) >= 2
            assert node_id not in neighbors

    def test_slices_wan_topology(self):
        adj = generate_slices_wan_topology(num_nodes=18, num_regions=3, seed=42)
        assert len(adj) == 18
        avg_deg = compute_average_degree(adj)
        assert avg_deg > 2.0

    def test_r_hop_neighborhood(self):
        adj = generate_ring_topology(num_nodes=10, degree=2)  # Single line/ring
        r1 = compute_r_hop_neighborhood(adj, node_id=0, radius=1)
        assert 0 in r1
        assert 1 in r1
        assert 9 in r1
        assert len(r1) == 3

        r2 = compute_r_hop_neighborhood(adj, node_id=0, radius=2)
        assert len(r2) == 5
        assert 2 in r2
        assert 8 in r2


class TestVRFElectionAndRoles:
    def test_vrf_score_determinism(self):
        s1 = compute_vrf_score(node_id=5, seed=100)
        s2 = compute_vrf_score(node_id=5, seed=100)
        s3 = compute_vrf_score(node_id=5, seed=101)
        assert s1 == s2
        assert s1 != s3

    def test_zone_initiator_election(self):
        adj = generate_ring_topology(num_nodes=20, degree=4)
        zm = ZoneManager(max_zone_size=5, radius_ttl=2, seed=42)
        initiators = zm.elect_initiators(node_ids=list(range(20)), adj=adj, epoch_seed=1)
        assert len(initiators) >= 4  # ceil(20 / 5) = 4
        for init in initiators:
            assert 0 <= init < 20

    def test_son_roles_assignment(self):
        torch.manual_seed(42)
        signatures = {i: torch.randn(100) for i in range(20)}
        for sig in signatures.values():
            sig[20:] = 0.0  # Sparsify

        adj = generate_ring_topology(num_nodes=20, degree=4)
        zm = ZoneManager(max_zone_size=5, radius_ttl=2, seed=42)
        son = SemanticOverlayNetwork(zone_manager=zm)
        node_meta = son.build_topology(
            signatures=signatures,
            p=20,
            adjacency_graph=adj,
            max_levels=2,
        )

        role_counts = son.get_role_distribution()
        assert role_counts[PeerRole.LEAF_PEER.value] == 20
        assert role_counts[PeerRole.ZONE_INITIATOR.value] >= 4
        assert role_counts[PeerRole.SUPER_PEER.value] >= 4
        assert role_counts[PeerRole.ROOT_SUPER_PEER.value] >= 1

        for nid, meta in node_meta.items():
            assert PeerRole.LEAF_PEER in meta.roles
            assert len(meta.physical_neighbors) == 4


class TestPullAggregatorFormulasAndFreshness:
    def test_alpha_from_psi(self):
        # When psi=2 and eta=1, alpha = 2 / 3 = 0.6667
        alpha = compute_alpha_from_psi(psi=2.0, eta=1.0)
        assert abs(alpha - (2.0 / 3.0)) < 1e-4

        # When psi=0, pure local learning
        alpha_zero = compute_alpha_from_psi(psi=0.0, eta=1.0)
        assert alpha_zero == 0.0

    def test_freshness_guard(self):
        agg = PullAggregator(t_fresh=0.8, max_staleness=2)
        u_query = torch.zeros(10)
        u_query[:5] = 1.0

        # Fresh signature (similarity = 1.0)
        u_fresh = u_query.clone()
        # Stale signature that has low similarity
        u_drifted = torch.zeros(10)
        u_drifted[5:] = 1.0

        cand_sigs = {1: u_fresh, 2: u_drifted, 3: u_fresh}
        cand_rounds = {1: 5, 2: 4, 3: 1}  # peer 1 is round 5 (curr), peer 2 is round 4 (staleness 1), peer 3 is round 1 (staleness 4 > 2)

        accepted = agg.filter_fresh_replies(
            query_signature=u_query,
            candidate_signatures=cand_sigs,
            candidate_rounds=cand_rounds,
            current_round=5,
        )
        assert 1 in accepted  # Same round
        assert 2 not in accepted  # Drifted (similarity low)
        assert 3 not in accepted  # Exceeds max_staleness


class TestSearchEngineTrafficAndStats:
    def test_search_traffic_accounting(self):
        torch.manual_seed(42)
        signatures = {i: torch.randn(50) for i in range(15)}
        zm = ZoneManager(max_zone_size=5, radius_ttl=2, seed=42)
        son = SemanticOverlayNetwork(zone_manager=zm)
        son.build_topology(signatures=signatures, p=10)

        engine = SimilaritySearchEngine(son=son, beam_cap=3)
        candidates, stats = engine.search_peers(
            query_node_id=0,
            query_signature=signatures[0],
            all_signatures=signatures,
            k=3,
            tau=0.2,
            return_stats=True,
        )

        assert len(candidates) <= 3
        assert stats["scored_edges"] > 0
        assert stats["control_bytes"] > 0
        assert 0.0 <= stats["acceptance_rate"] <= 1.0
