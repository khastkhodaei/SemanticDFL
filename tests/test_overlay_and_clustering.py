"""
Unit Tests for Semantic Overlay Network (SON), Zone Formation, and AP Clustering.
"""

import pytest
import torch
import numpy as np

from semantic_dfl.overlay.descriptors import (
    ClusterDescriptor,
    ZoneDescriptor,
    PeerNodeMetadata,
)
from semantic_dfl.overlay.clustering import ZoneClustering
from semantic_dfl.overlay.zone import ZoneManager
from semantic_dfl.overlay.son import SemanticOverlayNetwork


class TestZoneManager:
    """Tests for proximity zone formation and capacity bounds."""

    @pytest.mark.parametrize("num_nodes, max_zone_size", [(20, 5), (50, 12), (100, 20)])
    def test_zone_formation_bounds_and_coverage(self, num_nodes, max_zone_size):
        node_ids = list(range(num_nodes))
        zm = ZoneManager(max_zone_size=max_zone_size, radius_ttl=2, seed=42)
        zones = zm.form_zones(node_ids)

        assert len(zones) >= 1

        all_assigned = []
        for zid, zone in zones.items():
            # Zone capacity bound check
            assert len(zone.member_ids) <= max_zone_size
            assert zone.initiator_id in zone.member_ids
            all_assigned.extend(zone.member_ids)

        # Complete coverage & disjoint partition check
        assert len(all_assigned) == num_nodes
        assert sorted(all_assigned) == node_ids

    def test_deterministic_zone_splitting(self):
        zm = ZoneManager(max_zone_size=5, seed=42)
        members = list(range(14))
        splits = zm.split_zone(members, initiator_id=0)

        # 14 elements with max_size 5 => ceil(14/5) = 3 zones
        assert len(splits) == 3
        total_split_members = []
        for s in splits:
            assert len(s.member_ids) <= 5
            total_split_members.extend(s.member_ids)

        assert sorted(total_split_members) == members


class TestZoneClustering:
    """Tests for Affinity Propagation clustering and prototype extraction."""

    def test_cluster_zone_signatures(self):
        torch.manual_seed(42)
        p = 10
        m_dim = 100
        
        # Create 3 distinct clusters of signatures
        cluster1_base = torch.randn(m_dim)
        cluster2_base = torch.randn(m_dim)
        
        signatures = {}
        # Nodes 0..4 belong to cluster 1
        for i in range(5):
            signatures[i] = cluster1_base + 0.05 * torch.randn(m_dim)
        # Nodes 5..9 belong to cluster 2
        for i in range(5, 10):
            signatures[i] = cluster2_base + 0.05 * torch.randn(m_dim)

        zc = ZoneClustering(damping=0.7, preference_quantile=0.5, r_replicas=3)
        member_ids = list(range(10))
        clusters = zc.cluster_zone_signatures(member_ids, signatures, p=p, zone_id=1)

        assert len(clusters) >= 2

        all_clustered_members = []
        for cid, c_desc in clusters.items():
            assert c_desc.exemplar_id in c_desc.member_ids
            assert c_desc.replica_super_peers[0] == c_desc.exemplar_id
            assert len(c_desc.replica_super_peers) <= 3
            
            # Prototype signature must be sparsified to top-P
            nnz = torch.count_nonzero(c_desc.prototype_signature).item()
            assert nnz <= p

            all_clustered_members.extend(c_desc.member_ids)

        assert sorted(all_clustered_members) == member_ids


class TestSemanticOverlayNetwork:
    """Tests for hierarchical multi-level SON tree and failover promotion."""

    @pytest.mark.parametrize("num_nodes", [25, 60])
    def test_son_hierarchy_construction(self, num_nodes):
        torch.manual_seed(42)
        p = 8
        m_dim = 64
        signatures = {i: torch.randn(m_dim) for i in range(num_nodes)}

        zm = ZoneManager(max_zone_size=10, radius_ttl=2, seed=42)
        zc = ZoneClustering(damping=0.7, r_replicas=3)
        son = SemanticOverlayNetwork(zone_manager=zm, clustering_engine=zc)

        metadata = son.build_topology(signatures, p=p, max_levels=3)
        assert len(metadata) == num_nodes

        # Verify all nodes have assigned cluster and super-peer
        for nid, meta in metadata.items():
            assert meta.zone_id is not None
            assert meta.assigned_cluster_id is not None
            assert meta.super_peer_id is not None
            assert len(meta.replica_super_peers) >= 1

        # Check multi-level zone creation
        assert 1 in son.levels
        assert len(son.clusters) > 0

    def test_super_peer_failover_promotion(self):
        torch.manual_seed(42)
        p = 5
        m_dim = 30
        signatures = {i: torch.randn(m_dim) for i in range(8)}

        zm = ZoneManager(max_zone_size=10, seed=42)
        zc = ZoneClustering(damping=0.7, r_replicas=3)
        son = SemanticOverlayNetwork(zone_manager=zm, clustering_engine=zc)

        metadata = son.build_topology(signatures, p=p, max_levels=1)
        
        # Pick the first cluster
        first_cid = next(iter(son.clusters.keys()))
        cluster = son.clusters[first_cid]

        original_exemplar = cluster.exemplar_id
        assert len(cluster.replica_super_peers) >= 2
        expected_new_exemplar = cluster.replica_super_peers[1]

        # Trigger failover
        new_exemplar = son.handle_super_peer_failure(
            failed_peer_id=original_exemplar,
            cluster_id=first_cid,
        )

        assert new_exemplar == expected_new_exemplar
        assert cluster.exemplar_id == expected_new_exemplar
        assert cluster.replica_super_peers[0] == expected_new_exemplar

        # Verify updated metadata for members
        for mid in cluster.member_ids:
            meta = son.get_peer_metadata(mid)
            assert meta.super_peer_id == expected_new_exemplar
            if mid == expected_new_exemplar:
                assert meta.is_super_peer is True
            elif mid == original_exemplar:
                assert meta.is_super_peer is False
