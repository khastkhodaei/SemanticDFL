"""
Hierarchical Semantic Overlay Network (SON) Multi-Level Tree and Failover Coordinator.

Constructs and manages the hierarchical multi-level semantic overlay tree (from leaf peer
clusters up to root super-peers), maintains explicit network roles (Initiator, Super-Peer,
Replica, Leaf), and handles replica-based churn / failover promotions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Union
import torch

from semantic_dfl.overlay.descriptors import (
    ClusterDescriptor,
    PeerNodeMetadata,
    PeerRole,
    ZoneDescriptor,
)
from semantic_dfl.overlay.clustering import ZoneClustering
from semantic_dfl.overlay.zone import ZoneManager


class SemanticOverlayNetwork:
    """
    Hierarchical Semantic Overlay Network (SON) tree topology builder and routing coordinator.
    """

    def __init__(
        self,
        zone_manager: Optional[ZoneManager] = None,
        clustering_engine: Optional[ZoneClustering] = None,
    ) -> None:
        self.zone_manager = zone_manager or ZoneManager()
        self.clustering_engine = clustering_engine or ZoneClustering()

        # Structural State
        self.levels: Dict[int, Dict[Union[str, int], ZoneDescriptor]] = {}
        self.clusters: Dict[Union[str, int], ClusterDescriptor] = {}
        self.node_metadata: Dict[int, PeerNodeMetadata] = {}

    def build_topology(
        self,
        signatures: Dict[int, torch.Tensor],
        p: int,
        adjacency_graph: Optional[Dict[int, List[int]]] = None,
        max_levels: int = 3,
        r_replicas: int = 3,
        epoch_seed: Optional[int] = None,
    ) -> Dict[int, PeerNodeMetadata]:
        """
        Constructs the multi-level hierarchical SON topology.
        
        Args:
            signatures: Mapping from node_id -> 1D signature tensor u_i in R^M.
            p: Target signature cardinality for prototypes.
            adjacency_graph: Optional proximity adjacency list for Level 1 nodes.
            max_levels: Maximum height H of the hierarchical super-peer tree.
            r_replicas: Number of failover replica super-peers per cluster.
            epoch_seed: Epoch seed for VRF leader election.
            
        Returns:
            Dictionary mapping node_id -> local PeerNodeMetadata.
        """
        self.levels.clear()
        self.clusters.clear()
        self.node_metadata.clear()

        all_node_ids = sorted(signatures.keys())
        if not all_node_ids:
            return {}

        # Initialize base metadata for all nodes with LEAF_PEER role
        for nid in all_node_ids:
            phys_nbrs = adjacency_graph.get(nid, []) if adjacency_graph is not None else []
            self.node_metadata[nid] = PeerNodeMetadata(
                node_id=nid,
                level=1,
                roles={PeerRole.LEAF_PEER},
                physical_neighbors=list(phys_nbrs),
            )

        current_nodes = all_node_ids
        current_signatures = signatures
        current_adj = adjacency_graph

        for level in range(1, max_levels + 1):
            if not current_nodes:
                break

            # Step 1: Form proximity zones at current level
            zones = self.zone_manager.form_zones(
                node_ids=current_nodes,
                adjacency_graph=current_adj,
                level=level,
                epoch_seed=epoch_seed,
            )
            self.levels[level] = zones

            super_peers_next_level: List[int] = []
            signatures_next_level: Dict[int, torch.Tensor] = {}

            # Step 2: Cluster each zone independently via Affinity Propagation
            for zone_id, zone_desc in zones.items():
                clusters = self.clustering_engine.cluster_zone_signatures(
                    member_ids=zone_desc.member_ids,
                    signatures=current_signatures,
                    p=p,
                    zone_id=zone_id,
                    level=level,
                )
                zone_desc.clusters = clusters

                # Mark zone initiator role
                init_id = zone_desc.initiator_id
                if init_id in self.node_metadata:
                    self.node_metadata[init_id].roles.add(PeerRole.ZONE_INITIATOR)

                # Update metadata for members and register clusters
                for cid, cluster_desc in clusters.items():
                    self.clusters[cid] = cluster_desc
                    exemplar = cluster_desc.exemplar_id

                    if exemplar not in super_peers_next_level:
                        super_peers_next_level.append(exemplar)
                        signatures_next_level[exemplar] = cluster_desc.prototype_signature

                    for member_id in cluster_desc.member_ids:
                        meta = self.node_metadata[member_id]
                        if level == 1:
                            meta.zone_id = zone_id
                            meta.assigned_cluster_id = cid
                            meta.super_peer_id = exemplar
                            meta.replica_super_peers = list(cluster_desc.replica_super_peers)

                        # Assign SUPER_PEER / REPLICA roles
                        if member_id == exemplar:
                            meta.roles.add(PeerRole.SUPER_PEER)
                        elif member_id in cluster_desc.replica_super_peers:
                            meta.roles.add(PeerRole.REPLICA_SUPER_PEER)

            # If top level reached or only 1 super-peer remains, designate ROOT_SUPER_PEER
            is_top_level = (len(super_peers_next_level) <= 1 or level == max_levels)
            if is_top_level:
                for sp_id in super_peers_next_level:
                    if sp_id in self.node_metadata:
                        self.node_metadata[sp_id].roles.add(PeerRole.ROOT_SUPER_PEER)
                break

            # Link parent-child pointers for next level
            current_nodes = super_peers_next_level
            current_signatures = signatures_next_level
            current_adj = None  # Higher levels use full logical connectivity

        return self.node_metadata

    def handle_super_peer_failure(
        self,
        failed_peer_id: int,
        cluster_id: Union[str, int],
    ) -> int:
        """
        Handles churn / departure of a primary super-peer by promoting the next replica in R_j.
        
        Args:
            failed_peer_id: ID of the disconnected/failed super-peer.
            cluster_id: ID of the affected cluster descriptor.
            
        Returns:
            The newly promoted active exemplar/super-peer ID.
        """
        if cluster_id not in self.clusters:
            raise KeyError(f"Cluster '{cluster_id}' not found in overlay.")

        cluster = self.clusters[cluster_id]
        
        # Promote next replica in R_j
        new_exemplar = cluster.promote_replica()

        # Update metadata for all cluster members
        for mid in cluster.member_ids:
            if mid in self.node_metadata:
                meta = self.node_metadata[mid]
                meta.super_peer_id = new_exemplar
                meta.replica_super_peers = list(cluster.replica_super_peers)
                if mid == new_exemplar:
                    meta.roles.add(PeerRole.SUPER_PEER)
                elif mid == failed_peer_id:
                    meta.roles.discard(PeerRole.SUPER_PEER)
                    meta.roles.discard(PeerRole.ROOT_SUPER_PEER)

        return new_exemplar

    def get_cluster_descriptor(
        self, cluster_id: Union[str, int]
    ) -> Optional[ClusterDescriptor]:
        """Retrieves a cluster descriptor by ID."""
        return self.clusters.get(cluster_id)

    def get_peer_metadata(self, node_id: int) -> Optional[PeerNodeMetadata]:
        """Retrieves local metadata for a peer."""
        return self.node_metadata.get(node_id)

    def get_role_distribution(self) -> Dict[str, int]:
        """Returns the count of nodes holding each role."""
        counts = {role.value: 0 for role in PeerRole}
        for meta in self.node_metadata.values():
            for role in meta.roles:
                counts[role.value] += 1
        return counts
