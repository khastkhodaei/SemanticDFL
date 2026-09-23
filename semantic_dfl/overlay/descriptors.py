"""
Overlay Topology Descriptors and Lightweight Peer Metadata.

Defines formal peer roles, cluster descriptors, zone descriptors, and local peer
topology routing tables across multi-level Semantic Overlay Networks (SON).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Union
import torch


class PeerRole(str, Enum):
    """
    Formal network roles in the SemanticDFL hierarchy.
    """
    LEAF_PEER = "LEAF_PEER"                      # Regular peer participating in local SGD and model pull
    ZONE_INITIATOR = "ZONE_INITIATOR"            # Elected leader of a zone via VRF min; conducts AP clustering
    SUPER_PEER = "SUPER_PEER"                    # Cluster exemplar routing queries and maintaining cluster prototype
    REPLICA_SUPER_PEER = "REPLICA_SUPER_PEER"    # Backup super-peer replica (size r) for failover resilience
    ROOT_SUPER_PEER = "ROOT_SUPER_PEER"          # Top-level super-peer at level H (ingress for multi-hop queries)


@dataclass
class ClusterDescriptor:
    """
    Descriptor representing an intra-zone or inter-zone semantic cluster.
    
    Attributes:
        cluster_id: Unique identifier for this cluster.
        exemplar_id: Primary representative node (super-peer) for this cluster.
        prototype_signature: Sparsified centroid vector u_bar in R^M.
        replica_super_peers: Ordered list of failover replica super-peers R_j (starts with exemplar).
        member_ids: Participating peer IDs belonging to this cluster.
        zone_id: Identifier of the parent zone containing this cluster.
        level: Hierarchical SON tree level (1 for leaf peer clusters, >1 for meta-clusters).
        parent_cluster_id: Reference to higher-level cluster if part of hierarchical SON.
    """

    cluster_id: Union[str, int]
    exemplar_id: int
    prototype_signature: torch.Tensor
    replica_super_peers: List[int]
    member_ids: List[int]
    zone_id: Union[str, int]
    level: int = 1
    parent_cluster_id: Optional[Union[str, int]] = None

    @property
    def size(self) -> int:
        """Number of member peers in the cluster."""
        return len(self.member_ids)

    def promote_replica(self) -> int:
        """
        Promotes the next available replica to primary exemplar when the current
        exemplar fails or disconnects.
        
        Returns:
            The new exemplar peer ID.
        """
        if len(self.replica_super_peers) > 1:
            self.replica_super_peers.pop(0)
            self.exemplar_id = self.replica_super_peers[0]
        return self.exemplar_id


@dataclass
class ZoneDescriptor:
    """
    Descriptor representing a proximity-bounded overlay zone.
    
    Attributes:
        zone_id: Unique identifier for this zone.
        initiator_id: Elected zone initiator node coordinating zone membership & clustering.
        member_ids: List of peer node IDs in this zone.
        clusters: Dictionary mapping cluster_id to ClusterDescriptor for this zone.
        level: Hierarchical level (1 for physical node zones, >1 for super-peer zones).
        adjacent_zone_ids: Neighboring zone IDs discovered during probe boundary encounters.
    """

    zone_id: Union[str, int]
    initiator_id: int
    member_ids: List[int]
    clusters: Dict[Union[str, int], ClusterDescriptor] = field(default_factory=dict)
    level: int = 1
    adjacent_zone_ids: Set[Union[str, int]] = field(default_factory=set)

    @property
    def size(self) -> int:
        """Number of member peers in the zone."""
        return len(self.member_ids)


@dataclass
class PeerNodeMetadata:
    """
    Lightweight local routing table and overlay metadata maintained by an individual node.
    
    Ensures O(log N) state overhead: nodes store only their local zone, assigned super-peer,
    replica super-peers, and hierarchical parent/child pointers without requiring global state.
    """

    node_id: int
    zone_id: Optional[Union[str, int]] = None
    assigned_cluster_id: Optional[Union[str, int]] = None
    super_peer_id: Optional[int] = None
    replica_super_peers: List[int] = field(default_factory=list)
    roles: Set[PeerRole] = field(default_factory=lambda: {PeerRole.LEAF_PEER})
    level: int = 1
    parent_super_peer_id: Optional[int] = None
    child_super_peers: List[int] = field(default_factory=list)
    current_round: int = 0
    physical_neighbors: List[int] = field(default_factory=list)

    @property
    def is_super_peer(self) -> bool:
        return PeerRole.SUPER_PEER in self.roles or PeerRole.ROOT_SUPER_PEER in self.roles

    @property
    def is_initiator(self) -> bool:
        return PeerRole.ZONE_INITIATOR in self.roles

    @property
    def is_replica(self) -> bool:
        return PeerRole.REPLICA_SUPER_PEER in self.roles
