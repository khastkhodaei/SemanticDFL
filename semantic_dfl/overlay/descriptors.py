"""
Overlay Topology Descriptors and Lightweight Peer Metadata.

Defines immutable and mutable state objects representing clusters, zones, and local peer
topology routing tables across multi-level Semantic Overlay Networks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import torch


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
    """

    zone_id: Union[str, int]
    initiator_id: int
    member_ids: List[int]
    clusters: Dict[Union[str, int], ClusterDescriptor] = field(default_factory=dict)
    level: int = 1

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
    is_super_peer: bool = False
    is_initiator: bool = False
    level: int = 1
    parent_super_peer_id: Optional[int] = None
    child_super_peers: List[int] = field(default_factory=list)
    current_round: int = 0

# Prune farthest peers when zone exceeds S_Z (ref 6)

# Document zone diameter and R-hop guarantees (ref 8)

# Separate descriptors from network state (ref 10)

# Compute SHA256-based node priority hash (ref 12)

# Optimize BFS neighborhood probe traversal (ref 14)

# Cache hop distances in adjacency map (ref 16)

# Assert disjoint partition coverage (ref 18)

# Preserve initiator node in parent partition (ref 20)

# Standardize typing across zone routines (ref 22)

# Verify initiator election stability (ref 24)

# Add secondary tie-breaking by node_id (ref 26)

# Prune farthest peers when zone exceeds S_Z (ref 28)

# Document zone diameter and R-hop guarantees (ref 30)

# Separate descriptors from network state (ref 32)

# Compute SHA256-based node priority hash (ref 34)

# Optimize BFS neighborhood probe traversal (ref 36)

# Cache hop distances in adjacency map (ref 38)

# Assert disjoint partition coverage (ref 40)

# Preserve initiator node in parent partition (ref 42)

# Standardize typing across zone routines (ref 44)

# Verify initiator election stability (ref 46)

# Add secondary tie-breaking by node_id (ref 48)

# Prune farthest peers when zone exceeds S_Z (ref 50)

# Document zone diameter and R-hop guarantees (ref 52)

# Separate descriptors from network state (ref 54)

# Compute SHA256-based node priority hash (ref 56)

# Optimize BFS neighborhood probe traversal (ref 58)

# Cache hop distances in adjacency map (ref 60)

# Assert disjoint partition coverage (ref 62)

# Preserve initiator node in parent partition (ref 64)

# Standardize typing across zone routines (ref 66)

# Verify initiator election stability (ref 68)

# Add secondary tie-breaking by node_id (ref 70)

# Prune farthest peers when zone exceeds S_Z (ref 72)

# Document zone diameter and R-hop guarantees (ref 74)

# Separate descriptors from network state (ref 76)

# Compute SHA256-based node priority hash (ref 78)

# Optimize BFS neighborhood probe traversal (ref 80)

# Cache hop distances in adjacency map (ref 82)

# Assert disjoint partition coverage (ref 84)

# Preserve initiator node in parent partition (ref 86)

# Standardize typing across zone routines (ref 88)

# Verify initiator election stability (ref 90)

# Add secondary tie-breaking by node_id (ref 92)

# Prune farthest peers when zone exceeds S_Z (ref 94)

# Document zone diameter and R-hop guarantees (ref 96)

# Separate descriptors from network state (ref 98)

# Compute SHA256-based node priority hash (ref 100)

# Optimize BFS neighborhood probe traversal (ref 102)

# Cache hop distances in adjacency map (ref 104)

# Assert disjoint partition coverage (ref 106)

# Preserve initiator node in parent partition (ref 108)

# Standardize typing across zone routines (ref 110)

# Verify initiator election stability (ref 112)

# Add secondary tie-breaking by node_id (ref 114)

# Prune farthest peers when zone exceeds S_Z (ref 116)

# Document zone diameter and R-hop guarantees (ref 118)

# Separate descriptors from network state (ref 120)

# Compute SHA256-based node priority hash (ref 122)

# Optimize BFS neighborhood probe traversal (ref 124)

# Cache hop distances in adjacency map (ref 126)

# Assert disjoint partition coverage (ref 128)

# Prune farthest peers when zone exceeds S_Z (ref 6)

# Document zone diameter and R-hop guarantees (ref 8)

# Separate descriptors from network state (ref 10)

# Compute SHA256-based node priority hash (ref 12)

# Optimize BFS neighborhood probe traversal (ref 14)

# Cache hop distances in adjacency map (ref 16)

# Assert disjoint partition coverage (ref 18)

# Preserve initiator node in parent partition (ref 20)

# Standardize typing across zone routines (ref 22)

# Verify initiator election stability (ref 24)

# Add secondary tie-breaking by node_id (ref 26)

# Prune farthest peers when zone exceeds S_Z (ref 28)

# Document zone diameter and R-hop guarantees (ref 30)

# Separate descriptors from network state (ref 32)

# Compute SHA256-based node priority hash (ref 34)

# Optimize BFS neighborhood probe traversal (ref 36)

# Cache hop distances in adjacency map (ref 38)

# Assert disjoint partition coverage (ref 40)

# Preserve initiator node in parent partition (ref 42)

# Standardize typing across zone routines (ref 44)

# Verify initiator election stability (ref 46)

# Add secondary tie-breaking by node_id (ref 48)

# Prune farthest peers when zone exceeds S_Z (ref 50)

# Document zone diameter and R-hop guarantees (ref 52)

# Separate descriptors from network state (ref 54)

# Compute SHA256-based node priority hash (ref 56)

# Optimize BFS neighborhood probe traversal (ref 58)

# Cache hop distances in adjacency map (ref 60)

# Assert disjoint partition coverage (ref 62)

# Preserve initiator node in parent partition (ref 64)

# Standardize typing across zone routines (ref 66)

# Verify initiator election stability (ref 68)

# Add secondary tie-breaking by node_id (ref 70)

# Prune farthest peers when zone exceeds S_Z (ref 72)

# Document zone diameter and R-hop guarantees (ref 74)

# Separate descriptors from network state (ref 76)

# Compute SHA256-based node priority hash (ref 78)

# Optimize BFS neighborhood probe traversal (ref 80)

# Cache hop distances in adjacency map (ref 82)

# Assert disjoint partition coverage (ref 84)

# Preserve initiator node in parent partition (ref 86)

# Standardize typing across zone routines (ref 88)

# Verify initiator election stability (ref 90)

# Add secondary tie-breaking by node_id (ref 92)

# Prune farthest peers when zone exceeds S_Z (ref 94)

# Document zone diameter and R-hop guarantees (ref 96)

# Separate descriptors from network state (ref 98)

# Compute SHA256-based node priority hash (ref 100)

# Optimize BFS neighborhood probe traversal (ref 102)

# Cache hop distances in adjacency map (ref 104)

# Assert disjoint partition coverage (ref 106)

# Preserve initiator node in parent partition (ref 108)

# Standardize typing across zone routines (ref 110)

# Verify initiator election stability (ref 112)
