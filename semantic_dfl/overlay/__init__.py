"""
Semantic Overlay Network (SON), Clustering, Topology, and Multi-Hop Routing.
"""

from semantic_dfl.overlay.clustering import ZoneClustering
from semantic_dfl.overlay.descriptors import (
    ClusterDescriptor,
    PeerNodeMetadata,
    PeerRole,
    ZoneDescriptor,
)
from semantic_dfl.overlay.routing import MultiHopRouter
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
from semantic_dfl.overlay.zone import ZoneManager

__all__ = [
    "PeerRole",
    "ClusterDescriptor",
    "ZoneDescriptor",
    "PeerNodeMetadata",
    "ZoneClustering",
    "ZoneManager",
    "SemanticOverlayNetwork",
    "MultiHopRouter",
    "build_network_topology",
    "generate_ring_topology",
    "generate_small_world_topology",
    "generate_random_geometric_topology",
    "generate_grid_topology",
    "generate_slices_wan_topology",
    "compute_r_hop_neighborhood",
    "compute_average_degree",
]
