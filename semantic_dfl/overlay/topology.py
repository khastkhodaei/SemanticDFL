"""
Physical Network Topology Generators and Neighbor Graph Modeling for SemanticDFL.

Provides realistic physical/underlay peer graph generators G = (V, E) reflecting
edge, WAN, and P2P communication constraints:
- Ring / Regular Lattice
- Watts-Strogatz Small-World Network
- Random Geometric Graph (Spatial / Edge Proximity)
- 2D Torus / Grid Topology
- SLICES WAN Multi-Region Testbed Topology
"""

from __future__ import annotations

import collections
import math
import random
from typing import Dict, List, Optional, Set, Tuple
import networkx as nx
import numpy as np


def generate_ring_topology(num_nodes: int, degree: int = 4) -> Dict[int, List[int]]:
    """
    Generates a regular ring lattice topology where each node connects to `degree` nearest neighbors.
    
    Args:
        num_nodes: Total number of peers in the network.
        degree: Even integer indicating the degree of each node (number of physical neighbors).
        
    Returns:
        Adjacency dict mapping node_id -> sorted list of neighbor node_ids.
    """
    if num_nodes <= 1:
        return {0: []} if num_nodes == 1 else {}
    
    effective_degree = min(degree if degree % 2 == 0 else degree + 1, num_nodes - 1)
    k_half = max(1, effective_degree // 2)
    
    adj: Dict[int, List[int]] = {}
    for i in range(num_nodes):
        neighbors = set()
        for offset in range(1, k_half + 1):
            neighbors.add((i + offset) % num_nodes)
            neighbors.add((i - offset) % num_nodes)
        neighbors.discard(i)
        adj[i] = sorted(list(neighbors))
    return adj


def generate_small_world_topology(
    num_nodes: int,
    k_neighbors: int = 4,
    p_rewire: float = 0.15,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """
    Generates a Watts-Strogatz small-world graph topology capturing realistic P2P clustering.
    
    Args:
        num_nodes: Total number of peers.
        k_neighbors: Base degree (must be even).
        p_rewire: Rewiring probability in [0, 1].
        seed: Random seed for reproducibility.
        
    Returns:
        Adjacency dict mapping node_id -> sorted list of neighbor node_ids.
    """
    if num_nodes <= 1:
        return {0: []} if num_nodes == 1 else {}
    
    effective_k = min(k_neighbors if k_neighbors % 2 == 0 else k_neighbors + 1, num_nodes - 1)
    if effective_k < 2:
        effective_k = 2

    # Use networkx with deterministic seed
    g = nx.watts_strogatz_graph(n=num_nodes, k=effective_k, p=p_rewire, seed=seed)
    
    # Ensure connectivity: if disconnected, add bridge edges
    if not nx.is_connected(g):
        components = list(nx.connected_components(g))
        for i in range(len(components) - 1):
            u = random.choice(list(components[i]))
            v = random.choice(list(components[i + 1]))
            g.add_edge(u, v)

    adj: Dict[int, List[int]] = {n: sorted(list(g.neighbors(n))) for n in range(num_nodes)}
    return adj


def generate_random_geometric_topology(
    num_nodes: int,
    radius: float = 0.35,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """
    Generates a Random Geometric Graph (RGG) reflecting Euclidean edge/IoT device proximity.
    
    Args:
        num_nodes: Total number of peers.
        radius: Distance threshold for physical connectivity.
        seed: Random seed.
        
    Returns:
        Adjacency dict mapping node_id -> sorted list of neighbor node_ids.
    """
    if num_nodes <= 1:
        return {0: []} if num_nodes == 1 else {}

    g = nx.random_geometric_graph(n=num_nodes, radius=radius, seed=seed)
    
    # Ensure full connectivity
    if not nx.is_connected(g):
        components = list(nx.connected_components(g))
        for i in range(len(components) - 1):
            u = random.choice(list(components[i]))
            v = random.choice(list(components[i + 1]))
            g.add_edge(u, v)

    adj: Dict[int, List[int]] = {n: sorted(list(g.neighbors(n))) for n in range(num_nodes)}
    return adj


def generate_grid_topology(num_nodes: int) -> Dict[int, List[int]]:
    """
    Generates a 2D Grid / Torus topology.
    """
    if num_nodes <= 1:
        return {0: []} if num_nodes == 1 else {}

    side = int(math.ceil(math.sqrt(num_nodes)))
    adj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}

    for i in range(num_nodes):
        r = i // side
        c = i % side
        neighbors = set()

        # 4-connected grid neighbors
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        for nr, nc in candidates:
            if 0 <= nr and 0 <= nc:
                nid = nr * side + nc
                if nid < num_nodes and nid != i:
                    neighbors.add(nid)

        # Fallback to ensure min degree >= 2
        if len(neighbors) < 2:
            neighbors.add((i + 1) % num_nodes)
            neighbors.add((i - 1) % num_nodes)

        neighbors.discard(i)
        adj[i] = sorted(list(neighbors))

    return adj


def generate_slices_wan_topology(
    num_nodes: int,
    num_regions: int = 3,
    intra_region_prob: float = 0.7,
    inter_region_bridges: int = 2,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """
    Simulates the EU SLICES multi-site testbed WAN topology.
    Nodes are grouped into geographic sites/regions with dense intra-region links
    and bounded inter-region gateway bridges.
    
    Args:
        num_nodes: Total number of nodes.
        num_regions: Number of distinct SLICES sites (e.g., Paris, Madrid, Bologna).
        intra_region_prob: Probability of intra-site physical link.
        inter_region_bridges: Number of cross-site gateway links per site pair.
        seed: Random seed.
        
    Returns:
        Adjacency dict mapping node_id -> sorted list of neighbor node_ids.
    """
    if num_nodes <= 1:
        return {0: []} if num_nodes == 1 else {}

    rng = random.Random(seed)
    regions: List[List[int]] = [[] for _ in range(num_regions)]
    for i in range(num_nodes):
        regions[i % num_regions].append(i)

    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))

    # Intra-region connectivity (Erdos-Renyi within each region)
    for reg in regions:
        if len(reg) <= 1:
            continue
        reg_graph = nx.erdos_renyi_graph(len(reg), p=intra_region_prob, seed=seed)
        for u, v in reg_graph.edges():
            g.add_edge(reg[u], reg[v])

        # Ensure intra-region connectivity via simple ring
        for idx in range(len(reg)):
            g.add_edge(reg[idx], reg[(idx + 1) % len(reg)])

    # Inter-region gateway links
    for r1 in range(num_regions):
        for r2 in range(r1 + 1, num_regions):
            reg1 = regions[r1]
            reg2 = regions[r2]
            if not reg1 or not reg2:
                continue
            for _ in range(inter_region_bridges):
                u = rng.choice(reg1)
                v = rng.choice(reg2)
                g.add_edge(u, v)

    adj: Dict[int, List[int]] = {n: sorted(list(g.neighbors(n))) for n in range(num_nodes)}
    return adj


def build_network_topology(
    topology_type: str = "ring",
    num_nodes: int = 10,
    degree: int = 4,
    seed: int = 42,
    **kwargs,
) -> Dict[int, List[int]]:
    """
    Factory constructor for physical underlay network topologies.
    
    Supported types: 'ring', 'small_world', 'watts_strogatz', 'geometric', 'rgg', 'grid', 'slices_wan', 'slices'.
    """
    top = topology_type.strip().lower()
    if top in ("ring", "lattice"):
        return generate_ring_topology(num_nodes=num_nodes, degree=degree)
    elif top in ("small_world", "watts_strogatz", "ws"):
        p_rewire = kwargs.get("p_rewire", 0.15)
        return generate_small_world_topology(num_nodes=num_nodes, k_neighbors=degree, p_rewire=p_rewire, seed=seed)
    elif top in ("geometric", "rgg", "random_geometric"):
        radius = kwargs.get("radius", 0.35)
        return generate_random_geometric_topology(num_nodes=num_nodes, radius=radius, seed=seed)
    elif top in ("grid", "torus", "mesh"):
        return generate_grid_topology(num_nodes=num_nodes)
    elif top in ("slices_wan", "slices", "wan"):
        num_regions = kwargs.get("num_regions", 3)
        return generate_slices_wan_topology(num_nodes=num_nodes, num_regions=num_regions, seed=seed)
    elif top in ("fully_connected", "all_to_all", "clique"):
        return {i: [j for j in range(num_nodes) if j != i] for i in range(num_nodes)}
    else:
        # Default to regular ring
        return generate_ring_topology(num_nodes=num_nodes, degree=degree)


def compute_r_hop_neighborhood(
    adjacency_graph: Dict[int, List[int]],
    node_id: int,
    radius: int = 2,
) -> Set[int]:
    """
    Computes the set of all nodes reachable within `radius` hops of `node_id`.
    """
    visited: Set[int] = {node_id}
    queue = collections.deque([(node_id, 0)])

    while queue:
        curr, dist = queue.popleft()
        if dist < radius:
            for neighbor in adjacency_graph.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return visited


def compute_average_degree(adjacency_graph: Dict[int, List[int]]) -> float:
    """Computes the average node degree in the graph."""
    if not adjacency_graph:
        return 0.0
    total_edges = sum(len(neighbors) for neighbors in adjacency_graph.values())
    return total_edges / float(len(adjacency_graph))
