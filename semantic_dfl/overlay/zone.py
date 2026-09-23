"""
Proximity-bounded Overlay Zone Formation, VRF Initiator Election, and Splitting.

Implements Algorithm 2 (SelectInitiators) and Algorithm 3 (ZoneFormationProtocol)
from Section 3.3.1 and Appendix B.2 of the SemanticDFL paper:
- VRF-based initiator election within R-hop physical neighborhoods
- First-wins multi-source TTL-bounded probing with inter-zone boundary discovery
- Deterministic zone capacity enforcement and splitting under S_Z cap
"""

from __future__ import annotations

import collections
import hashlib
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np

from semantic_dfl.overlay.descriptors import ZoneDescriptor
from semantic_dfl.overlay.topology import compute_r_hop_neighborhood


def compute_vrf_score(node_id: int, seed: int) -> int:
    """
    Computes a verifiable pseudorandom score sigma_p = VRF_p(seed_{t-1}).
    
    Uses deterministic SHA-256 HMAC representation yielding a uniform 64-bit integer.
    """
    data = f"VRF:{node_id}:{seed}".encode("utf-8")
    return int(hashlib.sha256(data).hexdigest()[:16], 16)


class ZoneManager:
    """
    Coordinates distributed, proximity-aligned zone formation and size-bounded splitting (Algorithms 2 & 3).
    """

    def __init__(
        self,
        max_zone_size: int = 20,
        radius_ttl: int = 2,
        seed: int = 42,
    ) -> None:
        if max_zone_size < 2:
            raise ValueError(f"max_zone_size must be >= 2, got {max_zone_size}")
        if radius_ttl < 1:
            raise ValueError(f"radius_ttl must be >= 1, got {radius_ttl}")

        self.max_zone_size = max_zone_size
        self.radius_ttl = radius_ttl
        self.seed = seed

    def elect_initiators(
        self,
        node_ids: Sequence[int],
        adj: Dict[int, List[int]],
        epoch_seed: Optional[int] = None,
    ) -> List[int]:
        """
        Algorithm 2: SelectInitiators(P, R)
        
        Elects zone initiators based on verifiable random scores (sigma_p, id_p)
        over the R-hop physical neighborhood of each node.
        
        Args:
            node_ids: Sequence of active peer node IDs.
            adj: Physical adjacency graph.
            epoch_seed: Optional epoch seed (defaults to self.seed).
            
        Returns:
            List of elected initiator node IDs.
        """
        s = self.seed if epoch_seed is None else epoch_seed
        node_set = set(node_ids)
        vrf_scores = {u: compute_vrf_score(u, s) for u in node_ids}
        initiators: List[int] = []

        for u in node_ids:
            # Collect R-hop physical neighborhood N_R(u)
            r_neighbors = compute_r_hop_neighborhood(adj, u, radius=self.radius_ttl)
            r_neighbors_in_set = [v for v in r_neighbors if v in node_set]

            # u is marked as initiator if (sigma_u, u) is minimal in N_R(u)
            min_score_in_neighborhood = min((vrf_scores[v], v) for v in r_neighbors_in_set)
            if (vrf_scores[u], u) == min_score_in_neighborhood:
                initiators.append(u)

        # Invariant: Ensure at least ceil(N / S_Z) initiators exist to avoid oversized zones
        min_required = max(1, math.ceil(len(node_ids) / self.max_zone_size))
        if len(initiators) < min_required:
            sorted_nodes = sorted(node_ids, key=lambda n: (vrf_scores[n], n))
            for candidate in sorted_nodes:
                if candidate not in initiators:
                    initiators.append(candidate)
                if len(initiators) >= min_required:
                    break

        return sorted(initiators)

    def form_zones(
        self,
        node_ids: Sequence[int],
        adjacency_graph: Optional[Dict[int, List[int]]] = None,
        level: int = 1,
        epoch_seed: Optional[int] = None,
    ) -> Dict[Union[str, int], ZoneDescriptor]:
        """
        Algorithm 3: ZoneFormationProtocol(V, R, S_Z)
        
        Executes proximity-bounded zone formation over the set of nodes.
        
        Args:
            node_ids: List of peer node IDs.
            adjacency_graph: Adjacency list mapping node_id -> list of neighbor node_ids.
            level: Current hierarchical level (1 for leaf physical peers).
            epoch_seed: Epoch seed for VRF leader election.
            
        Returns:
            Dictionary mapping zone_id -> ZoneDescriptor.
        """
        if not node_ids:
            return {}

        n = len(node_ids)
        node_set = set(node_ids)
        s = self.seed if epoch_seed is None else epoch_seed

        # Build default adjacency graph if not provided (4-regular ring)
        if adjacency_graph is None:
            adj: Dict[int, List[int]] = {}
            for i, u in enumerate(node_ids):
                adj[u] = [
                    node_ids[(i - 2) % n],
                    node_ids[(i - 1) % n],
                    node_ids[(i + 1) % n],
                    node_ids[(i + 2) % n],
                ]
        else:
            adj = {u: [v for v in adjacency_graph.get(u, []) if v in node_set] for u in node_ids}

        # Step 1: Elect Initiators via Algorithm 2
        initiators = self.elect_initiators(node_ids, adj, epoch_seed=s)

        # Step 2: Multi-source TTL-bounded PROBE flood (First-Wins)
        # Node state: assigned initiator or None
        assigned_initiator: Dict[int, int] = {}
        zone_members: Dict[int, List[int]] = {init: [init] for init in initiators}
        zone_adjacencies: Dict[int, Set[int]] = {init: set() for init in initiators}

        for init in initiators:
            assigned_initiator[init] = init

        # Queue contains (current_node, initiator_id, remaining_ttl, sender)
        queue: collections.deque[Tuple[int, int, int, int]] = collections.deque()
        for init in initiators:
            queue.append((init, init, self.radius_ttl, init))

        while queue:
            curr, init_id, ttl, sender = queue.popleft()
            if ttl <= 0:
                continue

            for neighbor in adj.get(curr, []):
                if neighbor == sender:
                    continue

                if neighbor not in assigned_initiator:
                    # Handler: NOT_ASSIGNED -> First-wins assignment (Line 10-16)
                    assigned_initiator[neighbor] = init_id
                    zone_members[init_id].append(neighbor)
                    if ttl - 1 > 0:
                        queue.append((neighbor, init_id, ttl - 1, curr))
                else:
                    # Handler: Already ASSIGNED -> Record zone boundary adjacency (Line 17-20)
                    existing_init = assigned_initiator[neighbor]
                    if existing_init != init_id:
                        zone_adjacencies[init_id].add(existing_init)
                        zone_adjacencies[existing_init].add(init_id)

        # Step 3: Handle any unreached / orphan nodes (Line 22)
        vrf_scores = {u: compute_vrf_score(u, s) for u in node_ids}
        unassigned = [u for u in node_ids if u not in assigned_initiator]
        for orphan in unassigned:
            # Place in nearest physical initiator with capacity
            placed = False
            for init in initiators:
                if len(zone_members[init]) < self.max_zone_size:
                    zone_members[init].append(orphan)
                    assigned_initiator[orphan] = init
                    placed = True
                    break
            if not placed:
                # Seed new zone with orphan
                zone_members[orphan] = [orphan]
                zone_adjacencies[orphan] = set()
                assigned_initiator[orphan] = orphan
                initiators.append(orphan)

        # Step 4: Deterministic SplitZone for oversized zones (|Z_I| > S_Z) (Line 23-27)
        zones: Dict[Union[str, int], ZoneDescriptor] = {}
        zone_counter = 0

        for init, members in zone_members.items():
            if not members:
                continue

            if len(members) > self.max_zone_size:
                split_descriptors = self.split_zone(
                    members,
                    initiator_id=init,
                    level=level,
                    start_id=zone_counter,
                    epoch_seed=s,
                )
                for zd in split_descriptors:
                    zones[zd.zone_id] = zd
                    zone_counter += 1
            else:
                zid = f"L{level}_Z{zone_counter}" if level > 1 else zone_counter
                adj_zones = {f"L{level}_Z{o}" if level > 1 else o for o in zone_adjacencies.get(init, set())}
                zones[zid] = ZoneDescriptor(
                    zone_id=zid,
                    initiator_id=init,
                    member_ids=sorted(members),
                    level=level,
                    adjacent_zone_ids=adj_zones,
                )
                zone_counter += 1

        return zones

    def split_zone(
        self,
        member_ids: Sequence[int],
        initiator_id: int,
        level: int = 1,
        start_id: int = 0,
        epoch_seed: Optional[int] = None,
    ) -> List[ZoneDescriptor]:
        """
        SplitZone Protocol (Algorithm 3 Line 25):
        Delegates overflow disjoint member subsets to the next lowest VRF score in-zone.
        """
        s = self.seed if epoch_seed is None else epoch_seed
        num_splits = math.ceil(len(member_ids) / self.max_zone_size)
        splits = np.array_split(np.asarray(sorted(member_ids)), num_splits)

        result: List[ZoneDescriptor] = []
        for i, split_arr in enumerate(splits):
            members = sorted(split_arr.tolist())
            if initiator_id in members:
                init = initiator_id
            else:
                init = min(members, key=lambda n: (compute_vrf_score(n, s), n))

            zid = f"L{level}_Z{start_id + i}" if level > 1 else start_id + i
            result.append(
                ZoneDescriptor(
                    zone_id=zid,
                    initiator_id=init,
                    member_ids=members,
                    level=level,
                )
            )

        return result
