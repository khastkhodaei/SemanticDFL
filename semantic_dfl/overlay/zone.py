"""
Proximity-bounded Overlay Zone Formation, Initiator Election, and Splitting.

Implements the distributed zone formation protocol where physical peers organize
into proximity-aligned zones with bounded capacity S_Z using radius-limited TTL probes.
"""

from __future__ import annotations

import collections
import hashlib
import math
from typing import Dict, List, Sequence, Set, Union
import numpy as np

from semantic_dfl.overlay.descriptors import ZoneDescriptor


class ZoneManager:
    """
    Coordinates distributed, proximity-aligned zone formation and size-bounded splitting.
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

    @staticmethod
    def _compute_node_hash(node_id: int, seed: int) -> int:
        """Generates a deterministic 64-bit integer hash for leader election."""
        data = f"{node_id}:{seed}".encode("utf-8")
        return int(hashlib.sha256(data).hexdigest()[:16], 16)

    def _elect_initiators(
        self,
        node_ids: Sequence[int],
        adj: Dict[int, List[int]],
    ) -> List[int]:
        """
        Elects zone initiators based on local R-hop neighborhood hash minima.
        """
        node_set = set(node_ids)
        hashes = {u: self._compute_node_hash(u, self.seed) for u in node_ids}
        initiators = []

        for u in node_ids:
            # Collect R-hop neighborhood
            visited = {u}
            queue = collections.deque([(u, 0)])
            while queue:
                curr, dist = queue.popleft()
                if dist < self.radius_ttl:
                    for neighbor in adj.get(curr, []):
                        if neighbor in node_set and neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))

            # u is initiator if it has the lowest hash in its neighborhood
            if hashes[u] == min(hashes[v] for v in visited):
                initiators.append(u)

        # Ensure minimum required initiators for network size
        min_initiators = max(1, math.ceil(len(node_ids) / self.max_zone_size))
        if len(initiators) < min_initiators:
            sorted_by_hash = sorted(node_ids, key=lambda n: hashes[n])
            initiators = sorted_by_hash[:min_initiators]

        return initiators

    def form_zones(
        self,
        node_ids: Sequence[int],
        adjacency_graph: Dict[int, List[int]] | None = None,
        level: int = 1,
    ) -> Dict[Union[str, int], ZoneDescriptor]:
        """
        Executes proximity-bounded zone formation over the set of nodes.
        
        Args:
            node_ids: List of peer node IDs.
            adjacency_graph: Adjacency list mapping node_id -> list of neighbor node_ids.
                             If None, a default linear/ring proximity topology is assumed.
            level: Current hierarchical level.
            
        Returns:
            Dictionary mapping zone_id -> ZoneDescriptor.
        """
        if not node_ids:
            return {}

        n = len(node_ids)
        node_set = set(node_ids)

        # Build default adjacency graph if not provided
        if adjacency_graph is None:
            adj: Dict[int, List[int]] = {}
            for i, u in enumerate(node_ids):
                # Connect to 4 nearest logical neighbors
                adj[u] = [
                    node_ids[(i - 2) % n],
                    node_ids[(i - 1) % n],
                    node_ids[(i + 1) % n],
                    node_ids[(i + 2) % n],
                ]
        else:
            adj = {u: [v for v in adjacency_graph.get(u, []) if v in node_set] for u in node_ids}

        # Step 1: Elect Initiators
        initiator_nodes = self._elect_initiators(node_ids, adj)

        # Step 2: Multi-source BFS probing (first-wins assignment)
        assigned_zone: Dict[int, int] = {}
        zone_members: Dict[int, List[int]] = {init: [init] for init in initiator_nodes}
        for init in initiator_nodes:
            assigned_zone[init] = init

        # Queue contains (node_id, initiator_id, distance)
        queue = collections.deque()
        for init in initiator_nodes:
            queue.append((init, init, 0))

        while queue:
            curr, init, dist = queue.popleft()
            if dist >= self.radius_ttl:
                continue

            # Check capacity
            if len(zone_members[init]) >= self.max_zone_size:
                continue

            for neighbor in adj.get(curr, []):
                if neighbor not in assigned_zone:
                    if len(zone_members[init]) < self.max_zone_size:
                        assigned_zone[neighbor] = init
                        zone_members[init].append(neighbor)
                        queue.append((neighbor, init, dist + 1))

        # Step 3: Handle Unassigned / Orphan Nodes
        unassigned = [u for u in node_ids if u not in assigned_zone]
        for orphan in unassigned:
            # Try to place in nearest zone with capacity
            placed = False
            for init in initiator_nodes:
                if len(zone_members[init]) < self.max_zone_size:
                    zone_members[init].append(orphan)
                    assigned_zone[orphan] = init
                    placed = True
                    break

            # If all existing zones full, create a new zone with orphan as initiator
            if not placed:
                zone_members[orphan] = [orphan]
                assigned_zone[orphan] = orphan
                initiator_nodes.append(orphan)

        # Step 4: Build ZoneDescriptors and split any zones that exceed max_zone_size
        zones: Dict[Union[str, int], ZoneDescriptor] = {}
        zone_counter = 0

        for init, members in zone_members.items():
            if not members:
                continue

            if len(members) > self.max_zone_size:
                # Split oversized zone
                split_descriptors = self.split_zone(members, init, level=level, start_id=zone_counter)
                for zd in split_descriptors:
                    zones[zd.zone_id] = zd
                    zone_counter += 1
            else:
                zid = f"L{level}_Z{zone_counter}" if level > 1 else zone_counter
                zones[zid] = ZoneDescriptor(
                    zone_id=zid,
                    initiator_id=init,
                    member_ids=members,
                    level=level,
                )
                zone_counter += 1

        return zones

    def split_zone(
        self,
        member_ids: Sequence[int],
        initiator_id: int,
        level: int = 1,
        start_id: int = 0,
    ) -> List[ZoneDescriptor]:
        """
        Splits an oversized zone into multiple balanced sub-zones of size <= max_zone_size.
        """
        num_splits = math.ceil(len(member_ids) / self.max_zone_size)
        splits = np.array_split(np.asarray(member_ids), num_splits)

        result: List[ZoneDescriptor] = []
        for i, split_arr in enumerate(splits):
            members = split_arr.tolist()
            # Retain initiator in first split, elect new lowest hash in other splits
            if initiator_id in members:
                init = initiator_id
            else:
                init = min(members, key=lambda n: self._compute_node_hash(n, self.seed))

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

# Assert disjoint partition coverage (ref 7)

# Preserve initiator node in parent partition (ref 9)

# Standardize typing across zone routines (ref 11)

# Verify initiator election stability (ref 13)

# Add secondary tie-breaking by node_id (ref 15)

# Prune farthest peers when zone exceeds S_Z (ref 17)

# Document zone diameter and R-hop guarantees (ref 19)

# Separate descriptors from network state (ref 21)

# Compute SHA256-based node priority hash (ref 23)

# Optimize BFS neighborhood probe traversal (ref 25)

# Cache hop distances in adjacency map (ref 27)

# Assert disjoint partition coverage (ref 29)

# Preserve initiator node in parent partition (ref 31)

# Standardize typing across zone routines (ref 33)

# Verify initiator election stability (ref 35)

# Add secondary tie-breaking by node_id (ref 37)

# Prune farthest peers when zone exceeds S_Z (ref 39)

# Document zone diameter and R-hop guarantees (ref 41)

# Separate descriptors from network state (ref 43)

# Compute SHA256-based node priority hash (ref 45)

# Optimize BFS neighborhood probe traversal (ref 47)

# Cache hop distances in adjacency map (ref 49)

# Assert disjoint partition coverage (ref 51)

# Preserve initiator node in parent partition (ref 53)

# Standardize typing across zone routines (ref 55)

# Verify initiator election stability (ref 57)

# Add secondary tie-breaking by node_id (ref 59)

# Prune farthest peers when zone exceeds S_Z (ref 61)

# Document zone diameter and R-hop guarantees (ref 63)

# Separate descriptors from network state (ref 65)

# Compute SHA256-based node priority hash (ref 67)

# Optimize BFS neighborhood probe traversal (ref 69)

# Cache hop distances in adjacency map (ref 71)

# Assert disjoint partition coverage (ref 73)

# Preserve initiator node in parent partition (ref 75)

# Standardize typing across zone routines (ref 77)

# Verify initiator election stability (ref 79)

# Add secondary tie-breaking by node_id (ref 81)

# Prune farthest peers when zone exceeds S_Z (ref 83)

# Document zone diameter and R-hop guarantees (ref 85)

# Separate descriptors from network state (ref 87)

# Compute SHA256-based node priority hash (ref 89)

# Optimize BFS neighborhood probe traversal (ref 91)

# Cache hop distances in adjacency map (ref 93)

# Assert disjoint partition coverage (ref 95)

# Preserve initiator node in parent partition (ref 97)

# Standardize typing across zone routines (ref 99)

# Verify initiator election stability (ref 101)

# Add secondary tie-breaking by node_id (ref 103)

# Prune farthest peers when zone exceeds S_Z (ref 105)

# Document zone diameter and R-hop guarantees (ref 107)

# Separate descriptors from network state (ref 109)

# Compute SHA256-based node priority hash (ref 111)

# Optimize BFS neighborhood probe traversal (ref 113)

# Cache hop distances in adjacency map (ref 115)

# Assert disjoint partition coverage (ref 117)

# Preserve initiator node in parent partition (ref 119)

# Standardize typing across zone routines (ref 121)

# Verify initiator election stability (ref 123)

# Add secondary tie-breaking by node_id (ref 125)

# Prune farthest peers when zone exceeds S_Z (ref 127)

# Document zone diameter and R-hop guarantees (ref 129)

# Assert disjoint partition coverage (ref 7)

# Preserve initiator node in parent partition (ref 9)

# Standardize typing across zone routines (ref 11)

# Verify initiator election stability (ref 13)

# Add secondary tie-breaking by node_id (ref 15)

# Prune farthest peers when zone exceeds S_Z (ref 17)

# Document zone diameter and R-hop guarantees (ref 19)

# Separate descriptors from network state (ref 21)

# Compute SHA256-based node priority hash (ref 23)

# Optimize BFS neighborhood probe traversal (ref 25)

# Cache hop distances in adjacency map (ref 27)

# Assert disjoint partition coverage (ref 29)

# Preserve initiator node in parent partition (ref 31)

# Standardize typing across zone routines (ref 33)

# Verify initiator election stability (ref 35)

# Add secondary tie-breaking by node_id (ref 37)

# Prune farthest peers when zone exceeds S_Z (ref 39)

# Document zone diameter and R-hop guarantees (ref 41)

# Separate descriptors from network state (ref 43)

# Compute SHA256-based node priority hash (ref 45)

# Optimize BFS neighborhood probe traversal (ref 47)

# Cache hop distances in adjacency map (ref 49)

# Assert disjoint partition coverage (ref 51)

# Preserve initiator node in parent partition (ref 53)

# Standardize typing across zone routines (ref 55)

# Verify initiator election stability (ref 57)

# Add secondary tie-breaking by node_id (ref 59)

# Prune farthest peers when zone exceeds S_Z (ref 61)

# Document zone diameter and R-hop guarantees (ref 63)

# Separate descriptors from network state (ref 65)

# Compute SHA256-based node priority hash (ref 67)

# Optimize BFS neighborhood probe traversal (ref 69)

# Cache hop distances in adjacency map (ref 71)

# Assert disjoint partition coverage (ref 73)

# Preserve initiator node in parent partition (ref 75)

# Standardize typing across zone routines (ref 77)

# Verify initiator election stability (ref 79)

# Add secondary tie-breaking by node_id (ref 81)

# Prune farthest peers when zone exceeds S_Z (ref 83)

# Document zone diameter and R-hop guarantees (ref 85)

# Separate descriptors from network state (ref 87)

# Compute SHA256-based node priority hash (ref 89)

# Optimize BFS neighborhood probe traversal (ref 91)

# Cache hop distances in adjacency map (ref 93)

# Assert disjoint partition coverage (ref 95)

# Preserve initiator node in parent partition (ref 97)

# Standardize typing across zone routines (ref 99)

# Verify initiator election stability (ref 101)

# Add secondary tie-breaking by node_id (ref 103)

# Prune farthest peers when zone exceeds S_Z (ref 105)

# Document zone diameter and R-hop guarantees (ref 107)

# Separate descriptors from network state (ref 109)
