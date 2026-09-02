"""
In-Memory Concrete Transport Adapter for Local Simulations.

Provides fast, zero-socket message routing adhering to ITransport, enabling seamless
simulation benchmarking and rapid offline prototyping.
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any, Dict, List, Optional, Sequence
import torch

from semantic_dfl.transport.interface import ITransport, PeerEndpoint

logger = logging.getLogger("SemanticDFL.Transport.Memory")


class MemoryTransport(ITransport):
    """
    High-performance in-memory transport broker for centralized simulation runs.
    """

    # Global shared state across nodes in the same Python process
    _global_registry: Dict[int, PeerEndpoint] = {}
    _global_signatures: Dict[int, Dict[int, torch.Tensor]] = {}   # node_id -> {round: sig}
    _global_weights: Dict[int, Dict[int, torch.Tensor]] = {}      # node_id -> {round: weights}
    _global_topology: Dict[str, Any] = {}

    def __init__(self, node_id: int, simulated_latency_ms: float = 0.0) -> None:
        self.node_id = node_id
        self.simulated_latency_ms = simulated_latency_ms
        self.is_running = False
        self._local_endpoint: Optional[PeerEndpoint] = None

    def start(self) -> None:
        self.is_running = True
        self._local_endpoint = PeerEndpoint(node_id=self.node_id, role="peer")
        MemoryTransport._global_registry[self.node_id] = self._local_endpoint

    def stop(self) -> None:
        self.is_running = False
        if self.node_id in MemoryTransport._global_registry:
            del MemoryTransport._global_registry[self.node_id]

    def register_peer(self, endpoint: PeerEndpoint) -> bool:
        MemoryTransport._global_registry[endpoint.node_id] = endpoint
        return True

    def heartbeat(self, peer: PeerEndpoint, round_num: int = 0, current_loss: float = 0.0) -> bool:
        if peer.node_id in MemoryTransport._global_registry:
            MemoryTransport._global_registry[peer.node_id].last_seen = time.time()
            return True
        return False

    def discover_peers(
        self,
        query_node_id: int,
        zone_id: str = "",
        radius_ttl: int = 2,
    ) -> List[PeerEndpoint]:
        results = []
        for nid, ep in MemoryTransport._global_registry.items():
            if nid != query_node_id:
                if not zone_id or ep.zone_id == zone_id:
                    results.append(copy.deepcopy(ep))
        return results

    def send_signature(
        self,
        target_peer: PeerEndpoint,
        signature: torch.Tensor,
        round_num: int,
        cardinality_p: int = 0,
    ) -> bool:
        if self.node_id not in MemoryTransport._global_signatures:
            MemoryTransport._global_signatures[self.node_id] = {}
        MemoryTransport._global_signatures[self.node_id][round_num] = signature.detach().clone().cpu()
        return True

    def fetch_signature(
        self,
        target_peer: PeerEndpoint,
        round_num: int,
    ) -> Optional[torch.Tensor]:
        if target_peer.node_id in MemoryTransport._global_signatures:
            rounds_map = MemoryTransport._global_signatures[target_peer.node_id]
            if round_num in rounds_map:
                return rounds_map[round_num].clone()
            # If current round not yet deposited, return latest available
            if rounds_map:
                latest_r = max(rounds_map.keys())
                return rounds_map[latest_r].clone()
        return None

    def pull_model_weights(
        self,
        target_peer: PeerEndpoint,
        round_num: int = 0,
    ) -> Optional[torch.Tensor]:
        if target_peer.node_id in MemoryTransport._global_weights:
            rounds_map = MemoryTransport._global_weights[target_peer.node_id]
            if round_num in rounds_map:
                return rounds_map[round_num].clone()
            if rounds_map:
                latest_r = max(rounds_map.keys())
                return rounds_map[latest_r].clone()
        return None

    def broadcast_topology(
        self,
        notification: Dict[str, Any],
        targets: Sequence[PeerEndpoint],
    ) -> None:
        MemoryTransport._global_topology.update(notification)

    def set_local_weights(self, weights: torch.Tensor, round_num: int) -> None:
        if self.node_id not in MemoryTransport._global_weights:
            MemoryTransport._global_weights[self.node_id] = {}
        MemoryTransport._global_weights[self.node_id][round_num] = weights.detach().clone().cpu()

    def set_local_signature(self, signature: torch.Tensor, round_num: int, cardinality_p: int) -> None:
        if self.node_id not in MemoryTransport._global_signatures:
            MemoryTransport._global_signatures[self.node_id] = {}
        MemoryTransport._global_signatures[self.node_id][round_num] = signature.detach().clone().cpu()

    @classmethod
    def reset_global_state(cls) -> None:
        """Clears in-memory shared state for clean test isolation."""
        cls._global_registry.clear()
        cls._global_signatures.clear()
        cls._global_weights.clear()
        cls._global_topology.clear()
