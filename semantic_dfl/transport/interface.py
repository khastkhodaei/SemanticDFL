"""
Abstract Transport Interfaces and Peer Descriptors for SemanticDFL.

Decouples decentralized learning algorithms (zone clustering, similarity search,
pull aggregation) from concrete network communication layers (In-Memory, gRPC, REST).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Sequence
import torch


@dataclass
class PeerEndpoint:
    """Network connection coordinates and metadata for a peer node."""
    node_id: int
    hostname: str = "127.0.0.1"
    port: int = 50051
    role: str = "peer"               # "peer", "super_peer", "bootstrap"
    zone_id: str = ""
    level: int = 1
    is_super_peer: bool = False
    replica_super_peers: List[int] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)

    @property
    def target_address(self) -> str:
        """Returns standard host:port string."""
        return f"{self.hostname}:{self.port}"


class ITransport(ABC):
    """
    Abstract communication contract for decentralized federated learning operations.
    """

    @abstractmethod
    def start(self) -> None:
        """Initializes network listeners, servers, and connection pools."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Gracefully tears down open sockets, channels, and background threads."""
        pass

    @abstractmethod
    def register_peer(self, endpoint: PeerEndpoint) -> bool:
        """Registers a remote or local peer into the routing registry."""
        pass

    @abstractmethod
    def heartbeat(self, peer: PeerEndpoint, round_num: int = 0, current_loss: float = 0.0) -> bool:
        """Sends liveness ping and state synchronization probe to peer."""
        pass

    @abstractmethod
    def discover_peers(
        self,
        query_node_id: int,
        zone_id: str = "",
        radius_ttl: int = 2,
    ) -> List[PeerEndpoint]:
        """Discovers reachable peers within proximity / zone constraints."""
        pass

    @abstractmethod
    def send_signature(
        self,
        target_peer: PeerEndpoint,
        signature: torch.Tensor,
        round_num: int,
        cardinality_p: int = 0,
    ) -> bool:
        """Transmits coordinate-wise Top-P signature vector to a target peer."""
        pass

    @abstractmethod
    def fetch_signature(
        self,
        target_peer: PeerEndpoint,
        round_num: int,
    ) -> Optional[torch.Tensor]:
        """Retrieves Top-P signature tensor from target peer."""
        pass

    @abstractmethod
    def pull_model_weights(
        self,
        target_peer: PeerEndpoint,
        round_num: int = 0,
    ) -> Optional[torch.Tensor]:
        """Pulls full 1D parameter vector from target peer for personalized aggregation."""
        pass

    @abstractmethod
    def broadcast_topology(
        self,
        notification: Dict[str, Any],
        targets: Sequence[PeerEndpoint],
    ) -> None:
        """Disseminates zone cluster descriptors or super-peer election notices."""
        pass

    @abstractmethod
    def set_local_weights(self, weights: torch.Tensor, round_num: int) -> None:
        """Stores local weights buffer for remote pulling."""
        pass

    @abstractmethod
    def set_local_signature(self, signature: torch.Tensor, round_num: int, cardinality_p: int) -> None:
        """Stores local signature buffer for remote queries."""
        pass
