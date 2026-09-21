"""
Transport Factory and Dynamic Routing Provider for SemanticDFL.

Provides centralized instantiation for network transport adapters:
- InMemoryTransport / MemoryTransport: Zero-socket in-process broker for simulations.
- GRPCTransport: Socket-based distributed transport over TCP/IP channels.
"""

from __future__ import annotations

from typing import Any, Optional

from semantic_dfl.transport.grpc_transport import GRPCTransport
from semantic_dfl.transport.interface import ITransport
from semantic_dfl.transport.memory_transport import MemoryTransport

# Canonical alias for simulation memory transport
InMemoryTransport = MemoryTransport


def create_transport(
    mode: str,
    node_id: int,
    **kwargs: Any,
) -> ITransport:
    """
    Factory creating concrete transport instances based on execution mode.

    Args:
        mode: Execution mode ('simulation', 'memory', 'distributed', 'grpc').
        node_id: Integer identifier of the local node.
        **kwargs: Additional mode-specific configuration parameters:
            - host (str): Host binding address for distributed mode (default '127.0.0.1').
            - base_port (int): Base port for dynamic port calculation (default 50050).
            - port (int, optional): Explicit port overriding base_port + node_id.
            - max_workers (int): Thread pool size for gRPC server (default 10).
            - timeout_seconds (float): RPC timeout in seconds (default 15.0).
            - simulated_latency_ms (float): In-memory artificial latency (default 0.0).

    Returns:
        ITransport: Initialized concrete transport adapter.

    Raises:
        ValueError: If mode is not recognized.
    """
    normalized_mode = mode.strip().lower().replace("-", "_")

    if normalized_mode in ("simulation", "memory", "in_memory"):
        simulated_latency = float(kwargs.get("simulated_latency_ms", 0.0))
        return MemoryTransport(
            node_id=node_id,
            simulated_latency_ms=simulated_latency,
        )

    if normalized_mode in ("distributed", "grpc", "slices_distributed", "slices"):
        host = str(kwargs.get("host", "127.0.0.1"))
        base_port = int(kwargs.get("base_port", 50050))
        explicit_port = kwargs.get("port")
        port = int(explicit_port) if explicit_port is not None else (base_port + node_id)
        max_workers = int(kwargs.get("max_workers", 10))
        timeout_seconds = float(kwargs.get("timeout_seconds", 15.0))

        return GRPCTransport(
            node_id=node_id,
            host=host,
            port=port,
            max_workers=max_workers,
            timeout_seconds=timeout_seconds,
        )

    raise ValueError(
        f"Unsupported execution/transport mode: '{mode}'. "
        f"Supported options are: 'simulation' (or 'memory') and 'distributed' (or 'grpc')."
    )
