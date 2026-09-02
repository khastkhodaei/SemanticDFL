"""
SemanticDFL Network Transport Layer.

Exports abstract and concrete transport mechanisms:
- MemoryTransport: High-performance in-memory simulation broker.
- GRPCTransport: Production-grade distributed gRPC socket transport.
- TensorSerializer: Fast PyTorch tensor protobuf encoder/decoder.
- PeerEndpoint: Node network coordinate descriptor.
"""

from semantic_dfl.transport.interface import ITransport, PeerEndpoint
from semantic_dfl.transport.memory_transport import MemoryTransport
from semantic_dfl.transport.grpc_transport import (
    DFLGrpcServicer,
    GRPCTransport,
    TensorSerializer,
)

__all__ = [
    "ITransport",
    "PeerEndpoint",
    "MemoryTransport",
    "GRPCTransport",
    "DFLGrpcServicer",
    "TensorSerializer",
]
