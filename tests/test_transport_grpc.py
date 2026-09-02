"""
Integration and Unit Tests for gRPC and Memory Transport Layers.
"""

import time
import pytest
import torch
import numpy as np

from semantic_dfl.transport.interface import PeerEndpoint
from semantic_dfl.transport.memory_transport import MemoryTransport
from semantic_dfl.transport.grpc_transport import (
    GRPCTransport,
    TensorSerializer,
)


class TestTensorSerializer:
    """Tests for zero-copy PyTorch tensor serialization and reconstruction."""

    def test_float32_1d_tensor_serialization(self):
        tensor = torch.randn(500, dtype=torch.float32)
        proto = TensorSerializer.tensor_to_proto(tensor)
        reconstructed = TensorSerializer.proto_to_tensor(proto)

        assert reconstructed.shape == tensor.shape
        assert reconstructed.dtype == tensor.dtype
        assert torch.allclose(reconstructed, tensor)

    def test_multidimensional_tensor_serialization(self):
        tensor = torch.randint(0, 100, (32, 64, 3, 3), dtype=torch.int64)
        proto = TensorSerializer.tensor_to_proto(tensor)
        reconstructed = TensorSerializer.proto_to_tensor(proto)

        assert reconstructed.shape == tensor.shape
        assert reconstructed.dtype == torch.int64
        assert torch.equal(reconstructed, tensor)

    def test_sparse_masked_signature_tensor(self):
        tensor = torch.zeros(1000, dtype=torch.float32)
        tensor[::10] = torch.randn(100)
        proto = TensorSerializer.tensor_to_proto(tensor)
        reconstructed = TensorSerializer.proto_to_tensor(proto)

        assert torch.count_nonzero(reconstructed) == 100
        assert torch.allclose(reconstructed, tensor)


class TestMemoryTransport:
    """Tests for in-memory simulation transport broker."""

    def setup_method(self):
        MemoryTransport.reset_global_state()

    def test_peer_registration_and_discovery(self):
        t0 = MemoryTransport(node_id=0)
        t1 = MemoryTransport(node_id=1)
        t0.start()
        t1.start()

        ep0 = PeerEndpoint(node_id=0, role="bootstrap")
        ep1 = PeerEndpoint(node_id=1, role="peer")
        t0.register_peer(ep0)
        t1.register_peer(ep1)

        peers_from_0 = t0.discover_peers(query_node_id=0)
        assert len(peers_from_0) == 1
        assert peers_from_0[0].node_id == 1

        t0.stop()
        t1.stop()

    def test_signature_and_weight_exchange(self):
        t0 = MemoryTransport(node_id=0)
        t1 = MemoryTransport(node_id=1)
        t0.start()
        t1.start()

        sig0 = torch.tensor([1.0, 2.0, 3.0])
        weights0 = torch.tensor([10.0, 20.0, 30.0])

        t0.set_local_signature(sig0, round_num=1, cardinality_p=3)
        t0.set_local_weights(weights0, round_num=1)

        ep0 = PeerEndpoint(node_id=0)
        fetched_sig = t1.fetch_signature(ep0, round_num=1)
        pulled_w = t1.pull_model_weights(ep0, round_num=1)

        assert fetched_sig is not None
        assert torch.allclose(fetched_sig, sig0)
        assert pulled_w is not None
        assert torch.allclose(pulled_w, weights0)

        t0.stop()
        t1.stop()


class TestGRPCTransportIntegration:
    """Live integration tests using real TCP socket communication."""

    def test_two_node_grpc_rpc_exchange(self):
        # Pick ephemeral test ports
        port_a = 50251
        port_b = 50252

        transport_a = GRPCTransport(node_id=10, host="127.0.0.1", port=port_a, timeout_seconds=5.0)
        transport_b = GRPCTransport(node_id=20, host="127.0.0.1", port=port_b, timeout_seconds=5.0)

        transport_a.start()
        transport_b.start()
        time.sleep(0.2)

        try:
            # 1. Node B registers with Node A
            ep_a = PeerEndpoint(node_id=10, hostname="127.0.0.1", port=port_a)
            ep_b = PeerEndpoint(node_id=20, hostname="127.0.0.1", port=port_b)

            reg_ok = transport_b.register_peer(ep_a)
            assert reg_ok is True

            # 2. Heartbeat check
            hb_ok = transport_b.heartbeat(ep_a, round_num=1, current_loss=0.5)
            assert hb_ok is True

            # 3. Signature transmission and retrieval
            sig_a = torch.tensor([0.5, -0.2, 0.9, 0.0, 1.2], dtype=torch.float32)
            transport_a.set_local_signature(sig_a, round_num=1, cardinality_p=4)

            fetched_sig = transport_b.fetch_signature(ep_a, round_num=1)
            assert fetched_sig is not None
            assert torch.allclose(fetched_sig, sig_a, atol=1e-6)

            # 4. Model parameter pull
            weights_a = torch.randn(1000, dtype=torch.float32)
            transport_a.set_local_weights(weights_a, round_num=1)

            pulled_w = transport_b.pull_model_weights(ep_a, round_num=1)
            assert pulled_w is not None
            assert torch.allclose(pulled_w, weights_a, atol=1e-6)

        finally:
            transport_a.stop()
            transport_b.stop()
