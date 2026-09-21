"""
Unit and Integration Tests for Unified Execution Modes and Transport Factory.

Verifies:
- Factory instantiation and dynamic routing for simulation vs distributed modes.
- Dynamic port allocation logic.
- Exception handling for invalid transport configurations.
- YAML configuration schema parsing for execution block.
- Coordinator lifecycle management with factory-instantiated transports.
"""

from __future__ import annotations

import os
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from semantic_dfl.federated.client import DFLClient
from semantic_dfl.federated.coordinator import SemanticDFLCoordinator
from semantic_dfl.overlay.son import SemanticOverlayNetwork
from semantic_dfl.overlay.zone import ZoneManager
from semantic_dfl.transport import (
    GRPCTransport,
    InMemoryTransport,
    MemoryTransport,
    create_transport,
)


class SimpleLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def dummy_dataset():
    x = torch.randn(20, 8)
    y = torch.randint(0, 2, (20,))
    return DataLoader(TensorDataset(x, y), batch_size=4)


def test_factory_simulation_mode():
    t1 = create_transport(mode="simulation", node_id=0)
    assert isinstance(t1, MemoryTransport)
    assert isinstance(t1, InMemoryTransport)
    assert t1.node_id == 0

    t2 = create_transport(mode="memory", node_id=5, simulated_latency_ms=2.5)
    assert isinstance(t2, MemoryTransport)
    assert t2.node_id == 5
    assert t2.simulated_latency_ms == 2.5

    t3 = create_transport(mode="in_memory", node_id=10)
    assert isinstance(t3, MemoryTransport)


def test_factory_distributed_mode():
    t1 = create_transport(mode="distributed", node_id=0, host="127.0.0.1", base_port=50050)
    assert isinstance(t1, GRPCTransport)
    assert t1.node_id == 0
    assert t1.port == 50050
    assert t1.host == "127.0.0.1"

    t2 = create_transport(mode="distributed", node_id=4, host="127.0.0.1", base_port=50050)
    assert isinstance(t2, GRPCTransport)
    assert t2.node_id == 4
    assert t2.port == 50054

    t3 = create_transport(mode="grpc", node_id=2, port=58000)
    assert isinstance(t3, GRPCTransport)
    assert t3.port == 58000


def test_factory_invalid_mode():
    with pytest.raises(ValueError, match="Unsupported execution/transport mode"):
        create_transport(mode="quantum_entanglement", node_id=0)


def test_default_config_execution_schema():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default_config.yaml")
    assert os.path.exists(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert "execution" in config
    exec_cfg = config["execution"]
    assert "mode" in exec_cfg
    assert exec_cfg["mode"] in ("simulation", "distributed")
    assert "backend" in exec_cfg
    assert "distributed" in exec_cfg

    dist_cfg = exec_cfg["distributed"]
    assert "host" in dist_cfg
    assert "base_port" in dist_cfg
    assert "spawn_subprocesses" in dist_cfg


def test_isolated_profiles_schema():
    configs_dir = os.path.join(os.path.dirname(__file__), "..", "configs")

    # 1. simulation.yaml
    sim_path = os.path.join(configs_dir, "simulation.yaml")
    assert os.path.exists(sim_path)
    with open(sim_path, "r", encoding="utf-8") as f:
        sim_cfg = yaml.safe_load(f)
    assert sim_cfg["execution"]["mode"] == "simulation"
    assert sim_cfg["federated"]["num_clients"] == 100
    assert sim_cfg["simulation"]["rounds"] == 50

    # 2. distributed_local.yaml
    dist_path = os.path.join(configs_dir, "distributed_local.yaml")
    assert os.path.exists(dist_path)
    with open(dist_path, "r", encoding="utf-8") as f:
        dist_cfg = yaml.safe_load(f)
    assert dist_cfg["execution"]["mode"] == "distributed"
    assert dist_cfg["network"]["host"] == "127.0.0.1"
    assert dist_cfg["federated"]["num_clients"] == 3

    # 3. slices_production.yaml
    slices_path = os.path.join(configs_dir, "slices_production.yaml")
    assert os.path.exists(slices_path)
    with open(slices_path, "r", encoding="utf-8") as f:
        slices_cfg = yaml.safe_load(f)
    assert slices_cfg["execution"]["mode"] == "slices_distributed"
    assert slices_cfg["slices"]["num_nodes"] == 100
    assert len(slices_cfg["slices"]["super_peers"]) >= 2


def test_coordinator_lifecycle_with_factory_transports(dummy_dataset):
    MemoryTransport.reset_global_state()

    clients = {}
    for i in range(3):
        model = SimpleLinearModel()
        transport = create_transport(mode="simulation", node_id=i)
        client = DFLClient(
            node_id=i,
            model=model,
            train_loader=dummy_dataset,
            test_loader=dummy_dataset,
            transport=transport,
        )
        clients[i] = client

    zone_manager = ZoneManager(max_zone_size=3, seed=42)
    son = SemanticOverlayNetwork(zone_manager=zone_manager)

    coordinator = SemanticDFLCoordinator(
        clients=clients,
        k_neighbors=2,
        son=son,
        global_test_loader=dummy_dataset,
    )

    # Test context manager and lifecycle methods
    with coordinator:
        # Check that transports started
        for c in clients.values():
            assert c.transport.is_running is True

        # Run 1 round
        metrics = coordinator.step_round(epochs_per_round=1)
        assert metrics["round"] == 1
        assert "avg_train_loss" in metrics
        assert "avg_test_accuracy" in metrics
        assert "avg_recall_at_k" in metrics

    # Transports stopped upon exit
    for c in clients.values():
        assert c.transport.is_running is False
