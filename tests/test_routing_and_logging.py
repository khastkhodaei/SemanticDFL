"""
Unit and Integration Tests for Structured Per-Node Logging, Role State Ledgers, and Multi-Hop Routing.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from semantic_dfl.federated.client import DFLClient
from semantic_dfl.federated.coordinator import SemanticDFLCoordinator
from semantic_dfl.overlay.routing import MultiHopRouter
from semantic_dfl.overlay.son import SemanticOverlayNetwork
from semantic_dfl.overlay.topology import (
    generate_grid_topology,
    generate_ring_topology,
)
from semantic_dfl.overlay.zone import ZoneManager
from semantic_dfl.transport import create_transport
from semantic_dfl.utils.logger import (
    save_node_lifecycle_history,
    save_round_node_roles,
    setup_node_logger,
    setup_simulation_logger,
)


@pytest.fixture
def temp_results_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        return self.fc(x)


def test_setup_node_logger(temp_results_dir):
    log_dir = os.path.join(temp_results_dir, "logs", "nodes")
    logger = setup_node_logger(node_id=3, log_dir=log_dir)
    assert logger.name == "node_3"
    
    logger.info("Test message for node 3")
    
    log_file = os.path.join(log_dir, "node_3.log")
    assert os.path.exists(log_file)
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Test message for node 3" in content
    assert "[Node-node_3]" in content


def test_setup_simulation_logger(temp_results_dir):
    log_dir = os.path.join(temp_results_dir, "logs")
    logger, filepath = setup_simulation_logger(output_dir=log_dir, timestamp="test_run")
    assert os.path.exists(filepath)
    
    logger.info("Simulation global started")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Simulation global started" in content


def test_save_round_node_roles_and_lifecycle(temp_results_dir):
    node_states = {
        0: {
            "node_id": 0,
            "roles": ["LEAF_PEER", "ZONE_INITIATOR"],
            "zone_id": 1,
            "level": 1,
            "train_loss": 0.45,
            "test_accuracy": 82.5,
            "threshold_tau": 0.5,
            "selected_neighbors": [{"neighbor_id": 1, "similarity": 0.9}],
        },
        1: {
            "node_id": 1,
            "roles": ["LEAF_PEER"],
            "zone_id": 1,
            "level": 1,
            "train_loss": 0.52,
            "test_accuracy": 79.0,
            "threshold_tau": 0.48,
            "selected_neighbors": [{"neighbor_id": 0, "similarity": 0.9}],
        },
    }

    round_file = save_round_node_roles(round_num=1, node_states=node_states, output_dir=temp_results_dir)
    assert os.path.exists(round_file)
    with open(round_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["round"] == 1
    assert data["total_nodes"] == 2
    assert "0" in data["nodes"]
    assert data["nodes"]["0"]["roles"] == ["LEAF_PEER", "ZONE_INITIATOR"]

    history = [
        {"round": 1, "avg_train_loss": 0.485, "avg_test_accuracy": 80.75, "node_states": node_states}
    ]
    history_file = save_node_lifecycle_history(history=history, output_dir=temp_results_dir)
    assert os.path.exists(history_file)
    with open(history_file, "r", encoding="utf-8") as f:
        h_data = json.load(f)
    assert h_data["total_rounds"] == 1
    assert len(h_data["history"]) == 1


def test_multi_hop_router_ring_and_grid(temp_results_dir):
    # 1. Ring topology with 10 nodes (degree 2 -> simple cycle 0-1-2-3-4-5-6-7-8-9)
    ring_adj = generate_ring_topology(num_nodes=10, degree=2)
    router_ring = MultiHopRouter(adjacency_graph=ring_adj)

    # From 0 to 3 shortest path is [0, 1, 2, 3] (3 hops)
    path_0_3 = router_ring.get_shortest_path(0, 3)
    assert path_0_3 == [0, 1, 2, 3]
    assert router_ring.get_hop_count(0, 3) == 3

    # From 0 to 9 shortest path is [0, 9] (1 hop via wrap-around)
    path_0_9 = router_ring.get_shortest_path(0, 9)
    assert path_0_9 == [0, 9]
    assert router_ring.get_hop_count(0, 9) == 1

    # Route packet with forwarding log
    record = router_ring.route_packet(src=0, dst=3, packet_type="MODEL_PULL", payload_bytes=1000)
    assert record["src"] == 0
    assert record["dst"] == 3
    assert record["hop_count"] == 3
    assert len(record["hops"]) == 3
    assert record["hops"][0] == {"hop": 1, "from_node": 0, "to_node": 1}
    assert record["hops"][1] == {"hop": 2, "from_node": 1, "to_node": 2}
    assert record["hops"][2] == {"hop": 3, "from_node": 2, "to_node": 3}

    # 2. 2D Grid topology (16 nodes, 4x4)
    grid_adj = generate_grid_topology(num_nodes=16)
    router_grid = MultiHopRouter(adjacency_graph=grid_adj)
    
    # From top-left (0) to bottom-right (15): Manhattan distance = 3 + 3 = 6 hops
    hops_0_15 = router_grid.get_hop_count(0, 15)
    assert hops_0_15 >= 2
    path_0_15 = router_grid.get_shortest_path(0, 15)
    assert path_0_15[0] == 0
    assert path_0_15[-1] == 15

    # Export audit
    audit_file = router_ring.export_routing_audit(round_num=1, transactions=[record], output_dir=temp_results_dir)
    assert os.path.exists(audit_file)
    with open(audit_file, "r", encoding="utf-8") as f:
        audit_data = json.load(f)
    assert audit_data["round"] == 1
    assert audit_data["summary"]["total_hops"] == 3
    assert audit_data["summary"]["total_transactions"] == 1


def test_coordinator_per_node_ledger_and_routing_integration(temp_results_dir):
    x = torch.randn(20, 8)
    y = torch.randint(0, 2, (20,))
    dataset = DataLoader(TensorDataset(x, y), batch_size=4)

    clients = {}
    for i in range(4):
        model = SimpleModel()
        transport = create_transport(mode="simulation", node_id=i)
        client = DFLClient(
            node_id=i,
            model=model,
            train_loader=dataset,
            test_loader=dataset,
            transport=transport,
        )
        clients[i] = client

    # Ring adjacency for 4 nodes
    ring_adj = generate_ring_topology(num_nodes=4, degree=2)
    zone_manager = ZoneManager(max_zone_size=2, seed=42)
    son = SemanticOverlayNetwork(zone_manager=zone_manager)

    coordinator = SemanticDFLCoordinator(
        clients=clients,
        k_neighbors=2,
        adjacency_graph=ring_adj,
        son=son,
        global_test_loader=dataset,
        results_dir=temp_results_dir,
        export_node_states=True,
    )

    with coordinator:
        metrics = coordinator.step_round(epochs_per_round=1)
        assert metrics["round"] == 1
        assert "node_states" in metrics
        assert len(metrics["node_states"]) == 4
        assert "routing" in metrics
        assert metrics["routing"]["total_transactions"] >= 0

        # Check exported files
        round_1_file = os.path.join(temp_results_dir, "node_roles_round_1.json")
        assert os.path.exists(round_1_file)

        history_file = os.path.join(temp_results_dir, "node_lifecycle_history.json")
        assert os.path.exists(history_file)

        audit_file = os.path.join(temp_results_dir, "routing_audit_round_1.json")
        assert os.path.exists(audit_file)
