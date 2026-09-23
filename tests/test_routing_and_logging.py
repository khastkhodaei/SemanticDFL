"""
Unit and Integration Tests for Structured Per-Node Logging, Role State Ledgers, and Telemetry.
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
from semantic_dfl.overlay.son import SemanticOverlayNetwork
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

    # Save per-round JSON
    round_file = save_round_node_roles(round_num=1, node_states=node_states, output_dir=temp_results_dir)
    assert os.path.exists(round_file)
    with open(round_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["round"] == 1
    assert data["total_nodes"] == 2
    assert "0" in data["nodes"]
    assert data["nodes"]["0"]["roles"] == ["LEAF_PEER", "ZONE_INITIATOR"]

    # Save lifecycle history
    history = [
        {"round": 1, "avg_train_loss": 0.485, "avg_test_accuracy": 80.75, "node_states": node_states}
    ]
    history_file = save_node_lifecycle_history(history=history, output_dir=temp_results_dir)
    assert os.path.exists(history_file)
    with open(history_file, "r", encoding="utf-8") as f:
        h_data = json.load(f)
    assert h_data["total_rounds"] == 1
    assert len(h_data["history"]) == 1


def test_coordinator_per_node_ledger_integration(temp_results_dir):
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

    zone_manager = ZoneManager(max_zone_size=2, seed=42)
    son = SemanticOverlayNetwork(zone_manager=zone_manager)

    coordinator = SemanticDFLCoordinator(
        clients=clients,
        k_neighbors=2,
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

        # Check that individual node state entries are populated
        for nid in range(4):
            state = metrics["node_states"][nid]
            assert state["node_id"] == nid
            assert "roles" in state
            assert "train_loss" in state
            assert "test_accuracy" in state
            assert "threshold_tau" in state
            assert "selected_neighbors" in state

        # Check that files were exported to temp_results_dir
        round_1_file = os.path.join(temp_results_dir, "node_roles_round_1.json")
        assert os.path.exists(round_1_file)

        history_file = os.path.join(temp_results_dir, "node_lifecycle_history.json")
        assert os.path.exists(history_file)
