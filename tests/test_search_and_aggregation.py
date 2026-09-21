"""
Unit and Integration Tests for Peer Search, Adaptive Threshold, and Pull Aggregation.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from semantic_dfl.core.models import MLP2NN, FashionCNN
from semantic_dfl.overlay.clustering import ZoneClustering
from semantic_dfl.overlay.son import SemanticOverlayNetwork
from semantic_dfl.overlay.zone import ZoneManager
from semantic_dfl.search.controller import AdaptiveThresholdController
from semantic_dfl.search.engine import (
    SimilaritySearchEngine,
    compute_recall_at_k,
    ideal_centralized_top_k,
)
from semantic_dfl.aggregation.pull_aggregator import PullAggregator
from semantic_dfl.federated.client import DFLClient
from semantic_dfl.federated.coordinator import SemanticDFLCoordinator


class TestAdaptiveThresholdController:
    """Tests for threshold tuning logic and drift compensation."""

    def test_threshold_increase_on_high_acceptance(self):
        ctrl = AdaptiveThresholdController(tau_0=0.5, phi=0.5, eta_tau=0.1)
        # 100% acceptance rate > 0.5 target phi -> threshold should increase
        new_tau = ctrl.update_threshold(acceptance_rate=1.0, drift_proxy=1.0)
        assert new_tau > 0.5
        assert pytest.approx(new_tau, abs=1e-5) == 0.5 + 0.1 * (1.0 - 0.5)

    def test_threshold_decrease_on_low_acceptance(self):
        ctrl = AdaptiveThresholdController(tau_0=0.5, phi=0.5, eta_tau=0.1)
        # 0% acceptance rate < 0.5 -> threshold should decrease
        new_tau = ctrl.update_threshold(acceptance_rate=0.0, drift_proxy=1.0)
        assert new_tau < 0.5
        assert pytest.approx(new_tau, abs=1e-5) == 0.5 + 0.1 * (0.0 - 0.5)

    def test_threshold_drift_compensation(self):
        ctrl = AdaptiveThresholdController(tau_0=0.5, phi=0.5, eta_tau=0.1, gamma=0.05)
        # Acceptance matches phi (no rate delta), but drift is high (drift_proxy = 0.0)
        new_tau = ctrl.update_threshold(acceptance_rate=0.5, drift_proxy=0.0)
        assert new_tau < 0.5
        assert pytest.approx(new_tau, abs=1e-5) == 0.5 - 0.05 * 1.0

    def test_threshold_relaxation(self):
        ctrl = AdaptiveThresholdController(tau_0=0.6, relax_delta=0.1)
        relaxed = ctrl.relax_threshold()
        assert pytest.approx(relaxed, abs=1e-5) == 0.5


class TestSimilaritySearchEngine:
    """Tests for hierarchical SON top-down search and Recall@K."""

    def test_search_engine_dimension_mismatch_resilience(self):
        """Verifies peer search does not crash on tensor dimension mismatches."""
        query_sig = torch.randn(10)
        all_sigs = {
            0: query_sig,
            1: torch.randn(100),
            2: torch.randn(100),
        }
        son = SemanticOverlayNetwork()
        engine = SimilaritySearchEngine(son=son)
        peers = engine.search_peers(
            query_node_id=0,
            query_signature=query_sig,
            all_signatures=all_sigs,
            k=2,
            tau=0.0,
        )
        assert len(peers) <= 2

    def test_search_engine_recall_against_oracle(self):
        torch.manual_seed(42)
        num_nodes = 30
        p = 8
        m_dim = 50
        k = 4

        # Generate clustered signatures
        c1 = torch.randn(m_dim)
        c2 = torch.randn(m_dim)
        signatures = {}
        for i in range(15):
            signatures[i] = c1 + 0.05 * torch.randn(m_dim)
        for i in range(15, 30):
            signatures[i] = c2 + 0.05 * torch.randn(m_dim)

        # Build overlay
        zm = ZoneManager(max_zone_size=8, seed=42)
        zc = ZoneClustering(damping=0.7, r_replicas=3)
        son = SemanticOverlayNetwork(zone_manager=zm, clustering_engine=zc)
        son.build_topology(signatures, p=p, max_levels=2)

        engine = SimilaritySearchEngine(son=son, beam_cap=4)

        recalls = []
        for q_id in range(num_nodes):
            retrieved = engine.search_peers(
                query_node_id=q_id,
                query_signature=signatures[q_id],
                all_signatures=signatures,
                k=k,
                tau=0.3,
            )
            oracle = ideal_centralized_top_k(
                query_node_id=q_id,
                query_signature=signatures[q_id],
                all_signatures=signatures,
                k=k,
            )
            rec = compute_recall_at_k(retrieved, oracle, k=k)
            recalls.append(rec)

        avg_recall = float(np.mean(recalls))
        assert avg_recall >= 0.80, f"Average Recall@K is {avg_recall:.2f}, expected >= 0.80"


class TestPullAggregator:
    """Tests for row-stochastic softmax weights and collaborative anchor mixing."""

    def test_pull_weights_sum_to_one(self):
        aggregator = PullAggregator(temperature=0.5)
        sims = [0.8, 0.6, 0.4, 0.2]
        weights = aggregator.compute_pull_weights(sims)

        assert weights.shape == (4,)
        assert pytest.approx(weights.sum().item(), abs=1e-5) == 1.0
        # Higher similarity must have strictly higher weight
        assert weights[0] > weights[1] > weights[2] > weights[3]

    def test_aggregation_and_anchor_mixing(self):
        aggregator = PullAggregator(temperature=1.0, default_alpha=0.5)
        
        client_w = torch.tensor([10.0, 10.0])
        # Two identical neighbors with parameters [0, 0]
        neighbor_w = [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])]
        similarities = [0.5, 0.5]

        # Anchor should be [0.0, 0.0]
        # w_new = 0.5 * [10, 10] + 0.5 * [0, 0] = [5.0, 5.0]
        w_new = aggregator.aggregate(
            client_weights=client_w,
            neighbor_weights=neighbor_w,
            similarities=similarities,
            alpha=0.5,
        )
        assert torch.allclose(w_new, torch.tensor([5.0, 5.0]))

    def test_aggregation_empty_neighbors_retains_client_state(self):
        aggregator = PullAggregator()
        client_w = torch.tensor([1.0, 2.0, 3.0])
        w_new = aggregator.aggregate(
            client_weights=client_w,
            neighbor_weights=[],
            similarities=[],
        )
        assert torch.allclose(w_new, client_w)


class TestFederatedCoordinatorIntegration:
    """End-to-end multi-client decentralized learning simulation."""

    def test_multiclient_simulation_run(self):
        torch.manual_seed(42)
        num_clients = 10
        num_samples_per_client = 32
        input_dim = 20
        num_classes = 2

        clients = {}
        for i in range(num_clients):
            # Synthetic binary classification dataset
            x = torch.randn(num_samples_per_client, input_dim)
            y = torch.randint(0, num_classes, (num_samples_per_client,))
            loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)
            
            model = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Linear(16, num_classes),
            )
            optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

            client = DFLClient(
                node_id=i,
                model=model,
                train_loader=loader,
                test_loader=loader,
                optimizer=optimizer,
                tau_0=0.4,
                phi=0.5,
            )
            clients[i] = client

        coordinator = SemanticDFLCoordinator(
            clients=clients,
            p_cardinality=30,
            k_neighbors=3,
            tau_min_fidelity=0.7,
            son_refresh_interval=1,
        )

        # Run 2 decentralized rounds
        res1 = coordinator.step_round(epochs_per_round=1)
        res2 = coordinator.step_round(epochs_per_round=1)

        assert res1["round"] == 1
        assert res2["round"] == 2
        assert res1["cardinality_p"] > 0
        assert 0.0 <= res1["avg_recall_at_k"] <= 1.0
        assert 0.0 <= res2["avg_recall_at_k"] <= 1.0
        assert res1["avg_train_loss"] > 0.0
