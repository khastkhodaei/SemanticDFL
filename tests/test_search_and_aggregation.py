"""
Unit and Integration Tests for Peer Search, Adaptive Threshold, and Pull Aggregation.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from semantic_dfl.core.models import (
    FashionCNN,
    ResNet18Small,
    GoogleSpeechResNet,
    AlbertTextClassifier,
    MLP2NN,
    get_flattened_params,
    set_flattened_params,
)
from semantic_dfl.overlay.clustering import ZoneClustering
from semantic_dfl.overlay.son import SemanticOverlayNetwork
from semantic_dfl.overlay.zone import ZoneManager
from semantic_dfl.search.controller import AdaptiveThresholdController
from semantic_dfl.search.engine import (
    SimilaritySearchEngine,
    compute_recall_at_k,
    ideal_centralized_top_k,
)
from semantic_dfl.aggregation.pull_aggregator import (
    PullAggregator,
    compute_alpha_from_psi,
    compute_data_plane_traffic_bytes,
)
from semantic_dfl.federated.client import DFLClient
from semantic_dfl.federated.coordinator import SemanticDFLCoordinator


class TestAdaptiveThresholdController:
    """Tests for threshold tuning logic and drift compensation."""

    def test_threshold_increase_on_high_acceptance(self):
        ctrl = AdaptiveThresholdController(tau_0=0.5, phi=0.5, eta_tau=0.1)
        new_tau = ctrl.update_threshold(acceptance_rate=1.0, drift_proxy=1.0)
        assert new_tau > 0.5
        assert pytest.approx(new_tau, abs=1e-5) == 0.5 + 0.1 * (1.0 - 0.5)

    def test_threshold_decrease_on_low_acceptance(self):
        ctrl = AdaptiveThresholdController(tau_0=0.5, phi=0.5, eta_tau=0.1)
        new_tau = ctrl.update_threshold(acceptance_rate=0.0, drift_proxy=1.0)
        assert new_tau < 0.5
        assert pytest.approx(new_tau, abs=1e-5) == 0.5 + 0.1 * (0.0 - 0.5)

    def test_threshold_drift_compensation(self):
        ctrl = AdaptiveThresholdController(tau_0=0.5, phi=0.5, eta_tau=0.1, gamma=0.05)
        new_tau = ctrl.update_threshold(acceptance_rate=0.5, drift_proxy=0.0)
        assert new_tau < 0.5
        assert pytest.approx(new_tau, abs=1e-5) == 0.5 - 0.05 * 1.0

    def test_threshold_relaxation(self):
        ctrl = AdaptiveThresholdController(tau_0=0.6, relax_delta=0.1)
        relaxed = ctrl.relax_threshold()
        assert pytest.approx(relaxed, abs=1e-5) == 0.5

        ctrl_default = AdaptiveThresholdController(tau_0=0.5)
        relaxed_default = ctrl_default.relax_threshold()
        assert pytest.approx(relaxed_default, abs=1e-5) == 0.48


class TestSimilaritySearchEngine:
    """Tests for hierarchical SON top-down search and Recall@K."""

    def test_search_engine_dimension_mismatch_resilience(self):
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

        c1 = torch.randn(m_dim)
        c2 = torch.randn(m_dim)
        signatures = {}
        for i in range(15):
            signatures[i] = c1 + 0.05 * torch.randn(m_dim)
        for i in range(15, 30):
            signatures[i] = c2 + 0.05 * torch.randn(m_dim)

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
    """Tests for row-stochastic softmax weights, alpha personalization, freshness guards, and multi-model support."""

    def test_alpha_coupling_formula(self):
        # psi=0 => alpha=0 (pure local)
        assert compute_alpha_from_psi(psi=0.0, eta=1.0) == 0.0

        # psi=2, eta=1 => alpha = 2/3
        assert pytest.approx(compute_alpha_from_psi(psi=2.0, eta=1.0), abs=1e-4) == 2.0 / 3.0

        # psi=100 => alpha approaches 1.0
        assert pytest.approx(compute_alpha_from_psi(psi=100.0, eta=1.0), abs=1e-2) == 1.0

        # Negative psi raises ValueError
        with pytest.raises(ValueError):
            compute_alpha_from_psi(psi=-1.0)

    def test_data_plane_traffic_accounting(self):
        # 1000 parameters * 4 bytes/float * 5 pulled models = 20000 bytes
        traffic = compute_data_plane_traffic_bytes(num_pulled_models=5, num_params=1000)
        assert traffic == 20000

    @pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0, 2.0])
    def test_pull_weights_temperatures(self, temperature):
        aggregator = PullAggregator(temperature=temperature)
        sims = [0.9, 0.7, 0.4, 0.1]
        weights = aggregator.compute_pull_weights(sims)

        assert weights.shape == (4,)
        assert pytest.approx(weights.sum().item(), abs=1e-5) == 1.0
        assert weights[0] > weights[1] > weights[2] > weights[3]

        # Lower temperature produces sharper softmax distribution
        if temperature == 0.1:
            assert weights[0] > 0.8

    def test_freshness_guard_filter(self):
        agg = PullAggregator(t_fresh=0.7, max_staleness=2)
        q_sig = torch.zeros(20)
        q_sig[:10] = 1.0

        fresh_sig = q_sig.clone()
        drifted_sig = torch.zeros(20)
        drifted_sig[10:] = 1.0

        cand_sigs = {
            1: fresh_sig,    # Same round (staleness 0) -> accept
            2: fresh_sig,    # Delayed by 1 round (staleness 1 <= 2, sim 1.0 >= 0.7) -> accept
            3: drifted_sig,  # Delayed by 1 round (staleness 1 <= 2, sim 0.0 < 0.7) -> reject
            4: fresh_sig,    # Stale by 3 rounds (staleness 3 > 2) -> reject
        }
        cand_rounds = {1: 10, 2: 9, 3: 9, 4: 7}

        accepted = agg.filter_fresh_replies(
            query_signature=q_sig,
            candidate_signatures=cand_sigs,
            candidate_rounds=cand_rounds,
            current_round=10,
        )
        assert accepted == {1, 2}

    def test_aggregation_and_anchor_mixing(self):
        aggregator = PullAggregator(temperature=1.0, default_alpha=0.5)
        
        client_w = torch.tensor([10.0, 10.0])
        neighbor_w = [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])]
        similarities = [0.5, 0.5]

        w_new = aggregator.aggregate(
            client_weights=client_w,
            neighbor_weights=neighbor_w,
            similarities=similarities,
            alpha=0.5,
        )
        assert torch.allclose(w_new, torch.tensor([5.0, 5.0]))

    def test_aggregation_with_psi_coupling(self):
        aggregator = PullAggregator(eta=1.0)
        client_w = torch.tensor([10.0, 10.0])
        neighbor_w = [torch.tensor([0.0, 0.0])]
        similarities = [1.0]

        # psi = 0 => alpha = 0 => w_new = client_w
        w_pure_local = aggregator.aggregate(
            client_weights=client_w,
            neighbor_weights=neighbor_w,
            similarities=similarities,
            psi=0.0,
        )
        assert torch.allclose(w_pure_local, client_w)

        # psi = 2.0 => alpha = 2/3 => w_new = 1/3 * 10 + 2/3 * 0 = 3.333
        w_mixed = aggregator.aggregate(
            client_weights=client_w,
            neighbor_weights=neighbor_w,
            similarities=similarities,
            psi=2.0,
        )
        assert pytest.approx(w_mixed[0].item(), abs=1e-3) == 10.0 / 3.0

    @pytest.mark.parametrize(
        "model_cls, kwargs",
        [
            (FashionCNN, {"in_channels": 1, "num_classes": 10}),
            (ResNet18Small, {"in_channels": 3, "num_classes": 10}),
            (GoogleSpeechResNet, {"in_channels": 1, "num_classes": 35}),
            (AlbertTextClassifier, {"vocab_size": 100, "embed_dim": 32, "num_heads": 2, "hidden_dim": 64, "num_layers": 1, "num_classes": 20, "max_seq_len": 32}),
        ],
    )
    def test_aggregation_across_benchmark_models(self, model_cls, kwargs):
        aggregator = PullAggregator(temperature=0.5, default_alpha=0.6)
        
        # Instantiate 3 peer models with same architecture
        m1 = model_cls(**kwargs)
        m2 = model_cls(**kwargs)
        m3 = model_cls(**kwargs)

        p1 = get_flattened_params(m1)
        p2 = get_flattened_params(m2)
        p3 = get_flattened_params(m3)

        updated_p = aggregator.aggregate(
            client_weights=p1,
            neighbor_weights=[p2, p3],
            similarities=[0.8, 0.6],
        )

        assert updated_p.shape == p1.shape
        assert not torch.isnan(updated_p).any()
        
        # Verify unflattening into model succeeds
        set_flattened_params(m1, updated_p)
        assert torch.allclose(get_flattened_params(m1), updated_p)


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

        res1 = coordinator.step_round(epochs_per_round=1)
        res2 = coordinator.step_round(epochs_per_round=1)

        assert res1["round"] == 1
        assert res2["round"] == 2
        assert res1["cardinality_p"] > 0
        assert 0.0 <= res1["avg_recall_at_k"] <= 1.0
        assert 0.0 <= res2["avg_recall_at_k"] <= 1.0
        assert res1["avg_train_loss"] > 0.0
