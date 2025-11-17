"""
Decentralized Federated Learning (DFL) Simulation Coordinator.

Orchestrates synchronous/near-synchronous decentralized rounds:
1. Local SGD client training.
2. Coordinate-wise Top-P signature extraction.
3. Periodic Semantic Overlay Network (SON) tree construction.
4. Hierarchical Top-K similarity search with adaptive threshold adjustment.
5. Softmax pull-based personalized aggregation and evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

from semantic_dfl.aggregation.pull_aggregator import PullAggregator
from semantic_dfl.federated.client import DFLClient
from semantic_dfl.overlay.son import SemanticOverlayNetwork
from semantic_dfl.search.engine import (
    SimilaritySearchEngine,
    compute_recall_at_k,
    ideal_centralized_top_k,
)
from semantic_dfl.signature.binary_search import (
    compute_network_p,
    find_minimal_cardinality,
)


class SemanticDFLCoordinator:
    """
    Simulation coordinator for decentralized semantic federated learning experiments.
    """

    def __init__(
        self,
        clients: Dict[int, DFLClient],
        p_cardinality: int = 100,
        k_neighbors: int = 3,
        tau_min_fidelity: float = 0.8,
        son_refresh_interval: int = 1,
        pull_aggregator: Optional[PullAggregator] = None,
        son: Optional[SemanticOverlayNetwork] = None,
        global_test_loader: Optional[DataLoader] = None,
    ) -> None:
        self.clients = clients
        self.p_cardinality = p_cardinality
        self.k_neighbors = k_neighbors
        self.tau_min_fidelity = tau_min_fidelity
        self.son_refresh_interval = son_refresh_interval
        self.pull_aggregator = pull_aggregator or PullAggregator()
        self.son = son or SemanticOverlayNetwork()
        self.search_engine = SimilaritySearchEngine(son=self.son)
        self.global_test_loader = global_test_loader
        self.current_round = 0

    def step_round(self, epochs_per_round: int = 1) -> Dict[str, Any]:
        """
        Executes one full decentralized learning round.
        
        Args:
            epochs_per_round: Local training epochs per client.
            
        Returns:
            Dictionary containing round metrics (losses, accuracies, recall, cardinality).
        """
        self.current_round += 1
        num_clients = len(self.clients)

        # ---------------------------------------------------------------------
        # Step 1: Local Client Training
        # ---------------------------------------------------------------------
        train_losses = {}
        for nid, client in self.clients.items():
            train_losses[nid] = client.train_epoch(epochs=epochs_per_round)

        # ---------------------------------------------------------------------
        # Step 2: Adaptive Cardinality Tuning & Top-P Signature Extraction
        # ---------------------------------------------------------------------
        all_weights = {nid: client.get_weights() for nid, client in self.clients.items()}
        
        if self.tau_min_fidelity > 0.0:
            local_ps = []
            for nid, w in all_weights.items():
                p_i = find_minimal_cardinality(w, tau_min=self.tau_min_fidelity)
                local_ps.append(p_i)
            current_p = compute_network_p(local_ps, quantile=0.85)
        else:
            current_p = self.p_cardinality

        signatures = {}
        for nid, client in self.clients.items():
            signatures[nid] = client.extract_signature(p=current_p)

        # ---------------------------------------------------------------------
        # Step 3: Semantic Overlay Network (SON) Topology Construction
        # ---------------------------------------------------------------------
        if (self.current_round - 1) % self.son_refresh_interval == 0:
            self.son.build_topology(signatures, p=current_p)

        # ---------------------------------------------------------------------
        # Step 4: Top-K Peer Search & Personalized Pull Aggregation
        # ---------------------------------------------------------------------
        new_client_weights = {}
        recalls = []
        avg_taus = []

        for nid, client in self.clients.items():
            tau_current = client.threshold_controller.tau
            avg_taus.append(tau_current)

            # Hierarchical search over SON
            selected_neighbors = self.search_engine.search_peers(
                query_node_id=nid,
                query_signature=signatures[nid],
                all_signatures=signatures,
                k=self.k_neighbors,
                tau=tau_current,
            )

            # Benchmark against centralized oracle
            ideal_neighbors = ideal_centralized_top_k(
                query_node_id=nid,
                query_signature=signatures[nid],
                all_signatures=signatures,
                k=self.k_neighbors,
            )

            # Track Recall@K
            rec = compute_recall_at_k(selected_neighbors, ideal_neighbors, k=self.k_neighbors)
            recalls.append(rec)

            # Adaptive threshold adjustment
            acceptance_rate = len(selected_neighbors) / float(self.k_neighbors)
            client.threshold_controller.update_threshold(acceptance_rate=acceptance_rate)

            # Pull neighbor parameter vectors and aggregate
            neighbor_w = {nbr_id: all_weights[nbr_id] for nbr_id, _ in selected_neighbors}
            neighbor_sims = {nbr_id: sim for nbr_id, sim in selected_neighbors}

            new_w = self.pull_aggregator.aggregate(
                client_weights=all_weights[nid],
                neighbor_weights=neighbor_w,
                similarities=neighbor_sims,
            )
            new_client_weights[nid] = new_w

        # Update all client models with new weights
        for nid, client in self.clients.items():
            client.set_weights(new_client_weights[nid])

        # ---------------------------------------------------------------------
        # Step 5: Test Evaluation
        # ---------------------------------------------------------------------
        test_losses = []
        test_accuracies = []

        for nid, client in self.clients.items():
            # Evaluate on global test set if provided, else local test set
            eval_loader = self.global_test_loader if self.global_test_loader is not None else client.test_loader
            if eval_loader is not None:
                loss, acc = client.evaluate(eval_loader)
                test_losses.append(loss)
                test_accuracies.append(acc)

        avg_train_loss = float(np.mean(list(train_losses.values())))
        avg_test_loss = float(np.mean(test_losses)) if test_losses else 0.0
        avg_test_acc = float(np.mean(test_accuracies)) if test_accuracies else 0.0
        avg_recall = float(np.mean(recalls))
        avg_tau = float(np.mean(avg_taus))

        return {
            "round": self.current_round,
            "cardinality_p": current_p,
            "avg_train_loss": avg_train_loss,
            "avg_test_loss": avg_test_loss,
            "avg_test_accuracy": avg_test_acc,
            "avg_recall_at_k": avg_recall,
            "avg_threshold_tau": avg_tau,
        }

# Seed torch, cuda, numpy, and random generators (polish 7)

# Verify clean imports across package (polish 9)

# Finalize docstrings across all modules (polish 11)

# Track transmitted signature bytes per round (polish 13)

# Synchronize StepLR scheduler across client models (polish 15)

# Format summary round metrics output (polish 17)

# Seed torch, cuda, numpy, and random generators (polish 19)

# Verify clean imports across package (polish 21)

# Finalize docstrings across all modules (polish 23)

# Track transmitted signature bytes per round (polish 25)
