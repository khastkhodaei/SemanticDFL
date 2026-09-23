"""
Decentralized Federated Learning (DFL) Simulation Coordinator.

Orchestrates synchronous/near-synchronous decentralized rounds:
1. Local SGD client training.
2. Coordinate-wise Top-P signature extraction with cosine-retention binary search.
3. Periodic Semantic Overlay Network (SON) tree construction with physical topology awareness.
4. Hierarchical Top-K similarity search with adaptive threshold adjustment and search stats.
5. Softmax pull-based personalized aggregation with freshness checking.
6. Granular academic logging (control/data plane traffic, role distribution, Recall@K, timings).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        adjacency_graph: Optional[Dict[int, List[int]]] = None,
        pull_aggregator: Optional[PullAggregator] = None,
        son: Optional[SemanticOverlayNetwork] = None,
        global_test_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        verbose: bool = False,
    ) -> None:
        self.clients = clients
        self.p_cardinality = p_cardinality
        self.k_neighbors = k_neighbors
        self.tau_min_fidelity = tau_min_fidelity
        self.son_refresh_interval = son_refresh_interval
        self.adjacency_graph = adjacency_graph
        self.pull_aggregator = pull_aggregator or PullAggregator()
        self.son = son or SemanticOverlayNetwork()
        self.search_engine = SimilaritySearchEngine(son=self.son)
        self.global_test_loader = global_test_loader
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.verbose = verbose
        self.current_round = 0

    def start(self) -> None:
        """Starts underlying client transport adapters and initializes round-0 weights."""
        for client in self.clients.values():
            if client.transport is not None:
                client.transport.start()
                client.transport.set_local_weights(client.get_weights(), round_num=0)

    def stop(self) -> None:
        """Gracefully stops all client transport layers and releases sockets."""
        for client in self.clients.values():
            if client.transport is not None:
                client.transport.stop()

    def __enter__(self) -> SemanticDFLCoordinator:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    def step_round(self, epochs_per_round: int = 1) -> Dict[str, Any]:
        """
        Executes one full decentralized learning round with granular phase timers and progress tracking.
        
        Args:
            epochs_per_round: Local training epochs per client.
            
        Returns:
            Dictionary containing round metrics (losses, accuracies, recall, cardinality, traffic, timings).
        """
        self.current_round += 1
        num_clients = len(self.clients)
        timings: Dict[str, float] = {}

        # ---------------------------------------------------------------------
        # Step 1: Local Client Training
        # ---------------------------------------------------------------------
        if self.verbose:
            print("  [1/5] Local Training ...", flush=True)
        t0 = time.perf_counter()
        train_losses = {}
        client_iter = tqdm(self.clients.items(), desc="Training Clients", leave=False) if self.verbose else self.clients.items()
        for nid, client in client_iter:
            train_losses[nid] = client.train_epoch(epochs=epochs_per_round)
        timings["time_local_training"] = time.perf_counter() - t0

        # ---------------------------------------------------------------------
        # Step 2: Adaptive Cardinality Tuning & Top-P Signature Extraction
        # ---------------------------------------------------------------------
        if self.verbose:
            print("  [2/5] Signature Extraction ...", flush=True)
        t0 = time.perf_counter()
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
        sig_iter = tqdm(self.clients.items(), desc="Extracting Signatures", leave=False) if self.verbose else self.clients.items()
        for nid, client in sig_iter:
            signatures[nid] = client.extract_signature(p=current_p, round_num=self.current_round)
        timings["time_signatures"] = time.perf_counter() - t0

        # ---------------------------------------------------------------------
        # Step 3: Semantic Overlay Network (SON) Topology Construction
        # ---------------------------------------------------------------------
        if self.verbose:
            print("  [3/5] SON & AP Clustering ...", flush=True)
        t0 = time.perf_counter()
        if (self.current_round - 1) % self.son_refresh_interval == 0:
            self.son.build_topology(
                signatures=signatures,
                p=current_p,
                adjacency_graph=self.adjacency_graph,
                epoch_seed=self.current_round,
            )
        timings["time_topology"] = time.perf_counter() - t0
        role_counts = self.son.get_role_distribution()

        # ---------------------------------------------------------------------
        # Step 4: Top-K Peer Search & Communication Traffic Accounting
        # ---------------------------------------------------------------------
        if self.verbose:
            print("  [4/5] Similarity Search ...", flush=True)
        t_search_start = time.perf_counter()
        selected_neighbors_all = {}
        recalls = []
        avg_taus = []
        total_control_bytes = 0

        search_iter = tqdm(self.clients.items(), desc="Similarity Search", leave=False) if self.verbose else self.clients.items()
        for nid, client in search_iter:
            tau_current = client.threshold_controller.tau
            avg_taus.append(tau_current)

            # Hierarchical search over SON
            selected_neighbors, search_stats = self.search_engine.search_peers(
                query_node_id=nid,
                query_signature=signatures[nid],
                all_signatures=signatures,
                k=self.k_neighbors,
                tau=tau_current,
                return_stats=True,
            )
            selected_neighbors_all[nid] = selected_neighbors
            total_control_bytes += search_stats.get("control_bytes", 0)

            # Centralized oracle baseline for Recall@K
            ideal_neighbors = ideal_centralized_top_k(
                query_node_id=nid,
                query_signature=signatures[nid],
                all_signatures=signatures,
                k=self.k_neighbors,
            )

            rec = compute_recall_at_k(selected_neighbors, ideal_neighbors, k=self.k_neighbors)
            recalls.append(rec)

            # Update threshold based on observed acceptance rate in search
            acceptance_rate = search_stats.get("acceptance_rate", len(selected_neighbors) / float(self.k_neighbors))
            client.threshold_controller.update_threshold(acceptance_rate=acceptance_rate)
        timings["time_search"] = time.perf_counter() - t_search_start

        # ---------------------------------------------------------------------
        # Step 5: Pull Aggregation & Data Traffic Accounting
        # ---------------------------------------------------------------------
        if self.verbose:
            print("  [5/5] Pull Aggregation ...", flush=True)
        t_agg_start = time.perf_counter()
        new_client_weights = {}
        total_data_bytes = 0
        m_params = next(iter(all_weights.values())).numel() if all_weights else 0
        model_bytes_fp32 = m_params * 4

        agg_iter = tqdm(self.clients.items(), desc="Pull Aggregation", leave=False) if self.verbose else self.clients.items()
        for nid, client in agg_iter:
            selected_neighbors = selected_neighbors_all[nid]
            neighbor_w = {nbr_id: all_weights[nbr_id] for nbr_id, _ in selected_neighbors}
            neighbor_sims = {nbr_id: sim for nbr_id, sim in selected_neighbors}
            total_data_bytes += len(neighbor_w) * model_bytes_fp32

            new_w = self.pull_aggregator.aggregate(
                client_weights=all_weights[nid],
                neighbor_weights=neighbor_w,
                similarities=neighbor_sims,
            )
            new_client_weights[nid] = new_w

        # Update all client models with new weights
        for nid, client in self.clients.items():
            client.set_weights(new_client_weights[nid], round_num=self.current_round)
        timings["time_aggregation"] = time.perf_counter() - t_agg_start

        # ---------------------------------------------------------------------
        # Step 6: Test Evaluation
        # ---------------------------------------------------------------------
        t0 = time.perf_counter()
        test_losses = []
        test_accuracies = []

        eval_iter = tqdm(self.clients.items(), desc="Evaluating Clients", leave=False) if self.verbose else self.clients.items()
        for nid, client in eval_iter:
            eval_loader = self.global_test_loader if self.global_test_loader is not None else client.test_loader
            if eval_loader is not None:
                loss, acc = client.evaluate(eval_loader)
                test_losses.append(loss)
                test_accuracies.append(acc)
        timings["time_evaluation"] = time.perf_counter() - t0

        avg_train_loss = float(np.mean(list(train_losses.values())))
        avg_test_loss = float(np.mean(test_losses)) if test_losses else 0.0
        avg_test_acc = float(np.mean(test_accuracies)) if test_accuracies else 0.0
        avg_recall = float(np.mean(recalls))
        avg_tau = float(np.mean(avg_taus))

        control_mb = total_control_bytes / (1024.0 * 1024.0)
        data_mb = total_data_bytes / (1024.0 * 1024.0)
        total_mb = control_mb + data_mb

        return {
            "round": self.current_round,
            "cardinality_p": current_p,
            "avg_train_loss": avg_train_loss,
            "avg_test_loss": avg_test_loss,
            "avg_test_accuracy": avg_test_acc,
            "avg_recall_at_k": avg_recall,
            "avg_threshold_tau": avg_tau,
            "role_distribution": role_counts,
            "traffic": {
                "control_mb": control_mb,
                "data_mb": data_mb,
                "total_mb": total_mb,
                "per_node_total_mb": total_mb / max(1, num_clients),
            },
            "timings": timings,
        }
