"""
Distributed Node Process CLI for Real-World Network & Testbed Execution.

Starts an autonomous SemanticDFL peer node running on dedicated host:port sockets,
interacting with remote peers over gRPC network channels.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from semantic_dfl.aggregation.pull_aggregator import PullAggregator
from semantic_dfl.core.models import FashionCNN, MLP2NN, ResNet18Small
from semantic_dfl.data.datasets import load_partitioned_dataset
from semantic_dfl.data.partitioning import DirichletPartitioner, IIDPartitioner, PathologicalPartitioner
from semantic_dfl.federated.client import DFLClient
from semantic_dfl.overlay.son import SemanticOverlayNetwork
from semantic_dfl.overlay.zone import ZoneManager
from semantic_dfl.search.engine import SimilaritySearchEngine, compute_recall_at_k, ideal_centralized_top_k
from semantic_dfl.transport import PeerEndpoint, create_transport

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Node-%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def load_yaml_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="SemanticDFL Distributed Peer Node Runner")
    parser.add_argument("--node_id", type=int, required=True, help="Unique integer ID of this peer")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding host address")
    parser.add_argument("--port", type=int, default=50051, help="Binding TCP port")
    parser.add_argument("--bootstrap", type=str, default="", help="Bootstrap super-peer address (host:port)")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to YAML configuration")
    parser.add_argument("--is_super_peer", action="store_true", help="Designate node as initial super-peer")
    parser.add_argument("--rounds", type=int, default=0, help="Override communication rounds from config")
    args = parser.parse_args()

    node_logger = logging.getLogger(f"{args.node_id}")
    config = load_yaml_config(args.config)
    sim_cfg = config.get("simulation", {})
    fed_cfg = config.get("federated", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    rounds = args.rounds if args.rounds > 0 else sim_cfg.get("rounds", 10)
    seed = sim_cfg.get("seed", 42)
    device_cfg = sim_cfg.get("device", "auto")
    device = torch.device("cuda" if (device_cfg == "auto" and torch.cuda.is_available()) else "cpu")

    num_clients = fed_cfg.get("num_clients", 10)
    k_neighbors = fed_cfg.get("k_neighbors", 2)
    zone_size = fed_cfg.get("zone_size", 5)
    alpha_personalization = fed_cfg.get("alpha_personalization", 0.67)

    dataset_name = data_cfg.get("dataset", "fmnist")
    batch_size = data_cfg.get("batch_size", 32)
    data_root = data_cfg.get("data_dir", "./Datasets")
    lr = model_cfg.get("learning_rate", 0.01)
    local_epochs = model_cfg.get("local_epochs", 1)

    node_logger.info("=" * 60)
    node_logger.info(f"Starting Distributed DFL Peer Node {args.node_id}")
    node_logger.info(f"Address:      {args.host}:{args.port}")
    node_logger.info(f"Bootstrap:    {args.bootstrap or 'Self (Bootstrap Leader)'}")
    node_logger.info(f"Super-Peer:   {args.is_super_peer}")
    node_logger.info(f"Device:       {device}")
    node_logger.info(f"Total Rounds: {rounds}")
    node_logger.info("=" * 60)

    # Initialize and start gRPC Transport Layer via factory
    transport = create_transport(
        mode="distributed",
        node_id=args.node_id,
        host=args.host,
        port=args.port,
        timeout_seconds=15.0,
    )
    transport.start()

    # Graceful shutdown handler
    def shutdown_handler(sig, frame):
        node_logger.info("Received termination signal. Shutting down transport...")
        transport.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # Register with bootstrap node if specified
    if args.bootstrap and args.bootstrap != f"{args.host}:{args.port}":
        b_parts = args.bootstrap.split(":")
        b_host = b_parts[0]
        b_port = int(b_parts[1]) if len(b_parts) > 1 else 50050
        bootstrap_ep = PeerEndpoint(node_id=0, hostname=b_host, port=b_port, role="bootstrap")
        node_logger.info(f"Connecting and registering with bootstrap node at {args.bootstrap}...")
        
        reg_success = False
        for attempt in range(1, 6):
            if transport.register_peer(bootstrap_ep):
                reg_success = True
                node_logger.info("Successfully registered with bootstrap peer!")
                break
            time.sleep(1.5)

        if not reg_success:
            node_logger.warning("Failed to register with bootstrap after 5 attempts. Continuing independently.")

    # Load partition for this client
    partitioner = DirichletPartitioner(alpha=data_cfg.get("alpha", 0.3), seed=seed)
    client_loaders, test_loader = load_partitioned_dataset(
        dataset_name=dataset_name,
        partitioner=partitioner,
        num_clients=num_clients,
        batch_size=batch_size,
        root=data_root,
        download=True,
    )

    client_id = args.node_id % num_clients
    train_loader = client_loaders[client_id]

    # Initialize model
    is_rgb = dataset_name.lower() in ("cifar10", "cifar-10")
    if model_cfg.get("architecture", "FashionCNN") == "FashionCNN":
        model = FashionCNN(in_channels=3 if is_rgb else 1, num_classes=10).to(device)
    else:
        model = MLP2NN(input_dim=3072 if is_rgb else 784, num_classes=10).to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    client = DFLClient(
        node_id=args.node_id,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        lr=lr,
        device=device,
        transport=transport,
    )

    pull_aggregator = PullAggregator(default_alpha=alpha_personalization)
    zone_manager = ZoneManager(max_zone_size=zone_size, seed=seed)
    son = SemanticOverlayNetwork(zone_manager=zone_manager)
    search_engine = SimilaritySearchEngine(son=son)

    # Initial weights registration
    initial_w = client.get_weights()
    transport.set_local_weights(initial_w, round_num=0)

    node_logger.info(f"Entering distributed federated training loop ({rounds} rounds)...")

    for r in range(1, rounds + 1):
        round_t0 = time.time()
        node_logger.info(f"--- Communication Round {r}/{rounds} ---")

        # Step 1: Local SGD
        train_loss = client.train_epoch(epochs=local_epochs)
        node_logger.info(f"Local SGD completed | Train Loss: {train_loss:.4f}")

        # Step 2: Signature Extraction
        sig = client.extract_signature(p=100, round_num=r)
        transport.set_local_weights(client.get_weights(), round_num=r)

        # Step 3: Peer Discovery & Signature Gathering via gRPC
        known_peers = transport.discover_peers(query_node_id=args.node_id)
        signatures: Dict[int, torch.Tensor] = {args.node_id: sig}

        for peer in known_peers:
            remote_sig = transport.fetch_signature(peer, round_num=r)
            if remote_sig is not None:
                signatures[peer.node_id] = remote_sig

        # Step 4: Top-K Similarity Search
        tau_current = client.threshold_controller.tau
        selected_neighbors = search_engine.search_peers(
            query_node_id=args.node_id,
            query_signature=sig,
            all_signatures=signatures,
            k=k_neighbors,
            tau=tau_current,
        )

        # Step 5: Pull Remote Model Weights via gRPC & Softmax Pull Aggregation
        neighbor_weights: Dict[int, torch.Tensor] = {}
        neighbor_sims: Dict[int, float] = {}

        for nbr_id, sim in selected_neighbors:
            # Find peer endpoint
            nbr_ep = next((p for p in known_peers if p.node_id == nbr_id), None)
            if nbr_ep is not None:
                pulled_w = transport.pull_model_weights(nbr_ep, round_num=r)
                if pulled_w is not None:
                    neighbor_weights[nbr_id] = pulled_w
                    neighbor_sims[nbr_id] = sim
            elif nbr_id == args.node_id:
                neighbor_weights[nbr_id] = client.get_weights()
                neighbor_sims[nbr_id] = sim

        if neighbor_weights:
            new_w = pull_aggregator.aggregate(
                client_weights=client.get_weights(),
                neighbor_weights=neighbor_weights,
                similarities=neighbor_sims,
            )
            client.set_weights(new_w, round_num=r)

        # Step 6: Evaluation
        eval_loss, eval_acc = client.evaluate(test_loader)
        duration = time.time() - round_t0

        node_logger.info(
            f"Round {r:2d} Done | Test Acc: {eval_acc:6.2f}% | Test Loss: {eval_loss:6.4f} | "
            f"Retrieved Peers: {len(neighbor_weights)} | Time: {duration:.2f}s"
        )

    node_logger.info("=" * 60)
    final_loss, final_acc = client.evaluate(test_loader)
    node_logger.info(f"Simulation Complete on Node {args.node_id} | Final Test Accuracy: {final_acc:.2f}%")
    node_logger.info("=" * 60)

    transport.stop()


if __name__ == "__main__":
    main()
