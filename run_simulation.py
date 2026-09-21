"""
Configuration-Driven Runner for SemanticDFL Simulations.

Parses YAML configuration files, configures datasets, clients, overlay topology,
and executes decentralized federated learning rounds with metric logging and reporting.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from semantic_dfl.aggregation.pull_aggregator import PullAggregator
from semantic_dfl.core.models import FashionCNN, MLP2NN, ResNet18Small
from semantic_dfl.data.datasets import load_partitioned_dataset
from semantic_dfl.data.partitioning import (
    DirichletPartitioner,
    IIDPartitioner,
    PathologicalPartitioner,
)
from semantic_dfl.federated.client import DFLClient
from semantic_dfl.federated.coordinator import SemanticDFLCoordinator
from semantic_dfl.overlay.son import SemanticOverlayNetwork
from semantic_dfl.overlay.zone import ZoneManager


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads and validates YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model_factory(architecture: str, dataset_name: str):
    """Returns a model constructor matching the specified architecture and dataset."""
    arch = architecture.strip()
    is_rgb = dataset_name.lower() in ("cifar10", "cifar-10")
    in_channels = 3 if is_rgb else 1

    if arch == "FashionCNN":
        return lambda: FashionCNN(in_channels=in_channels, num_classes=10)
    elif arch == "MLP2NN":
        input_dim = 3072 if is_rgb else 784
        return lambda: MLP2NN(input_dim=input_dim, num_classes=10)
    elif arch in ("ResNet18Small", "ResNet18"):
        return lambda: ResNet18Small(in_channels=in_channels, num_classes=10)
    else:
        raise ValueError(f"Unsupported model architecture: {architecture}")


def setup_logger(output_dir: str, timestamp_str: str) -> tuple[logging.Logger, str]:
    """Configures multi-handler logger writing to console and persistent log file."""
    os.makedirs(output_dir, exist_ok=True)
    log_filename = f"simulation_{timestamp_str}.log"
    log_filepath = os.path.join(output_dir, log_filename)

    logger = logging.getLogger("SemanticDFL")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # File Handler
    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Stream Handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger, log_filepath


def main():
    parser = argparse.ArgumentParser(description="Run YAML Config-Driven SemanticDFL Simulation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    sim_cfg = config.get("simulation", {})
    fed_cfg = config.get("federated", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    log_cfg = config.get("logging", {})

    # Extract simulation parameters
    rounds = sim_cfg.get("rounds", 10)
    seed = sim_cfg.get("seed", 42)
    device_cfg = sim_cfg.get("device", "auto")
    if device_cfg == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device_cfg
    device = torch.device(device_str)

    # Extract federated parameters
    num_clients = fed_cfg.get("num_clients", 10)
    k_neighbors = fed_cfg.get("k_neighbors", 2)
    zone_size = fed_cfg.get("zone_size", 5)
    alpha_personalization = fed_cfg.get("alpha_personalization", 0.67)

    # Extract data parameters
    dataset_name = data_cfg.get("dataset", "fmnist")
    partitioner_type = data_cfg.get("partitioner", "dirichlet").lower()
    alpha_dirichlet = data_cfg.get("alpha", 0.3)
    batch_size = data_cfg.get("batch_size", 32)
    data_root = data_cfg.get("data_dir", "./Datasets")

    # Extract model parameters
    architecture = model_cfg.get("architecture", "FashionCNN")
    local_epochs = model_cfg.get("local_epochs", 1)
    learning_rate = model_cfg.get("learning_rate", 0.01)

    # Extract logging parameters
    save_results_json = log_cfg.get("save_results_json", True)
    detailed_client_metrics = log_cfg.get("detailed_client_metrics", True)
    output_dir = log_cfg.get("output_dir", "results")

    # Setup logger
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger, log_filepath = setup_logger(output_dir, timestamp_str)

    # Set seed
    set_seed(seed)

    logger.info("=" * 65)
    logger.info("           SemanticDFL Simulation Runner")
    logger.info("=" * 65)
    logger.info(f"Config File:       {args.config}")
    logger.info(f"Dataset:           {dataset_name.upper()} (Partitioner: {partitioner_type}, alpha={alpha_dirichlet})")
    logger.info(f"Model Architecture:{architecture} (Local Epochs: {local_epochs}, LR: {learning_rate})")
    logger.info(f"Federated Network: {num_clients} clients, k={k_neighbors}, zone_size={zone_size}, alpha_p={alpha_personalization}")
    logger.info(f"Simulation:        {rounds} rounds, seed={seed}, device={device}")
    logger.info(f"Log Transcript:    {log_filepath}")
    logger.info("=" * 65)

    # Configure data partitioner
    if partitioner_type == "dirichlet":
        partitioner = DirichletPartitioner(alpha=alpha_dirichlet, seed=seed)
    elif partitioner_type == "pathological":
        num_classes = data_cfg.get("num_classes_per_client", 2)
        partitioner = PathologicalPartitioner(num_classes_per_client=num_classes, seed=seed)
    elif partitioner_type == "iid":
        partitioner = IIDPartitioner(seed=seed)
    else:
        raise ValueError(f"Unsupported partitioner type: {partitioner_type}")

    # Load partitioned dataset
    logger.info("Loading dataset and creating client partitions...")
    client_loaders, test_loader = load_partitioned_dataset(
        dataset_name=dataset_name,
        partitioner=partitioner,
        num_clients=num_clients,
        batch_size=batch_size,
        root=data_root,
        download=True,
    )

    # Initialize model factory
    model_fn = create_model_factory(architecture, dataset_name)

    # Create DFL clients
    clients: Dict[int, DFLClient] = {}
    for client_id, train_loader in client_loaders.items():
        model = model_fn().to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        client = DFLClient(
            node_id=client_id,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            lr=learning_rate,
            device=device,
        )
        clients[client_id] = client

    # Initialize overlay network with zone size
    zone_manager = ZoneManager(max_zone_size=zone_size, seed=seed)
    son = SemanticOverlayNetwork(zone_manager=zone_manager)
    pull_aggregator = PullAggregator(default_alpha=alpha_personalization)

    # Initialize coordinator
    coordinator = SemanticDFLCoordinator(
        clients=clients,
        k_neighbors=k_neighbors,
        pull_aggregator=pull_aggregator,
        son=son,
        global_test_loader=test_loader,
        device=device,
        verbose=True,
    )

    # Run federated rounds
    logger.info(f"Starting {rounds} decentralized federated learning rounds...")
    round_history = []
    start_time = time.time()

    for r in range(1, rounds + 1):
        round_start = time.time()
        metrics = coordinator.step_round(epochs_per_round=local_epochs)
        round_duration = time.time() - round_start

        metrics["duration_seconds"] = round_duration
        round_history.append(metrics)

        timings = metrics.get("timings", {})
        timing_str = (
            f"[Train: {timings.get('time_local_training', 0.0):.2f}s, "
            f"Sig: {timings.get('time_signatures', 0.0):.2f}s, "
            f"SON: {timings.get('time_topology', 0.0):.2f}s, "
            f"Search: {timings.get('time_search', 0.0):.2f}s, "
            f"Agg: {timings.get('time_aggregation', 0.0):.2f}s, "
            f"Eval: {timings.get('time_evaluation', 0.0):.2f}s]"
        )

        logger.info(
            f"Round {metrics['round']:3d}/{rounds:3d} | "
            f"Test Acc: {metrics['avg_test_accuracy']:6.2f}% | "
            f"Train Loss: {metrics['avg_train_loss']:6.4f} | "
            f"Test Loss: {metrics['avg_test_loss']:6.4f} | "
            f"Recall@K: {metrics['avg_recall_at_k']:5.2f} | "
            f"Card P: {metrics['cardinality_p']:4d} | "
            f"Time: {round_duration:.2f}s | {timing_str}"
        )

    total_duration = time.time() - start_time
    logger.info(f"Simulation finished in {total_duration:.2f}s.")

    # Final evaluation
    logger.info("=" * 65)
    logger.info("                    Final Evaluation")
    logger.info("=" * 65)
    final_client_accuracies = {}
    final_client_losses = {}
    for cid, client in clients.items():
        loss, acc = client.evaluate(test_loader)
        final_client_accuracies[cid] = acc
        final_client_losses[cid] = loss

    acc_values = list(final_client_accuracies.values())
    mean_acc = float(np.mean(acc_values))
    std_acc = float(np.std(acc_values))
    min_acc = float(np.min(acc_values))
    max_acc = float(np.max(acc_values))

    logger.info(f"Mean Test Accuracy:   {mean_acc:.2f}% +/- {std_acc:.2f}%")
    logger.info(f"Accuracy Range:       [{min_acc:.2f}%, {max_acc:.2f}%]")
    logger.info(f"Best Final Peer:      {max_acc:.2f}%")
    logger.info("=" * 65)

    # Save summary metrics to output_dir
    if save_results_json:
        results_data = {
            "timestamp": timestamp_str,
            "config": config,
            "total_duration_seconds": total_duration,
            "summary": {
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "min_accuracy": min_acc,
                "max_accuracy": max_acc,
                "final_round_train_loss": round_history[-1]["avg_train_loss"] if round_history else None,
                "final_round_test_loss": round_history[-1]["avg_test_loss"] if round_history else None,
            },
            "per_client_final_accuracy": final_client_accuracies if detailed_client_metrics else None,
            "per_client_final_loss": final_client_losses if detailed_client_metrics else None,
            "round_history": round_history,
        }

        json_filename = f"simulation_results_{timestamp_str}.json"
        json_path = os.path.join(output_dir, json_filename)
        latest_path = os.path.join(output_dir, "latest_results.json")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)

        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)

        logger.info(f"[Logging] Results JSON successfully saved to:")
        logger.info(f"  - {json_path}")
        logger.info(f"  - {latest_path}")


if __name__ == "__main__":
    main()
