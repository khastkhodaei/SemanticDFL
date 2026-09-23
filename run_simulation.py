"""
Configuration-Driven Runner for SemanticDFL Simulations and Distributed Network Execution.

Parses YAML configuration files, configures datasets, clients, overlay topology,
and executes decentralized federated learning rounds in either:
1. In-Memory Simulation mode (ultra-fast zero-socket broker)
2. Distributed gRPC Network mode (real TCP/IP communication and multi-process orchestration)
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import random
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from semantic_dfl.aggregation.pull_aggregator import PullAggregator
from semantic_dfl.aggregation.pull_aggregator import PullAggregator
from semantic_dfl.core.models import (
    AlbertTextClassifier,
    FashionCNN,
    GoogleSpeechResNet,
    MLP2NN,
    ResNet18Small,
    build_model,
)
from semantic_dfl.data.datasets import load_partitioned_dataset
from semantic_dfl.data.partitioning import (
    DirichletPartitioner,
    IIDPartitioner,
    PathologicalPartitioner,
)
from semantic_dfl.federated.client import DFLClient
from semantic_dfl.federated.coordinator import SemanticDFLCoordinator
from semantic_dfl.overlay.son import SemanticOverlayNetwork
from semantic_dfl.overlay.topology import (
    build_network_topology,
    compute_average_degree,
)
from semantic_dfl.overlay.zone import ZoneManager
from semantic_dfl.transport.factory import create_transport
from semantic_dfl.visualization.plotter import generate_all_plots


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads and validates YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


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
    dname = dataset_name.lower()
    is_rgb = dname in ("cifar10", "cifar-10", "tiny_imagenet", "tinyimagenet")
    in_channels = 3 if is_rgb else 1

    if arch in ("FashionCNN", "fashion_cnn", "CNN"):
        return lambda: FashionCNN(in_channels=in_channels, num_classes=10)
    elif arch == "MLP2NN":
        input_dim = 3072 if is_rgb else 784
        return lambda: MLP2NN(input_dim=input_dim, num_classes=10)
    elif arch in ("ResNet18Small", "ResNet18", "resnet18_small"):
        num_classes = 200 if dname in ("tiny_imagenet", "tinyimagenet") else 10
        return lambda: ResNet18Small(in_channels=in_channels, num_classes=num_classes)
    elif arch in ("GoogleSpeechResNet", "SpeechResNet", "speech_resnet", "speech"):
        return lambda: GoogleSpeechResNet(in_channels=1, num_classes=35)
    elif arch in ("AlbertTextClassifier", "Albert", "TextClassifier", "20newsgroups"):
        return lambda: AlbertTextClassifier(num_classes=20)
    else:
        return lambda: build_model(model_name=arch, in_channels=in_channels)


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


def run_in_memory_simulation(
    config: Dict[str, Any],
    args: argparse.Namespace,
    logger: logging.Logger,
    timestamp_str: str,
) -> Dict[str, Any]:
    """Executes high-performance centralized In-Memory simulation."""
    sim_cfg = config.get("simulation", {})
    fed_cfg = config.get("federated", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    log_cfg = config.get("logging", {})

    rounds = args.rounds if args.rounds is not None else sim_cfg.get("rounds", 10)
    seed = sim_cfg.get("seed", 42)
    device_cfg = sim_cfg.get("device", "auto")
    if device_cfg == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device_cfg
    device = torch.device(device_str)

    num_clients = args.num_clients if args.num_clients is not None else fed_cfg.get("num_clients", 10)
    k_neighbors = fed_cfg.get("k_neighbors", 2)
    zone_size = fed_cfg.get("zone_size", 5)
    alpha_personalization = fed_cfg.get("alpha_personalization", 0.67)

    dataset_name = data_cfg.get("dataset", "fmnist")
    partitioner_type = data_cfg.get("partitioner", "dirichlet").lower()
    alpha_dirichlet = data_cfg.get("alpha", 0.3)
    batch_size = data_cfg.get("batch_size", 32)
    data_root = data_cfg.get("data_dir", "./Datasets")

    architecture = model_cfg.get("architecture", "FashionCNN")
    local_epochs = model_cfg.get("local_epochs", 1)
    learning_rate = model_cfg.get("learning_rate", 0.01)

    save_results_json = log_cfg.get("save_results_json", True)
    detailed_client_metrics = log_cfg.get("detailed_client_metrics", True)
    generate_plots = log_cfg.get("generate_plots", True)
    output_dir = log_cfg.get("output_dir", "results")

    seeds_cfg = config.get("seeds", {})
    topology_seed = seeds_cfg.get("topology_seed", seed)
    data_seed = seeds_cfg.get("data_seed", seed)
    vrf_seed = seeds_cfg.get("vrf_seed", seed)
    model_seed = seeds_cfg.get("model_seed", seed)

    net_cfg = config.get("network", {})
    dynamic_topology = net_cfg.get("dynamic_topology", False)
    if dynamic_topology:
        topology_seed = random.randint(1, 1000000)

    set_seed(seed)

    logger.info("=" * 65)
    logger.info("       SemanticDFL Simulation (In-Memory Execution)")
    logger.info("=" * 65)
    logger.info(f"Dataset:           {dataset_name.upper()} (Partitioner: {partitioner_type}, alpha={alpha_dirichlet})")
    logger.info(f"Model Architecture:{architecture} (Local Epochs: {local_epochs}, LR: {learning_rate})")
    logger.info(f"Federated Network: {num_clients} clients, k={k_neighbors}, zone_size={zone_size}, alpha_p={alpha_personalization}")
    logger.info(f"Seeds:             Topo={topology_seed} (Dynamic={dynamic_topology}), Data={data_seed}, VRF={vrf_seed}, Model={model_seed}")
    logger.info(f"Simulation:        {rounds} rounds, global_seed={seed}, device={device}")
    logger.info("=" * 65)

    # Configure data partitioner with data_seed
    if partitioner_type == "dirichlet":
        partitioner = DirichletPartitioner(alpha=alpha_dirichlet, seed=data_seed)
    elif partitioner_type == "pathological":
        num_classes = data_cfg.get("num_classes_per_client", 2)
        partitioner = PathologicalPartitioner(num_classes_per_client=num_classes, seed=data_seed)
    elif partitioner_type == "iid":
        partitioner = IIDPartitioner(seed=data_seed)
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

    model_fn = create_model_factory(architecture, dataset_name)

    # Create DFL clients with In-Memory Transport & model_seed
    torch.manual_seed(model_seed)
    clients: Dict[int, DFLClient] = {}
    for client_id, train_loader in client_loaders.items():
        model = model_fn().to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        transport = create_transport(mode="simulation", node_id=client_id)

        client = DFLClient(
            node_id=client_id,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            lr=learning_rate,
            device=device,
            transport=transport,
        )
        clients[client_id] = client

    topology_type = net_cfg.get("topology") or fed_cfg.get("topology", "ring")
    topology_degree = net_cfg.get("degree") or fed_cfg.get("degree", 4)

    # Build physical network topology
    adjacency_graph = build_network_topology(
        topology_type=topology_type,
        num_nodes=num_clients,
        degree=topology_degree,
        seed=topology_seed,
    )
    avg_degree = compute_average_degree(adjacency_graph)

    # Export topology graph JSON
    topo_graph_path = os.path.join(output_dir, "topology_graph.json")
    from semantic_dfl.overlay.topology import export_topology_graph_json
    export_topology_graph_json(adjacency_graph=adjacency_graph, output_path=topo_graph_path)

    zone_manager = ZoneManager(max_zone_size=zone_size, seed=vrf_seed)
    son = SemanticOverlayNetwork(zone_manager=zone_manager)
    pull_aggregator = PullAggregator(default_alpha=alpha_personalization)

    coordinator = SemanticDFLCoordinator(
        clients=clients,
        k_neighbors=k_neighbors,
        adjacency_graph=adjacency_graph,
        pull_aggregator=pull_aggregator,
        son=son,
        global_test_loader=test_loader,
        device=device,
        verbose=True,
    )

    logger.info(f"Physical Topology: {topology_type.upper()} (Nodes: {num_clients}, Avg Degree: {avg_degree:.2f})")
    round_history: List[Dict[str, Any]] = []
    start_time = time.time()

    with coordinator:
        logger.info(f"Starting {rounds} decentralized federated learning rounds...")
        for r in range(1, rounds + 1):
            round_start = time.time()
            metrics = coordinator.step_round(epochs_per_round=local_epochs)
            round_duration = time.time() - round_start

            metrics["duration_seconds"] = round_duration
            round_history.append(metrics)

            timings = metrics.get("timings", {})
            traffic = metrics.get("traffic", {})
            roles = metrics.get("role_distribution", {})
            
            timing_str = (
                f"[Train: {timings.get('time_local_training', 0.0):.2f}s, "
                f"Sig: {timings.get('time_signatures', 0.0):.2f}s, "
                f"SON: {timings.get('time_topology', 0.0):.2f}s, "
                f"Search: {timings.get('time_search', 0.0):.2f}s, "
                f"Agg: {timings.get('time_aggregation', 0.0):.2f}s, "
                f"Eval: {timings.get('time_evaluation', 0.0):.2f}s]"
            )
            traffic_str = (
                f"Traffic: [Ctrl: {traffic.get('control_mb', 0.0):.2f}MB, "
                f"Data: {traffic.get('data_mb', 0.0):.2f}MB, "
                f"Total: {traffic.get('total_mb', 0.0):.2f}MB]"
            )
            roles_str = (
                f"Roles: [Init: {roles.get('ZONE_INITIATOR', 0)}, "
                f"Super: {roles.get('SUPER_PEER', 0)}, "
                f"Replica: {roles.get('REPLICA_SUPER_PEER', 0)}, "
                f"Leaf: {roles.get('LEAF_PEER', 0)}]"
            )

            logger.info(
                f"Round {metrics['round']:3d}/{rounds:3d} | "
                f"Test Acc: {metrics['avg_test_accuracy']:6.2f}% | "
                f"Train Loss: {metrics['avg_train_loss']:6.4f} | "
                f"Recall@K: {metrics['avg_recall_at_k']:5.2f} | "
                f"Card P: {metrics['cardinality_p']:4d} | "
                f"Time: {round_duration:.2f}s"
            )
            logger.info(f"  --> {timing_str} | {traffic_str}")
            logger.info(f"  --> {roles_str}")

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

    results_data = {
        "timestamp": timestamp_str,
        "config": config,
        "mode": "simulation",
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

    if save_results_json:
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

        if generate_plots:
            plots_dir = os.path.join(output_dir, "plots")
            logger.info(f"Generating academic figures in: {plots_dir}")
            generate_all_plots(data=results_data, output_dir=plots_dir, export_pdf=True)
            logger.info(f"[Visualization] Academic figures successfully generated in: {plots_dir}")

    return results_data


def stream_process_output(pipe: Any, prefix: str, logger: logging.Logger) -> None:
    """Reads lines asynchronously from a child process stdout/stderr pipe."""
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            clean_line = line.rstrip()
            if clean_line:
                logger.info(f"{prefix} {clean_line}")
    except Exception:
        pass
    finally:
        pipe.close()


def run_distributed_execution(
    config: Dict[str, Any],
    args: argparse.Namespace,
    logger: logging.Logger,
    timestamp_str: str,
) -> Dict[str, Any]:
    """Spins up independent distributed peer instances communicating over gRPC network sockets."""
    exec_cfg = config.get("execution", {})
    dist_cfg = exec_cfg.get("distributed") or {}
    net_cfg = config.get("network", {})
    slices_cfg = config.get("slices", {})
    fed_cfg = config.get("federated", {})
    sim_cfg = config.get("simulation", {})
    log_cfg = config.get("logging", {})

    host = str(dist_cfg.get("host") or net_cfg.get("host", "127.0.0.1"))
    base_port = int(dist_cfg.get("base_port") or net_cfg.get("base_port", 50050))
    spawn_subprocesses = exec_cfg.get(
        "spawn_subprocesses",
        dist_cfg.get("spawn_subprocesses", True)
    )
    num_clients = (
        args.num_clients
        if args.num_clients is not None
        else (fed_cfg.get("num_clients") or slices_cfg.get("num_nodes", 10))
    )
    rounds = (
        args.rounds
        if args.rounds is not None
        else (sim_cfg.get("rounds") or fed_cfg.get("rounds", 10))
    )
    output_dir = log_cfg.get("output_dir", "results")

    logger.info("=" * 65)
    logger.info("       SemanticDFL Distributed Network Execution (gRPC)")
    logger.info("=" * 65)
    logger.info(f"Host:              {host}")
    logger.info(f"Base Port:         {base_port} (Ports: {base_port} to {base_port + num_clients - 1})")
    logger.info(f"Total Peers:       {num_clients}")
    logger.info(f"Total Rounds:      {rounds}")
    logger.info(f"Subprocess Mode:   {spawn_subprocesses}")
    if slices_cfg:
        logger.info(f"SLICES Network:    Interface={slices_cfg.get('network_interface')}, Discovery={slices_cfg.get('peer_discovery')}")
        logger.info(f"Super-Peers:       {slices_cfg.get('super_peers', [])}")
    logger.info("=" * 65)

    if not spawn_subprocesses:
        # In-process distributed socket execution
        logger.info("Running in-process distributed socket execution...")
        return run_in_memory_simulation(config, args, logger, timestamp_str)

    # Subprocess-driven distributed orchestration
    processes: List[subprocess.Popen] = []
    threads: List[threading.Thread] = []
    start_time = time.time()

    def cleanup():
        if processes:
            logger.info("Stopping distributed peer subprocesses...")
            for p in processes:
                if p.poll() is None:
                    try:
                        p.terminate()
                    except Exception:
                        pass
            time.sleep(0.5)
            for p in processes:
                if p.poll() is None:
                    try:
                        p.kill()
                    except Exception:
                        pass

    atexit.register(cleanup)

    def signal_handler(sig, frame):
        logger.warning(f"Received signal {sig}. Terminating distributed cluster...")
        cleanup()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        bootstrap_address = f"{host}:{base_port}"

        # 1. Start Bootstrap Leader (Node 0)
        logger.info(f"Spawning Bootstrap Leader [Peer-0] on port {base_port}...")
        cmd_leader = [
            sys.executable,
            "run_node.py",
            "--node_id", "0",
            "--host", host,
            "--port", str(base_port),
            "--is_super_peer",
            "--rounds", str(rounds),
            "--config", args.config,
        ]

        proc_leader = subprocess.Popen(
            cmd_leader,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.getcwd(),
        )
        processes.append(proc_leader)

        t_leader = threading.Thread(
            target=stream_process_output,
            args=(proc_leader.stdout, "[Peer-0]", logger),
            daemon=True,
        )
        t_leader.start()
        threads.append(t_leader)

        # Allow leader socket to bind
        time.sleep(1.5)

        # 2. Start Worker Peers (Nodes 1 .. N-1)
        for nid in range(1, num_clients):
            port_i = base_port + nid
            logger.info(f"Spawning Distributed Worker [Peer-{nid}] on port {port_i} (Bootstrap: {bootstrap_address})...")
            cmd_worker = [
                sys.executable,
                "run_node.py",
                "--node_id", str(nid),
                "--host", host,
                "--port", str(port_i),
                "--bootstrap", bootstrap_address,
                "--rounds", str(rounds),
                "--config", args.config,
            ]

            proc_worker = subprocess.Popen(
                cmd_worker,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=os.getcwd(),
            )
            processes.append(proc_worker)

            t_worker = threading.Thread(
                target=stream_process_output,
                args=(proc_worker.stdout, f"[Peer-{nid}]", logger),
                daemon=True,
            )
            t_worker.start()
            threads.append(t_worker)

        logger.info(f"All {num_clients} distributed peer subprocesses running. Awaiting completion...")

        # Monitor processes
        exit_codes = {}
        while len(exit_codes) < len(processes):
            for idx, p in enumerate(processes):
                if idx not in exit_codes and p.poll() is not None:
                    exit_codes[idx] = p.returncode
                    if p.returncode == 0:
                        logger.info(f"[Peer-{idx}] exited cleanly with code 0.")
                    else:
                        logger.error(f"[Peer-{idx}] exited with error code {p.returncode}.")
            time.sleep(0.5)

        for t in threads:
            t.join(timeout=2.0)

        total_duration = time.time() - start_time
        all_clean = all(code == 0 for code in exit_codes.values())

        if all_clean:
            logger.info(f"Distributed execution completed successfully across all {num_clients} peers in {total_duration:.2f}s.")
        else:
            logger.warning(f"Distributed execution completed with non-zero exit codes: {exit_codes}")

        results_data = {
            "timestamp": timestamp_str,
            "config": config,
            "mode": "distributed",
            "total_duration_seconds": total_duration,
            "num_clients": num_clients,
            "exit_codes": exit_codes,
            "status": "success" if all_clean else "failed",
        }

        dist_results_path = os.path.join(output_dir, f"distributed_results_{timestamp_str}.json")
        with open(dist_results_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)

        return results_data

    except Exception as e:
        logger.error(f"Distributed execution error: {e}", exc_info=True)
        cleanup()
        raise


def main():
    parser = argparse.ArgumentParser(description="Unified Execution Runner for SemanticDFL")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["simulation", "distributed", "slices_distributed"],
        default=None,
        help="Override execution mode ('simulation', 'distributed', 'slices_distributed')",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Override total communication rounds",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=None,
        help="Override total number of federated clients/peers",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    exec_cfg = config.get("execution", {})
    exec_mode = (args.mode or exec_cfg.get("mode", "simulation")).strip().lower()

    log_cfg = config.get("logging", {})
    output_dir = log_cfg.get("output_dir", "results")

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger, log_filepath = setup_logger(output_dir, timestamp_str)

    logger.info("=" * 65)
    logger.info("           SemanticDFL Unified Execution Runner")
    logger.info("=" * 65)
    logger.info(f"Config File:       {args.config}")
    logger.info(f"Execution Mode:    {exec_mode.upper()}")
    logger.info(f"Log Transcript:    {log_filepath}")
    logger.info("=" * 65)

    if exec_mode in ("simulation", "memory", "in_memory"):
        run_in_memory_simulation(config, args, logger, timestamp_str)
    elif exec_mode in ("distributed", "slices_distributed", "slices", "grpc"):
        run_distributed_execution(config, args, logger, timestamp_str)
    else:
        raise ValueError(
            f"Unsupported execution mode '{exec_mode}'. Options: 'simulation', 'distributed', 'slices_distributed'."
        )


if __name__ == "__main__":
    main()
