#!/usr/bin/env python3
"""
Dynamic Docker-Compose Generator for SemanticDFL Multi-Container Clusters.

Generates reproducible, multi-node docker-compose definitions on a dedicated
isolated bridge network with per-container environment configuration,
sequential port exposure, and log streaming.

Usage:
    python scripts/generate_compose.py --num_nodes 10 --output docker-compose.yml
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional
import yaml


def generate_compose_dict(
    num_nodes: int = 10,
    base_port: int = 50050,
    rounds: int = 10,
    config_path: str = "configs/default_config.yaml",
    network_name: str = "semantic_dfl_net",
    subnet: str = "172.28.0.0/16",
    ip_prefix: str = "172.28.0",
    data_dir: str = "./Datasets",
    results_dir: str = "./results",
    logs_dir: str = "./logs",
) -> Dict[str, Any]:
    """
    Construct a complete docker-compose specification dictionary for N nodes.

    Args:
        num_nodes: Number of peer nodes in the cluster (>= 1).
        base_port: Base host port for node 0 (mapped sequentially).
        rounds: Number of federated communication rounds.
        config_path: Path to the YAML configuration file inside the container.
        network_name: Name of the bridge network.
        subnet: CIDR subnet block for the bridge network.
        ip_prefix: IP address prefix (first 3 octets) for static IP assignment.
        data_dir: Host directory mounted to /app/Datasets.
        results_dir: Host directory mounted to /app/results.
        logs_dir: Host directory mounted to /app/logs.

    Returns:
        Dict representing valid docker-compose structure.
    """
    if num_nodes < 1:
        raise ValueError("num_nodes must be at least 1")

    services: Dict[str, Any] = {}
    bootstrap_service_name = "semantic_dfl_node_0"
    bootstrap_hostname = "node-0"
    bootstrap_port = base_port

    for i in range(num_nodes):
        service_name = f"semantic_dfl_node_{i}"
        node_port = base_port + i
        node_ip = f"{ip_prefix}.{10 + i}"

        # Common service configuration
        service_cfg: Dict[str, Any] = {
            "build": {
                "context": ".",
                "dockerfile": "Dockerfile",
            },
            "container_name": service_name,
            "hostname": f"node-{i}",
            "networks": {
                network_name: {
                    "ipv4_address": node_ip,
                }
            },
            "ports": [
                f"{node_port}:{node_port}",
            ],
            "environment": {
                "PYTHONUNBUFFERED": "1",
                "NODE_ID": str(i),
                "NODE_PORT": str(node_port),
            },
            "volumes": [
                f"{results_dir}:/app/results",
                f"{data_dir}:/app/Datasets",
                f"{logs_dir}:/app/logs",
            ],
        }

        if i == 0:
            # Node 0 is the Bootstrap Super-Peer
            service_cfg["command"] = (
                f"--node_id 0 "
                f"--host 0.0.0.0 "
                f"--port {node_port} "
                f"--is_super_peer "
                f"--rounds {rounds} "
                f"--config {config_path}"
            )
            service_cfg["restart"] = "unless-stopped"
        else:
            # Nodes 1..N-1 connect to the Bootstrap node
            service_cfg["depends_on"] = [bootstrap_service_name]
            service_cfg["command"] = (
                f"--node_id {i} "
                f"--host 0.0.0.0 "
                f"--port {node_port} "
                f"--bootstrap {bootstrap_hostname}:{bootstrap_port} "
                f"--rounds {rounds} "
                f"--config {config_path}"
            )
            service_cfg["restart"] = "on-failure"

        services[service_name] = service_cfg

    compose_dict: Dict[str, Any] = {
        "version": "3.8",
        "networks": {
            network_name: {
                "driver": "bridge",
                "ipam": {
                    "config": [
                        {"subnet": subnet}
                    ]
                }
            }
        },
        "services": services,
    }

    return compose_dict


def generate_compose_yaml(
    num_nodes: int = 10,
    base_port: int = 50050,
    rounds: int = 10,
    config_path: str = "configs/default_config.yaml",
    network_name: str = "semantic_dfl_net",
    subnet: str = "172.28.0.0/16",
    ip_prefix: str = "172.28.0",
    data_dir: str = "./Datasets",
    results_dir: str = "./results",
    logs_dir: str = "./logs",
) -> str:
    """Generate serialized YAML string for docker-compose."""
    compose_dict = generate_compose_dict(
        num_nodes=num_nodes,
        base_port=base_port,
        rounds=rounds,
        config_path=config_path,
        network_name=network_name,
        subnet=subnet,
        ip_prefix=ip_prefix,
        data_dir=data_dir,
        results_dir=results_dir,
        logs_dir=logs_dir,
    )
    header = (
        "# ==============================================================================\n"
        f"# SemanticDFL Multi-Node Cluster Configuration ({num_nodes} Nodes)\n"
        "# Generated automatically by scripts/generate_compose.py\n"
        "# Paper: Dogani, Khastkhodaei, Khunjush, Laoutaris (POMACS / ACM SIGMETRICS 2026)\n"
        "# ==============================================================================\n\n"
    )
    return header + yaml.dump(compose_dict, sort_keys=False, default_flow_style=False)


def save_compose_file(
    output_path: str = "docker-compose.yml",
    num_nodes: int = 10,
    base_port: int = 50050,
    rounds: int = 10,
    config_path: str = "configs/default_config.yaml",
    network_name: str = "semantic_dfl_net",
    subnet: str = "172.28.0.0/16",
    ip_prefix: str = "172.28.0",
    data_dir: str = "./Datasets",
    results_dir: str = "./results",
    logs_dir: str = "./logs",
) -> str:
    """Generate and write docker-compose configuration to file."""
    yaml_content = generate_compose_yaml(
        num_nodes=num_nodes,
        base_port=base_port,
        rounds=rounds,
        config_path=config_path,
        network_name=network_name,
        subnet=subnet,
        ip_prefix=ip_prefix,
        data_dir=data_dir,
        results_dir=results_dir,
        logs_dir=logs_dir,
    )
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate Multi-Node Docker Compose Configuration for SemanticDFL"
    )
    parser.add_argument("--num_nodes", type=int, default=10, help="Number of peer nodes to deploy (default: 10)")
    parser.add_argument("--base_port", type=int, default=50050, help="Starting TCP port for node 0 (default: 50050)")
    parser.add_argument("--rounds", type=int, default=10, help="Total federated communication rounds (default: 10)")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Configuration profile path")
    parser.add_argument("--output", type=str, default="docker-compose.yml", help="Target output compose file path")
    parser.add_argument("--network", type=str, default="semantic_dfl_net", help="Docker bridge network name")
    parser.add_argument("--subnet", type=str, default="172.28.0.0/16", help="Subnet CIDR block")
    parser.add_argument("--ip_prefix", type=str, default="172.28.0", help="Subnet 3-octet prefix for static IPs")
    parser.add_argument("--data_dir", type=str, default="./Datasets", help="Host path to datasets directory")
    parser.add_argument("--results_dir", type=str, default="./results", help="Host path to results directory")
    parser.add_argument("--logs_dir", type=str, default="./logs", help="Host path to logs directory")

    args = parser.parse_args()

    out_file = save_compose_file(
        output_path=args.output,
        num_nodes=args.num_nodes,
        base_port=args.base_port,
        rounds=args.rounds,
        config_path=args.config,
        network_name=args.network,
        subnet=args.subnet,
        ip_prefix=args.ip_prefix,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        logs_dir=args.logs_dir,
    )
    print(f"Successfully generated SemanticDFL cluster compose file ({args.num_nodes} nodes) -> {out_file}")


if __name__ == "__main__":
    main()
