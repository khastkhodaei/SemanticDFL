"""
End-to-End Simulation Script for SemanticDFL on Fashion-MNIST.

Demonstrates decentralized federated learning with semantic peer discovery,
proximity-bounded zone clustering, and adaptive personalized pull aggregation.
"""

from __future__ import annotations

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from semantic_dfl.core.models import FashionCNN, MLP2NN, ResNet18Small
from semantic_dfl.data.datasets import load_partitioned_dataset
from semantic_dfl.data.partitioning import DirichletPartitioner, PathologicalPartitioner, IIDPartitioner
from semantic_dfl.federated.client import DFLClient
from semantic_dfl.federated.coordinator import SemanticDFLCoordinator
from semantic_dfl.overlay.zone import ZoneManager
from semantic_dfl.overlay.son import SemanticOverlayNetwork


def parse_args():
    parser = argparse.ArgumentParser(description="SemanticDFL Fashion-MNIST Simulation")
    parser.add_argument("--model", type=str, default="FashionCNN", choices=["FashionCNN", "MLP2NN", "ResNet18Small"], help="Model architecture")
    parser.add_argument("--num_clients", type=int, default=20, help="Total number of peer clients")
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated communication rounds")
    parser.add_argument("--local_epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--batch_size", type=int, default=32, help="Client training batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Client local learning rate")
    parser.add_argument("--partition", type=str, default="dirichlet", choices=["dirichlet", "pathological", "iid"], help="Data partition strategy")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration parameter")
    parser.add_argument("--tau_min", type=float, default=0.7, help="Minimum signature fidelity threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    print("=== Starting SemanticDFL Simulation on Fashion-MNIST ===")
    print(f"Model: {args.model} | Clients: {args.num_clients} | Rounds: {args.rounds} | Partition: {args.partition} (alpha={args.alpha}) | Device: {args.device}")

    # Partition dataset
    if args.partition == "dirichlet":
        partitioner = DirichletPartitioner(alpha=args.alpha, seed=args.seed)
    elif args.partition == "pathological":
        partitioner = PathologicalPartitioner(num_classes_per_client=2, seed=args.seed)
    else:
        partitioner = IIDPartitioner(seed=args.seed)

    client_loaders, test_loader = load_partitioned_dataset(
        dataset_name="fmnist",
        partitioner=partitioner,
        num_clients=args.num_clients,
        batch_size=args.batch_size,
        root="./Datasets",
        download=True,
    )

    # Initialize model template
    if args.model == "FashionCNN":
        model_fn = lambda: FashionCNN(in_channels=1, num_classes=10)
    elif args.model == "MLP2NN":
        model_fn = lambda: MLP2NN(input_dim=784, num_classes=10)
    else:
        model_fn = lambda: ResNet18Small(in_channels=1, num_classes=10)

    device = torch.device(args.device)

    # Create clients
    clients = {}
    for client_id, train_loader in client_loaders.items():
        model = model_fn().to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        client = DFLClient(
            node_id=client_id,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            lr=args.lr,
            device=device,
        )
        clients[client_id] = client

    # Initialize coordinator
    coordinator = SemanticDFLCoordinator(
        clients=clients,
        k_neighbors=2,
        tau_min_fidelity=args.tau_min,
        global_test_loader=test_loader,
    )

    print("\nRunning decentralized federated rounds...")
    for round_num in range(1, args.rounds + 1):
        round_metrics = coordinator.step_round(epochs_per_round=args.local_epochs)
        print(
            f"Round {round_num:2d}/{args.rounds:2d} | "
            f"Avg Test Acc: {round_metrics['avg_test_accuracy']:.2f}% | "
            f"Avg Train Loss: {round_metrics['avg_train_loss']:.4f} | "
            f"Cardinality P: {round_metrics['cardinality_p']}"
        )

    print("\nSimulation complete! Final Evaluation across all peers:")
    final_accs = [c.evaluate(test_loader)[1] for c in clients.values()]
    print(f"Mean Accuracy: {np.mean(final_accs):.2f}% +/- {np.std(final_accs):.2f}%")


if __name__ == "__main__":
    main()
