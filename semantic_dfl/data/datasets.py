"""
Dataset Loaders and Federated DataLoader Orchestration for SemanticDFL.

Provides clean interfaces for FashionMNIST and CIFAR-10, supporting automatic downloads,
standard normalization pipelines, and seamless subset construction from partitioner indices.
"""

from __future__ import annotations

import os
from typing import Sequence, Union
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from semantic_dfl.data.partitioning import BasePartitioner, DirichletPartitioner


def get_fashion_mnist(
    root: str = "./Datasets/fmnist",
    download: bool = True,
    transform: transforms.Compose | None = None,
) -> tuple[Dataset, Dataset]:
    """
    Loads FashionMNIST training and test datasets with standard normalization.
    
    Args:
        root: Storage directory for dataset files.
        download: Whether to download if not present locally.
        transform: Optional custom PyTorch torchvision transforms.
        
    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    os.makedirs(root, exist_ok=True)
    train_dataset = datasets.FashionMNIST(root=root, train=True, download=download, transform=transform)
    test_dataset = datasets.FashionMNIST(root=root, train=False, download=download, transform=transform)
    return train_dataset, test_dataset


def get_cifar10(
    root: str = "./Datasets/cifar10",
    download: bool = True,
    train_transform: transforms.Compose | None = None,
    test_transform: transforms.Compose | None = None,
) -> tuple[Dataset, Dataset]:
    """
    Loads CIFAR-10 training and test datasets with standard normalization.
    
    Args:
        root: Storage directory for dataset files.
        download: Whether to download if not present locally.
        train_transform: Optional custom training transform.
        test_transform: Optional custom test transform.
        
    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    os.makedirs(root, exist_ok=True)
    train_dataset = datasets.CIFAR10(root=root, train=True, download=download, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=download, transform=test_transform)
    return train_dataset, test_dataset


def create_client_dataloaders(
    dataset: Dataset,
    client_indices: Union[Sequence[int], dict[int, list[int]]],
    batch_size: int = 64,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
) -> Union[DataLoader, dict[int, DataLoader]]:
    """
    Constructs PyTorch DataLoader(s) from dataset sample index subsets.
    
    Args:
        dataset: Base PyTorch Dataset.
        client_indices: Single list of indices OR dictionary of {client_id: [indices]}.
        batch_size: Mini-batch size.
        shuffle: Whether to randomize sample ordering per epoch.
        drop_last: Whether to discard non-full final batches.
        num_workers: Subprocess data loading workers.
        
    Returns:
        Single DataLoader (if list given) or dictionary of {client_id: DataLoader}.
    """
    if isinstance(client_indices, dict):
        loaders: dict[int, DataLoader] = {}
        for client_id, indices in client_indices.items():
            subset = Subset(dataset, indices)
            loaders[client_id] = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=num_workers,
            )
        return loaders
    else:
        subset = Subset(dataset, list(client_indices))
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )


def load_partitioned_dataset(
    dataset_name: str = "fmnist",
    partitioner: BasePartitioner | None = None,
    num_clients: int = 10,
    batch_size: int = 64,
    root: str = "./Datasets",
    download: bool = True,
) -> tuple[dict[int, DataLoader], DataLoader]:
    """
    Convenience end-to-end pipeline loading a dataset and partitioning it among N clients.
    
    Args:
        dataset_name: 'fmnist' or 'cifar10'.
        partitioner: Partitioner instance (defaults to DirichletPartitioner(alpha=0.5)).
        num_clients: Number of simulated clients.
        batch_size: Batch size for client DataLoaders.
        root: Root storage folder.
        download: Auto-download flag.
        
    Returns:
        Tuple of (dict_of_client_train_loaders, global_test_loader).
    """
    if partitioner is None:
        partitioner = DirichletPartitioner(alpha=0.5, seed=42)

    name = dataset_name.lower()
    if name in ("fmnist", "fashionmnist", "fashion_mnist"):
        train_set, test_set = get_fashion_mnist(root=os.path.join(root, "fmnist"), download=download)
    elif name in ("cifar10", "cifar-10"):
        train_set, test_set = get_cifar10(root=os.path.join(root, "cifar10"), download=download)
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}. Expected 'fmnist' or 'cifar10'.")

    # Extract training targets
    if hasattr(train_set, "targets"):
        targets = train_set.targets
    else:
        targets = [train_set[i][1] for i in range(len(train_set))]

    client_splits = partitioner.partition(targets=targets, num_clients=num_clients)
    client_loaders = create_client_dataloaders(train_set, client_splits, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return client_loaders, test_loader
