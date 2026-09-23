"""
Data Partitioning Algorithms for Federated and Decentralized Learning.

Provides robust, reproducible IID and Non-IID partitioning strategies:
1. DirichletPartitioner: Simulates label distribution skew via Dirichlet distribution Dir(alpha).
2. PathologicalPartitioner: Strict class-quota (shard) partitioner where each client only observes K classes or a fraction (e.g. 20% or 30%) of classes.
3. IIDPartitioner: Uniform balanced random partitioning.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Sequence
import numpy as np
import torch


class BasePartitioner(ABC):
    """Abstract base class for federated dataset partitioners."""

    @abstractmethod
    def partition(
        self,
        targets: Sequence[int] | np.ndarray | torch.Tensor,
        num_clients: int,
    ) -> dict[int, list[int]]:
        """
        Partitions sample indices among num_clients.
        
        Args:
            targets: Array-like sequence of target labels.
            num_clients: Number of participating clients.
            
        Returns:
            Dictionary mapping client_id (0..num_clients-1) to list of sample indices.
        """
        pass


class DirichletPartitioner(BasePartitioner):
    """
    Non-IID Dirichlet Partitioner.
    
    Generates class-imbalanced data distributions for each client by sampling from a
    Dirichlet distribution with concentration parameter alpha:
        p_{c} ~ Dir(alpha, ..., alpha)
    where smaller alpha values produce more severe label heterogeneity.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        min_require_size: int = 1,
        seed: int | None = 42,
    ) -> None:
        if alpha <= 0:
            raise ValueError(f"Concentration parameter alpha must be positive, got {alpha}")
        self.alpha = alpha
        self.min_require_size = min_require_size
        self.seed = seed

    def partition(
        self,
        targets: Sequence[int] | np.ndarray | torch.Tensor,
        num_clients: int,
    ) -> dict[int, list[int]]:
        if num_clients <= 0:
            raise ValueError(f"num_clients must be positive, got {num_clients}")

        if isinstance(targets, torch.Tensor):
            targets_np = targets.detach().cpu().numpy()
        else:
            targets_np = np.asarray(targets)

        num_samples = len(targets_np)
        unique_classes = np.unique(targets_np)
        num_classes = len(unique_classes)

        rng = np.random.default_rng(self.seed)

        # Track indices per class
        class_indices = {c: np.where(targets_np == c)[0] for c in unique_classes}
        for c in unique_classes:
            rng.shuffle(class_indices[c])

        # If min_require_size * num_clients > num_samples, scale min_require_size down
        effective_min_size = min(self.min_require_size, num_samples // num_clients)

        client_indices: dict[int, list[int]] = {i: [] for i in range(num_clients)}
        allocated = False
        max_retries = 100
        retry_count = 0

        while not allocated and retry_count < max_retries:
            retry_count += 1
            temp_client_indices: dict[int, list[int]] = {i: [] for i in range(num_clients)}

            for c in unique_classes:
                idx_c = class_indices[c]
                # Sample proportions from Dirichlet
                proportions = rng.dirichlet(np.repeat(self.alpha, num_clients))
                # Normalize and assign sample counts
                counts = (proportions * len(idx_c)).astype(int)
                remainder = len(idx_c) - counts.sum()
                if remainder > 0:
                    top_clients = np.argsort(-proportions)[:remainder]
                    for cl in top_clients:
                        counts[cl] += 1

                # Split class indices according to counts
                start = 0
                for client_id, count in enumerate(counts):
                    if count > 0:
                        temp_client_indices[client_id].extend(idx_c[start : start + count])
                        start += count

            min_size = min(len(idxs) for idxs in temp_client_indices.values())
            if min_size >= effective_min_size or retry_count == max_retries:
                client_indices = temp_client_indices
                allocated = True

        # Shuffle each client's assigned sample indices
        for i in range(num_clients):
            rng.shuffle(client_indices[i])

        return client_indices


class PathologicalPartitioner(BasePartitioner):
    """
    Pathological Non-IID Partitioner (Class-Quota / Shard Splitting).
    
    Each client receives samples from only a restricted subset of classes (e.g. 2 classes or 20% of total classes).
    Supports either explicit integer `classes_per_client` or a float ratio `classes_fraction` in (0, 1].
    """

    def __init__(
        self,
        classes_per_client: int | None = 2,
        classes_fraction: float | None = None,
        seed: int | None = 42,
    ) -> None:
        if classes_per_client is not None and classes_per_client <= 0:
            raise ValueError(f"classes_per_client must be >= 1, got {classes_per_client}")
        if classes_fraction is not None and not (0.0 < classes_fraction <= 1.0):
            raise ValueError(f"classes_fraction must be in (0, 1], got {classes_fraction}")
            
        self.classes_per_client = classes_per_client
        self.classes_fraction = classes_fraction
        self.seed = seed

    def partition(
        self,
        targets: Sequence[int] | np.ndarray | torch.Tensor,
        num_clients: int,
    ) -> dict[int, list[int]]:
        if num_clients <= 0:
            raise ValueError(f"num_clients must be positive, got {num_clients}")

        if isinstance(targets, torch.Tensor):
            targets_np = targets.detach().cpu().numpy()
        else:
            targets_np = np.asarray(targets)

        unique_classes = np.unique(targets_np)
        num_classes = len(unique_classes)
        
        if self.classes_fraction is not None:
            k = max(1, int(math.ceil(self.classes_fraction * num_classes)))
        elif self.classes_per_client is not None:
            k = min(self.classes_per_client, num_classes)
        else:
            k = 2

        rng = np.random.default_rng(self.seed)

        # Sort indices by class
        class_indices = {c: np.where(targets_np == c)[0] for c in unique_classes}
        for c in unique_classes:
            rng.shuffle(class_indices[c])

        # Total shards needed = num_clients * k
        total_shards = num_clients * k
        shards_per_class = total_shards // num_classes
        extra_shards = total_shards % num_classes

        # Split each class into shards
        all_shards: list[list[int]] = []
        for i, c in enumerate(unique_classes):
            n_shards = shards_per_class + (1 if i < extra_shards else 0)
            if n_shards > 0:
                splits = np.array_split(class_indices[c], n_shards)
                for s in splits:
                    all_shards.append(s.tolist())

        # Shuffle shards and assign k shards per client
        rng.shuffle(all_shards)
        client_indices: dict[int, list[int]] = {i: [] for i in range(num_clients)}
        for client_id in range(num_clients):
            assigned = all_shards[client_id * k : (client_id + 1) * k]
            for shard in assigned:
                client_indices[client_id].extend(shard)
            rng.shuffle(client_indices[client_id])

        return client_indices


class IIDPartitioner(BasePartitioner):
    """
    Uniform IID Partitioner.
    
    Evenly and uniformly distributes all dataset indices among clients at random.
    """

    def __init__(self, seed: int | None = 42) -> None:
        self.seed = seed

    def partition(
        self,
        targets: Sequence[int] | np.ndarray | torch.Tensor,
        num_clients: int,
    ) -> dict[int, list[int]]:
        if num_clients <= 0:
            raise ValueError(f"num_clients must be positive, got {num_clients}")

        num_samples = len(targets)
        rng = np.random.default_rng(self.seed)
        indices = np.arange(num_samples)
        rng.shuffle(indices)

        splits = np.array_split(indices, num_clients)
        return {i: splits[i].tolist() for i in range(num_clients)}


def build_partitioner(
    partition_type: str = "dirichlet",
    alpha: float = 0.5,
    classes_per_client: int | None = None,
    classes_fraction: float | None = None,
    seed: int = 42,
    **kwargs,
) -> BasePartitioner:
    """
    Factory constructor for federated dataset partitioners.
    
    Supported types: 'dirichlet', 'non_iid_dirichlet', 'pathological', 'shard', 'iid'.
    """
    ptype = partition_type.strip().lower()
    if ptype in ("dirichlet", "non_iid", "dir", "alpha"):
        return DirichletPartitioner(alpha=alpha, seed=seed, **kwargs)
    elif ptype in ("pathological", "shard", "class_quota", "pathology"):
        return PathologicalPartitioner(
            classes_per_client=classes_per_client or 2 if classes_fraction is None else None,
            classes_fraction=classes_fraction,
            seed=seed,
        )
    elif ptype in ("iid", "uniform", "balanced"):
        return IIDPartitioner(seed=seed)
    else:
        return DirichletPartitioner(alpha=alpha, seed=seed)
