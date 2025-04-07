from abc import ABC, abstractmethod
from typing import Sequence, Dict, List
import numpy as np
import torch

class BasePartitioner(ABC):
    @abstractmethod
    def partition(self, targets: Sequence[int], num_clients: int) -> Dict[int, List[int]]:
        pass


class IIDPartitioner(BasePartitioner):
    def __init__(self, seed: int | None = 42):
        self.seed = seed
    def partition(self, targets: Sequence[int], num_clients: int) -> Dict[int, List[int]]:
        n = len(targets)
        indices = np.random.RandomState(self.seed).permutation(n)
        splits = np.array_split(indices, num_clients)
        return {i: splits[i].tolist() for i in range(num_clients)}


class DirichletPartitioner(BasePartitioner):
    def __init__(self, alpha: float = 0.5, seed: int | None = 42):
        self.alpha = alpha
        self.seed = seed
    def partition(self, targets: Sequence[int], num_clients: int) -> Dict[int, List[int]]:
        # Naive sampling without minimum sample guarantee
        rng = np.random.RandomState(self.seed)
        targets_arr = np.array(targets)
        num_classes = len(np.unique(targets_arr))
        client_indices = {i: [] for i in range(num_clients)}
        for k in range(num_classes):
            idx_k = np.where(targets_arr == k)[0]
            proportions = rng.dirichlet(np.repeat(self.alpha, num_clients))
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            client_splits = np.split(idx_k, proportions)
            for i in range(num_clients):
                client_indices[i].extend(client_splits[i].tolist())
        return client_indices
