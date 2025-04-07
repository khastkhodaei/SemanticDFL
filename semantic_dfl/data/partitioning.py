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
