from abc import ABC, abstractmethod
from typing import Sequence, Dict, List
import numpy as np
import torch

class BasePartitioner(ABC):
    @abstractmethod
    def partition(self, targets: Sequence[int], num_clients: int) -> Dict[int, List[int]]:
        pass
