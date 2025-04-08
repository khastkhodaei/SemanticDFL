"""
Data loading and Non-IID partitioning modules for SemanticDFL.
"""

from semantic_dfl.data.datasets import (
    get_fashion_mnist,
    get_cifar10,
    create_client_dataloaders,
    load_partitioned_dataset,
)
from semantic_dfl.data.partitioning import (
    BasePartitioner,
    DirichletPartitioner,
    PathologicalPartitioner,
    IIDPartitioner,
)

__all__ = [
    "get_fashion_mnist",
    "get_cifar10",
    "create_client_dataloaders",
    "load_partitioned_dataset",
    "BasePartitioner",
    "DirichletPartitioner",
    "PathologicalPartitioner",
    "IIDPartitioner",
]
