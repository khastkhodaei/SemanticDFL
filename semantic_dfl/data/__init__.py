"""
Data loading and Non-IID partitioning modules for SemanticDFL.
"""

from semantic_dfl.data.datasets import (
    create_client_dataloaders,
    get_20newsgroups,
    get_cifar10,
    get_dataset,
    get_fashion_mnist,
    get_google_speech_commands,
    get_tiny_imagenet,
    load_partitioned_dataset,
)
from semantic_dfl.data.partitioning import (
    BasePartitioner,
    DirichletPartitioner,
    IIDPartitioner,
    PathologicalPartitioner,
    build_partitioner,
)

__all__ = [
    "get_fashion_mnist",
    "get_cifar10",
    "get_tiny_imagenet",
    "get_google_speech_commands",
    "get_20newsgroups",
    "get_dataset",
    "create_client_dataloaders",
    "load_partitioned_dataset",
    "BasePartitioner",
    "DirichletPartitioner",
    "PathologicalPartitioner",
    "IIDPartitioner",
    "build_partitioner",
]
