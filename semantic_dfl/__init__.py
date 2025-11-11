"""
SemanticDFL: A Decentralized Federated Learning Framework with Semantic Overlay Networks.
"""

from semantic_dfl.core.models import (
    FashionCNN,
    ResNet18Small,
    MLP2NN,
    get_flattened_params,
    set_flattened_params,
    get_flattened_grads,
    get_model_fingerprint,
)
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
from semantic_dfl.signature.extractor import SignatureExtractor
from semantic_dfl.signature.binary_search import (
    find_minimal_cardinality,
    compute_network_p,
)
from semantic_dfl.signature.metrics import (
    sparse_cosine_similarity,
    pairwise_cosine_similarity_matrix,
    affinity_propagation_preference,
)
from semantic_dfl.overlay.descriptors import (
    ClusterDescriptor,
    ZoneDescriptor,
    PeerNodeMetadata,
)
from semantic_dfl.overlay.clustering import ZoneClustering
from semantic_dfl.overlay.zone import ZoneManager
from semantic_dfl.overlay.son import SemanticOverlayNetwork
from semantic_dfl.search.controller import AdaptiveThresholdController
from semantic_dfl.search.engine import (
    SimilaritySearchEngine,
    ideal_centralized_top_k,
    compute_recall_at_k,
)
from semantic_dfl.aggregation.pull_aggregator import PullAggregator
from semantic_dfl.federated.client import DFLClient
from semantic_dfl.federated.coordinator import SemanticDFLCoordinator

__version__ = "0.1.0"

__all__ = [
    # Core Models
    "FashionCNN",
    "ResNet18Small",
    "MLP2NN",
    "get_flattened_params",
    "set_flattened_params",
    "get_flattened_grads",
    "get_model_fingerprint",
    # Data & Partitioning
    "get_fashion_mnist",
    "get_cifar10",
    "create_client_dataloaders",
    "load_partitioned_dataset",
    "BasePartitioner",
    "DirichletPartitioner",
    "PathologicalPartitioner",
    "IIDPartitioner",
    # Signature
    "SignatureExtractor",
    "find_minimal_cardinality",
    "compute_network_p",
    "sparse_cosine_similarity",
    "pairwise_cosine_similarity_matrix",
    "affinity_propagation_preference",
    # Overlay & SON
    "ClusterDescriptor",
    "ZoneDescriptor",
    "PeerNodeMetadata",
    "ZoneClustering",
    "ZoneManager",
    "SemanticOverlayNetwork",
    # Search & Tuning
    "AdaptiveThresholdController",
    "SimilaritySearchEngine",
    "ideal_centralized_top_k",
    "compute_recall_at_k",
    # Aggregation & Federated
    "PullAggregator",
    "DFLClient",
    "SemanticDFLCoordinator",
]
