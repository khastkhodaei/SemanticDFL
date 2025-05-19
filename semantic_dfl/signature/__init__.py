"""
Semantic Signature Extraction, Sparsification, and Metric Utilities.
"""

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

__all__ = [
    "SignatureExtractor",
    "find_minimal_cardinality",
    "compute_network_p",
    "sparse_cosine_similarity",
    "pairwise_cosine_similarity_matrix",
    "affinity_propagation_preference",
]
