"""
Core neural network models and parameter manipulation utilities for SemanticDFL.
"""

from semantic_dfl.core.models import (
    FashionCNN,
    ResNet18Small,
    MLP2NN,
    get_flattened_params,
    set_flattened_params,
    get_flattened_grads,
)

__all__ = [
    "FashionCNN",
    "ResNet18Small",
    "MLP2NN",
    "get_flattened_params",
    "set_flattened_params",
    "get_flattened_grads",
]
