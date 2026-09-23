"""
Core neural network models and parameter manipulation utilities for SemanticDFL.
"""

from semantic_dfl.core.models import (
    AlbertTextClassifier,
    FashionCNN,
    GoogleSpeechResNet,
    MLP2NN,
    ResNet18Small,
    build_model,
    get_flattened_grads,
    get_flattened_params,
    get_model_fingerprint,
    set_flattened_params,
)

__all__ = [
    "FashionCNN",
    "ResNet18Small",
    "GoogleSpeechResNet",
    "AlbertTextClassifier",
    "MLP2NN",
    "build_model",
    "get_flattened_params",
    "set_flattened_params",
    "get_flattened_grads",
    "get_model_fingerprint",
]
