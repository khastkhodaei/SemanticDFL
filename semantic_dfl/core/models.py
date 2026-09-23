"""
Neural Network Architectures and Parameter Utilities for SemanticDFL.

This module provides standard deep learning architectures evaluated across multi-modal
benchmarks in SemanticDFL (ACM 2026):
1. Fashion-MNIST: 2-layer CNN (FashionCNN) for 10-class fashion item classification.
2. Tiny-ImageNet / CIFAR: ResNet-18 adapted for small inputs (ResNet18Small).
3. Google Speech Commands: ResNet-18 with 1-channel Log-Mel spectrogram input (GoogleSpeechResNet).
4. 20Newsgroups: Transformer / ALBERT-style text classifier (AlbertTextClassifier).

Along with clean, robust parameter serialization, flattening, and gradient extraction routines.
"""

from __future__ import annotations

import hashlib
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torchvision_models
from torchvision.models.resnet import BasicBlock, ResNet


class FashionCNN(nn.Module):
    """
    Standard 2-layer Convolutional Neural Network for FashionMNIST / MNIST classification.
    
    Architecture:
        - Conv2d(in_channels, 32, kernel_size=5, padding=2) -> ReLU -> MaxPool2d(2, 2)
        - Conv2d(32, 64, kernel_size=5, padding=2) -> ReLU -> MaxPool2d(2, 2)
        - Linear(64 * (H // 4) * (W // 4), 512) -> ReLU
        - Linear(512, num_classes)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        input_shape: tuple[int, int] = (28, 28),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_shape = input_shape

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Compute flattened feature size dynamically based on input resolution
        h_out = input_shape[0] // 4
        w_out = input_shape[1] // 4
        self.flattened_dim = 64 * h_out * w_out

        self.fc1 = nn.Linear(in_features=self.flattened_dim, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch_size, in_channels, H, W)
        Returns:
            Unnormalized logits of shape (batch_size, num_classes)
        """
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ResNet18Small(ResNet):
    """
    ResNet-18 modified for small image resolutions (e.g. 32x32 for CIFAR-10, 64x64 for Tiny-ImageNet).
    
    Standard ResNet-18 uses a 7x7 conv with stride 2 and max pooling, which aggressively
    reduces spatial dimensions of small images. This adaptation uses a 3x3 conv with stride 1
    and replaces the initial maxpool with Identity, preserving feature maps for small inputs.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
    ) -> None:
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Replace initial 7x7 conv with 3x3 conv (stride 1)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        # Remove initial maxpool to avoid excessive spatial downsampling on small images
        self.maxpool = nn.Identity()


class GoogleSpeechResNet(ResNet):
    """
    ResNet-18 adapted for 1-channel Log-Mel Spectrogram speech classification (Google Speech Commands).
    
    Default configuration classifies 35 speech command classes from single-channel spectrograms.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 35,
    ) -> None:
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.maxpool = nn.Identity()


class AlbertTextClassifier(nn.Module):
    """
    Lightweight Transformer Text Classifier for 20Newsgroups text categorization.
    
    Architecture:
        - Embedding(vocab_size, embed_dim) + Positional Encoding
        - TransformerEncoderLayer (multi-head self-attention + feed-forward)
        - Mean/Cls pooling
        - Linear(embed_dim, hidden_dim) -> ReLU -> Dropout
        - Linear(hidden_dim, num_classes)
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        embed_dim: int = 128,
        num_heads: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 20,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input token indices tensor of shape (batch_size, seq_len)
        Returns:
            Unnormalized logits of shape (batch_size, num_classes)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch_size, seq_len = x.shape
        seq_len = min(seq_len, self.max_seq_len)
        x = x[:, :seq_len]

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        h = self.token_embedding(x) + self.pos_embedding(positions)
        encoded = self.transformer(h)

        # Global average pooling over token sequence
        pooled = encoded.mean(dim=1)
        out = F.relu(self.fc1(pooled))
        out = self.dropout(out)
        return self.fc2(out)


class MLP2NN(nn.Module):
    """
    2-Layer Multi-Layer Perceptron (2NN) benchmark baseline.
    
    Architecture (from McMahan et al., 2017):
        - Flatten -> Linear(input_dim, 200) -> ReLU
        - Linear(200, 200) -> ReLU
        - Linear(200, num_classes)
    """

    def __init__(
        self,
        input_dim: int = 784,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(
    model_name: str = "fashion_cnn",
    num_classes: int = 10,
    in_channels: int | None = None,
    **kwargs,
) -> nn.Module:
    """
    Factory function for instantiating model architectures evaluated in SemanticDFL.
    
    Args:
        model_name: Name of model ('fashion_cnn', 'fmnist_cnn', 'resnet18_small',
                    'tiny_imagenet', 'cifar10_resnet', 'google_speech', 'speech_resnet',
                    'albert', '20newsgroups', 'mlp2nn').
        num_classes: Number of output target classes.
        in_channels: Input channels (optional override).
        
    Returns:
        Instantiated nn.Module instance.
    """
    name = model_name.strip().lower()
    if name in ("fashion_cnn", "fashioncnn", "fmnist_cnn", "cnn"):
        ch = in_channels if in_channels is not None else 1
        return FashionCNN(in_channels=ch, num_classes=num_classes, **kwargs)
    elif name in ("resnet18_small", "resnet18", "tiny_imagenet", "cifar10_resnet", "cifar_resnet"):
        ch = in_channels if in_channels is not None else 3
        return ResNet18Small(in_channels=ch, num_classes=num_classes)
    elif name in ("google_speech", "google_speech_resnet", "speech_resnet", "speech"):
        ch = in_channels if in_channels is not None else 1
        n_cls = num_classes if num_classes != 10 else 35
        return GoogleSpeechResNet(in_channels=ch, num_classes=n_cls)
    elif name in ("albert", "albert_text", "20newsgroups", "newsgroups", "transformer_text"):
        n_cls = num_classes if num_classes != 10 else 20
        return AlbertTextClassifier(num_classes=n_cls, **kwargs)
    elif name in ("mlp2nn", "2nn", "mlp"):
        input_dim = kwargs.get("input_dim", 784)
        return MLP2NN(input_dim=input_dim, num_classes=num_classes)
    else:
        # Default to FashionCNN
        ch = in_channels if in_channels is not None else 1
        return FashionCNN(in_channels=ch, num_classes=num_classes)


# -----------------------------------------------------------------------------
# Parameter & Gradient Flattening / Unflattening Utilities
# -----------------------------------------------------------------------------

def get_flattened_params(model: nn.Module) -> torch.Tensor:
    """
    Extracts all trainable parameters from the model as a contiguous 1D float Tensor.
    
    Args:
        model: PyTorch Module
        
    Returns:
        1D Tensor containing concatenated, detached parameter values.
    """
    params = [p.detach().flatten() for p in model.parameters() if p.requires_grad]
    if not params:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(params)


def set_flattened_params(model: nn.Module, flat_params: torch.Tensor) -> None:
    """
    Updates model parameters in-place from a contiguous 1D Tensor.
    
    Args:
        model: PyTorch Module whose parameters will be updated.
        flat_params: 1D Tensor containing the new parameter values.
        
    Raises:
        ValueError: If total elements in flat_params does not match model parameters.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if flat_params.numel() != total_params:
        raise ValueError(
            f"Dimension mismatch: flat_params has {flat_params.numel()} elements, "
            f"but model expects {total_params} trainable parameters."
        )

    offset = 0
    with torch.no_grad():
        for param in model.parameters():
            if not param.requires_grad:
                continue
            numel = param.numel()
            slice_data = flat_params[offset : offset + numel].to(device=param.device, dtype=param.dtype).view_as(param)
            param.copy_(slice_data)
            offset += numel


def get_flattened_grads(model: nn.Module) -> torch.Tensor:
    """
    Extracts all gradient tensors from trainable parameters as a contiguous 1D float Tensor.
    If a parameter has no gradient, zeros of matching shape are substituted.
    
    Args:
        model: PyTorch Module
        
    Returns:
        1D Tensor containing concatenated gradient values.
    """
    grads = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.grad is not None:
            grads.append(p.grad.detach().flatten())
        else:
            grads.append(torch.zeros(p.numel(), dtype=p.dtype, device=p.device))

    if not grads:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(grads)


def get_model_fingerprint(model: nn.Module) -> str:
    """
    Generates a deterministic MD5 hash fingerprint of the model's state dictionary.
    Useful for asserting identical model initialization across decentralized peers.
    """
    hash_gen = hashlib.md5()
    for name in sorted(model.state_dict().keys()):
        tensor = model.state_dict()[name]
        hash_gen.update(tensor.detach().cpu().numpy().tobytes())
    return hash_gen.hexdigest()
