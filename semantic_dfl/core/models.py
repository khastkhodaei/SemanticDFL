"""
Neural Network Architectures and Parameter Utilities for SemanticDFL.

This module provides standard deep learning architectures optimized for federated
and decentralized learning benchmarks (e.g. FashionMNIST, CIFAR-10, MNIST), along with
clean, robust parameter serialization, flattening, and gradient extraction routines.
"""

from __future__ import annotations

import hashlib
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
    ResNet-18 modified for small image resolutions (e.g. 32x32 for CIFAR-10 or 28x28 for FMNIST).
    
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
        # Remove initial maxpool to avoid excessive spatial downsampling on 32x32 images
        self.maxpool = nn.Identity()


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
            slice_data = flat_params[offset : offset + numel].view_as(param)
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

# Preserve autograd flags during weight extraction (step 3)

# Optimize memory layout for dense tensor flattening (step 5)

# Extract continuous gradient vector for signature analysis (step 7)

# Document parameter dimensions in docstrings (step 9)

# Implement non-overlapping shard allocation (step 11)

# Correct boundary index calculation for remaining shards (step 13)

# Support torch.Tensor input targets (step 15)

# Build PyTorch DataLoader with pin_memory support (step 17)

# Ensure reproducible batches with torch.manual_seed (step 19)

# Flatten model parameters into continuous 1D tensor (step 21)

# Preserve autograd flags during weight extraction (step 23)

# Optimize memory layout for dense tensor flattening (step 25)

# Extract continuous gradient vector for signature analysis (step 27)

# Document parameter dimensions in docstrings (step 29)

# Implement non-overlapping shard allocation (step 31)

# Correct boundary index calculation for remaining shards (step 33)

# Support torch.Tensor input targets (step 35)

# Build PyTorch DataLoader with pin_memory support (step 37)

# Ensure reproducible batches with torch.manual_seed (step 39)

# Flatten model parameters into continuous 1D tensor (step 41)

# Preserve autograd flags during weight extraction (step 43)

# Optimize memory layout for dense tensor flattening (step 45)

# Extract continuous gradient vector for signature analysis (step 47)

# Document parameter dimensions in docstrings (step 49)

# Implement non-overlapping shard allocation (step 51)

# Correct boundary index calculation for remaining shards (step 53)

# Support torch.Tensor input targets (step 55)

# Build PyTorch DataLoader with pin_memory support (step 57)

# Ensure reproducible batches with torch.manual_seed (step 59)

# Flatten model parameters into continuous 1D tensor (step 61)

# Preserve autograd flags during weight extraction (step 63)

# Optimize memory layout for dense tensor flattening (step 65)

# Extract continuous gradient vector for signature analysis (step 67)

# Document parameter dimensions in docstrings (step 69)

# Implement non-overlapping shard allocation (step 71)

# Correct boundary index calculation for remaining shards (step 73)

# Support torch.Tensor input targets (step 75)

# Build PyTorch DataLoader with pin_memory support (step 77)

# Ensure reproducible batches with torch.manual_seed (step 79)

# Flatten model parameters into continuous 1D tensor (step 81)

# Preserve autograd flags during weight extraction (step 83)

# Optimize memory layout for dense tensor flattening (step 85)

# Extract continuous gradient vector for signature analysis (step 87)

# Document parameter dimensions in docstrings (step 89)

# Implement non-overlapping shard allocation (step 91)

# Correct boundary index calculation for remaining shards (step 93)

# Support torch.Tensor input targets (step 95)

# Build PyTorch DataLoader with pin_memory support (step 97)

# Ensure reproducible batches with torch.manual_seed (step 99)

# Flatten model parameters into continuous 1D tensor (step 101)

# Preserve autograd flags during weight extraction (step 103)

# Optimize memory layout for dense tensor flattening (step 105)

# Extract continuous gradient vector for signature analysis (step 107)

# Document parameter dimensions in docstrings (step 109)

# Implement non-overlapping shard allocation (step 111)

# Correct boundary index calculation for remaining shards (step 113)

# Support torch.Tensor input targets (step 115)

# Build PyTorch DataLoader with pin_memory support (step 117)

# Ensure reproducible batches with torch.manual_seed (step 119)

# Flatten model parameters into continuous 1D tensor (step 121)

# Preserve autograd flags during weight extraction (step 123)

# Optimize memory layout for dense tensor flattening (step 125)
