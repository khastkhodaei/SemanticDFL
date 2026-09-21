"""
Decentralized Federated Learning (DFL) Client Node State and Training Kernel.

Encapsulates local deep learning model training, EMA importance scoring, signature extraction,
and adaptive threshold management for an autonomous peer.
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from semantic_dfl.core.models import get_flattened_params, set_flattened_params
from semantic_dfl.search.controller import AdaptiveThresholdController
from semantic_dfl.signature.extractor import SignatureExtractor


class DFLClient:
    """
    Simulated autonomous peer in the SemanticDFL network.
    """

    def __init__(
        self,
        node_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        lr: float = 0.01,
        beta: float = 0.9,
        tau_0: float = 0.5,
        phi: float = 0.5,
        eta_tau: float = 0.05,
        gamma: float = 0.02,
        device: Optional[torch.device] = None,
    ) -> None:
        self.node_id = node_id
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

        # Local Adaptive Modules
        self.signature_extractor = SignatureExtractor(beta=beta)
        self.threshold_controller = AdaptiveThresholdController(
            tau_0=tau_0,
            phi=phi,
            eta_tau=eta_tau,
            gamma=gamma,
        )

        self.last_signature: Optional[torch.Tensor] = None

    def get_weights(self) -> torch.Tensor:
        """Extracts flat 1D parameter vector from local model."""
        return get_flattened_params(self.model)

    def set_weights(self, flat_params: torch.Tensor) -> None:
        """Sets flat 1D parameter vector into local model."""
        set_flattened_params(self.model, flat_params.to(self.device))

    def train_epoch(self, epochs: int = 1) -> float:
        """
        Executes local SGD training iterations.
        
        Args:
            epochs: Number of complete passes over the local dataset.
            
        Returns:
            Average scalar training loss over all batches.
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for _ in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)

        return total_loss / max(1, total_samples)

    def extract_signature(self, p: int) -> torch.Tensor:
        """
        Extracts the top-P sparsified signature vector u_i based on EMA importance scores.
        
        Args:
            p: Cardinality budget.
            
        Returns:
            1D masked signature tensor u_i in R^M.
        """
        flat_w = self.get_weights()
        self.signature_extractor.update_scores(flat_w)
        self.last_signature = self.signature_extractor.extract_top_p(
            weights=flat_w,
            p=p,
            return_dense_masked=True,
        )
        return self.last_signature

    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Tuple[float, float]:
        """
        Evaluates the model on test data.
        
        Args:
            dataloader: Optional custom DataLoader (defaults to self.test_loader).
            
        Returns:
            Tuple of (average_test_loss, accuracy_percentage).
        """
        loader = dataloader or self.test_loader
        if loader is None:
            return 0.0, 0.0

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item() * data.size(0)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += data.size(0)

        if total_samples == 0:
            return 0.0, 0.0

        avg_loss = total_loss / total_samples
        accuracy = (correct / total_samples) * 100.0
        return avg_loss, accuracy
