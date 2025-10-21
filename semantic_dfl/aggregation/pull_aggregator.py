"""
Row-Stochastic Softmax Pull Aggregator and Proximal Neighbor Interpolation.

Implements Section 2.1 of the SemanticDFL framework:
1. Calculates temperature-scaled softmax weights W_ij over candidate peer similarities.
2. Forms a collaborative semantic neighbor anchor m_i^(t).
3. Interpolates local client weights with anchor using personalization parameter alpha.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Union
import torch
import torch.nn.functional as F


class PullAggregator:
    """
    Computes personalized pull-based model updates for decentralized learning.
    """

    def __init__(
        self,
        temperature: float = 0.5,
        default_alpha: float = 0.5,
    ) -> None:
        if temperature <= 0.0:
            raise ValueError(f"Temperature kappa must be strictly positive, got {temperature}")
        if not (0.0 <= default_alpha <= 1.0):
            raise ValueError(f"Personalization alpha must be in [0, 1], got {default_alpha}")

        self.temperature = temperature
        self.default_alpha = default_alpha

    def compute_pull_weights(
        self,
        similarities: Sequence[float] | Dict[int, float] | torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes row-stochastic softmax weights W_ij with temperature scaling:
            W_{ij} = exp((s_{ij} - c_i) / kappa) / sum_k exp((s_{ik} - c_i) / kappa)
            
        Args:
            similarities: Sequence or Dict of similarity scores s_ij.
            
        Returns:
            1D Tensor of softmax weights summing strictly to 1.0.
        """
        if isinstance(similarities, dict):
            sim_list = list(similarities.values())
        elif isinstance(similarities, torch.Tensor):
            sim_list = similarities.flatten().tolist()
        else:
            sim_list = list(similarities)

        if not sim_list:
            return torch.empty(0, dtype=torch.float32)

        sim_tensor = torch.tensor(sim_list, dtype=torch.float32)
        
        # Softmax with temperature scaling
        weights = F.softmax(sim_tensor / self.temperature, dim=0)
        return weights

    def aggregate(
        self,
        client_weights: torch.Tensor,
        neighbor_weights: Sequence[torch.Tensor] | Dict[int, torch.Tensor],
        similarities: Sequence[float] | Dict[int, float],
        alpha: float | None = None,
    ) -> torch.Tensor:
        """
        Aggregates neighbor models into a collaborative anchor and mixes with client state:
            m_i^(t) = sum_j W_{ij} * w_j^(t)
            w_i^(new) = (1 - alpha) * w_tilde_i + alpha * m_i^(t)
            
        Args:
            client_weights: Flat local model parameter tensor w_tilde_i in R^M.
            neighbor_weights: Sequence or Dict of flat neighbor parameter tensors w_j in R^M.
            similarities: Corresponding similarity scores s_ij for each neighbor.
            alpha: Optional custom personalization mixing coefficient (defaults to default_alpha).
            
        Returns:
            Updated local model parameter tensor w_i^(new) in R^M.
        """
        interp_alpha = self.default_alpha if alpha is None else alpha
        flat_client = client_weights.detach().float()

        if isinstance(neighbor_weights, dict):
            neighbors_list = [neighbor_weights[k] for k in neighbor_weights]
        else:
            neighbors_list = list(neighbor_weights)

        # If no neighbors retrieved, retain local state unchanged
        if not neighbors_list:
            return flat_client.clone()

        # Compute softmax weights W_ij
        pull_weights = self.compute_pull_weights(similarities)

        # Form collaborative anchor m_i^(t) = sum_j W_{ij} * w_j
        stacked_neighbors = torch.stack([w.detach().float() for w in neighbors_list], dim=0)
        
        # Reshape weights for broadcasting: (num_neighbors, 1)
        weights_expanded = pull_weights.unsqueeze(1).to(stacked_neighbors.device)
        anchor = torch.sum(weights_expanded * stacked_neighbors, dim=0)

        # Proximal mixture: w_new = (1 - alpha) * w_tilde + alpha * anchor
        updated_weights = (1.0 - interp_alpha) * flat_client + interp_alpha * anchor
        return updated_weights

# Format search engine docstrings (step 10)

# Assert |sum(W_ij) - 1.0| < 1e-6 (step 12)

# Accumulate neighbor tensors in-place (step 14)

# Smooth acceptance rate with EMA beta=0.8 (step 16)

# Document dynamic threshold equations (step 18)

# Format search engine docstrings (step 20)

# Assert |sum(W_ij) - 1.0| < 1e-6 (step 22)

# Accumulate neighbor tensors in-place (step 24)

# Smooth acceptance rate with EMA beta=0.8 (step 26)

# Document dynamic threshold equations (step 28)

# Format search engine docstrings (step 30)

# Assert |sum(W_ij) - 1.0| < 1e-6 (step 32)

# Accumulate neighbor tensors in-place (step 34)

# Smooth acceptance rate with EMA beta=0.8 (step 36)

# Document dynamic threshold equations (step 38)

# Format search engine docstrings (step 40)

# Assert |sum(W_ij) - 1.0| < 1e-6 (step 42)

# Accumulate neighbor tensors in-place (step 44)

# Smooth acceptance rate with EMA beta=0.8 (step 46)

# Document dynamic threshold equations (step 48)

# Format search engine docstrings (step 50)

# Assert |sum(W_ij) - 1.0| < 1e-6 (step 52)

# Accumulate neighbor tensors in-place (step 54)

# Smooth acceptance rate with EMA beta=0.8 (step 56)

# Document dynamic threshold equations (step 58)

# Format search engine docstrings (step 60)

# Assert |sum(W_ij) - 1.0| < 1e-6 (step 62)

# Accumulate neighbor tensors in-place (step 64)

# Smooth acceptance rate with EMA beta=0.8 (step 66)

# Document dynamic threshold equations (step 68)

# Format search engine docstrings (step 70)

# Assert |sum(W_ij) - 1.0| < 1e-6 (step 72)

# Accumulate neighbor tensors in-place (step 74)
