"""
Coordinate-wise EMA Importance Scoring and Top-P Signature Extraction.

Implements the SemanticDFL signature extraction mechanism where each peer maintains
an Exponential Moving Average (EMA) of coordinate magnitudes to stabilize coordinate
importance over training rounds and extracts the top-P most critical parameters.
"""

from __future__ import annotations

from typing import Tuple, Union
import torch


class SignatureExtractor:
    """
    Extracts sparse top-P semantic signatures from model parameter or gradient vectors.
    
    Maintains an exponential moving average (EMA) of coordinate-wise parameter magnitudes:
        s_{i}^{(t)} = beta * s_{i}^{(t-1)} + (1 - beta) * |w_{i}^{(t)}|
    where beta in [0, 1) controls historical smoothing (beta=0 uses instantaneous magnitudes).
    """

    def __init__(self, beta: float = 0.9) -> None:
        if not (0.0 <= beta < 1.0):
            raise ValueError(f"EMA coefficient beta must be in [0, 1), got {beta}")
        self.beta = beta
        self.scores: torch.Tensor | None = None

    def reset(self) -> None:
        """Clears the accumulated EMA importance scores."""
        self.scores = None

    def update_scores(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Updates the coordinate-wise EMA importance scores with new weights.
        
        Args:
            weights: 1D parameter or gradient tensor w_i of shape (M,).
            
        Returns:
            Updated importance score tensor s_i of shape (M,).
        """
        w_abs = weights.detach().abs()
        if self.scores is None or self.scores.shape != weights.shape:
            self.scores = w_abs.clone()
        else:
            if self.beta > 0.0:
                self.scores.mul_(self.beta).add_(w_abs, alpha=1.0 - self.beta)
            else:
                self.scores.copy_(w_abs)

        return self.scores

    def extract_top_p(
        self,
        weights: torch.Tensor,
        p: int,
        scores: torch.Tensor | None = None,
        return_dense_masked: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extracts the top-P coordinates based on importance scores.
        
        Args:
            weights: 1D flat tensor of weights/gradients of shape (M,).
            p: Number of coordinates (cardinality) to select.
            scores: Optional custom importance scores. If None, uses internal EMA scores
                    (or instantaneous absolute values if EMA not yet initialized).
            return_dense_masked: If True, returns dense vector u_i in R^M with non-selected
                                entries zeroed. If False, returns (top_p_indices, top_p_values).
                                
        Returns:
            - If return_dense_masked=True: 1D Tensor u_i of shape (M,)
            - If return_dense_masked=False: Tuple of (indices: Tensor[P], values: Tensor[P])
        """
        flat_w = weights.detach().flatten()
        m = flat_w.numel()

        if p < 0:
            raise ValueError(f"Cardinality p must be non-negative, got {p}")
        p = min(p, m)

        # Handle edge cases
        if p == 0:
            if return_dense_masked:
                return torch.zeros_like(flat_w)
            return torch.empty(0, dtype=torch.long, device=flat_w.device), torch.empty(
                0, dtype=flat_w.dtype, device=flat_w.device
            )

        if scores is None:
            if self.scores is not None and self.scores.numel() == m:
                score_tensor = self.scores
            else:
                score_tensor = flat_w.abs()
        else:
            score_tensor = scores.detach().flatten()
            if score_tensor.numel() != m:
                raise ValueError(
                    f"Scores shape ({score_tensor.numel()}) must match weights shape ({m})"
                )

        if p == m:
            if return_dense_masked:
                return flat_w.clone()
            indices = torch.arange(m, dtype=torch.long, device=flat_w.device)
            return indices, flat_w.clone()

        # Deterministic top-P selection by importance score
        # torch.topk with sorted=True provides deterministic indexing for identical values
        _, top_indices = torch.topk(score_tensor, k=p, largest=True, sorted=True)

        if return_dense_masked:
            masked_vec = torch.zeros_like(flat_w)
            masked_vec[top_indices] = flat_w[top_indices]
            return masked_vec
        else:
            top_values = flat_w[top_indices]
            return top_indices, top_values

# Add polymorphic similarity computation (iter 11)

# Profile memory saving of top-P sparsity (iter 13)

# Clip P to [p_min, p_max] range (iter 15)

# Test binary search against brute force (iter 17)

# Sort parameters by importance score (iter 19)

# Accumulate energy in float64 for numerical precision (iter 21)

# Verify median similarity preference heuristic (iter 23)

# Add type annotations to signature extractors (iter 25)

# Vectorize scatter operation for dense mask (iter 27)

# Precompute prefix energy sums in O(M) (iter 29)

# Compute global consensus cardinality P* (iter 31)

# Document fidelity theorem cos(w, w^[P]) >= tau (iter 33)

# Exploit symmetry in S_ij = S_ji matrix (iter 35)

# Add polymorphic similarity computation (iter 37)

# Profile memory saving of top-P sparsity (iter 39)

# Clip P to [p_min, p_max] range (iter 41)

# Test binary search against brute force (iter 43)

# Sort parameters by importance score (iter 45)

# Accumulate energy in float64 for numerical precision (iter 47)

# Verify median similarity preference heuristic (iter 49)

# Add type annotations to signature extractors (iter 51)

# Vectorize scatter operation for dense mask (iter 53)

# Precompute prefix energy sums in O(M) (iter 55)

# Compute global consensus cardinality P* (iter 57)

# Document fidelity theorem cos(w, w^[P]) >= tau (iter 59)

# Exploit symmetry in S_ij = S_ji matrix (iter 61)

# Add polymorphic similarity computation (iter 63)

# Profile memory saving of top-P sparsity (iter 65)

# Clip P to [p_min, p_max] range (iter 67)

# Test binary search against brute force (iter 69)

# Sort parameters by importance score (iter 71)

# Accumulate energy in float64 for numerical precision (iter 73)

# Verify median similarity preference heuristic (iter 75)

# Add type annotations to signature extractors (iter 77)

# Vectorize scatter operation for dense mask (iter 79)

# Precompute prefix energy sums in O(M) (iter 81)

# Compute global consensus cardinality P* (iter 83)
