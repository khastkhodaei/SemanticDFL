"""
Algorithm 1: Binary Search for Minimal Signature Cardinality and Network Aggregation.

Provides fast, exact calculation of the minimal signature dimension P required to preserve
at least tau_min cosine similarity with the full model parameter vector:
    cos(w, w^[P]) >= tau_min
using prefix sum arrays in O(M log M) preprocessing and O(log M) search time.
"""

from __future__ import annotations

import math
from typing import Sequence
import numpy as np
import torch


def find_minimal_cardinality(
    weights: torch.Tensor,
    tau_min: float = 0.8,
    scores: torch.Tensor | None = None,
    p_min: int = 1,
    p_max: int | None = None,
) -> int:
    """
    Implements Algorithm 1 (Binary Search for Minimal Signature Cardinality).
    
    Finds the smallest integer P in [p_min, p_max] such that the cosine similarity
    between the original vector w and its top-P truncated version w^[P] is >= tau_min:
        cos(w, w^[P]) = ||w^[P]||_2 / ||w||_2 >= tau_min
        
    Args:
        weights: 1D flat tensor of model parameters or gradients w in R^M.
        tau_min: Target cosine similarity fidelity threshold in (0, 1].
        scores: Optional importance scores s in R^M. Defaults to |w|.
        p_min: Lower search bound (>= 1).
        p_max: Upper search bound (<= M, defaults to M).
        
    Returns:
        Minimal integer cardinality P satisfying the threshold.
    """
    if tau_min <= 0.0:
        return max(1, p_min)

    flat_w = weights.detach().flatten()
    m = flat_w.numel()

    if m == 0:
        return 0

    if p_max is None or p_max > m:
        p_max = m
    p_min = max(1, min(p_min, p_max))

    # Total squared Euclidean norm ||w||_2^2
    w_sq = flat_w.pow(2)
    total_energy = w_sq.sum().item()

    # If all weights are 0, return p_min
    if total_energy <= 1e-12:
        return p_min

    # Use scores or absolute magnitudes for ranking
    if scores is None:
        ranking_scores = flat_w.abs()
    else:
        ranking_scores = scores.detach().flatten()

    # Step 1: Sort coordinates descending by importance score
    sorted_indices = torch.argsort(ranking_scores, descending=True)
    sorted_w_sq = w_sq[sorted_indices]

    # Step 2: Compute prefix sums of squared weights
    prefix_energy = torch.cumsum(sorted_w_sq, dim=0)

    # Required energy threshold: (||w^[P]|| / ||w|| >= tau_min) <=> (||w^[P]||^2 >= tau_min^2 * ||w||^2)
    target_energy = (tau_min ** 2) * total_energy

    # Step 3: Binary Search over P in [p_min, p_max]
    low = p_min
    high = p_max
    best_p = p_max

    while low <= high:
        mid = (low + high) // 2
        # prefix_energy[mid - 1] is the cumulative squared norm of top-mid parameters
        current_energy = prefix_energy[mid - 1].item()

        if current_energy >= target_energy - 1e-9:
            best_p = mid
            high = mid - 1  # Try searching for a smaller valid cardinality
        else:
            low = mid + 1

    return best_p


def compute_network_p(
    local_ps: Sequence[int],
    quantile: float = 0.85,
    p_max: int | None = None,
) -> int:
    """
    Computes a synchronized network-wide signature cardinality P_net from local client values.
    
    Uses the quantile aggregation rule specified in SemanticDFL:
        P_net = ceil(Quantile({P_1, ..., P_N}, q))
    ensuring that at least `quantile` percentage of clients satisfy their local fidelity threshold.
    
    Args:
        local_ps: Sequence of locally determined minimal P values across clients.
        quantile: Target coverage quantile in [0, 1] (default 0.85).
        p_max: Optional upper bound cap.
        
    Returns:
        Unified integer network signature cardinality P_net.
    """
    if not local_ps:
        raise ValueError("local_ps sequence cannot be empty")

    if not (0.0 <= quantile <= 1.0):
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")

    arr = np.asarray(local_ps)
    network_p = int(math.ceil(float(np.quantile(arr, quantile))))
    network_p = max(1, network_p)

    if p_max is not None:
        network_p = min(network_p, p_max)

    return network_p
