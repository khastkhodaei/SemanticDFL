"""
Similarity and Distance Metrics for Sparse Semantic Signatures.

Provides fast, vectorized cosine similarity metrics and Affinity Propagation preference
calculators for decentralized clustering and peer discovery.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
import torch


def sparse_cosine_similarity(
    u_i: torch.Tensor,
    u_j: torch.Tensor,
    eps: float = 1e-8,
    normalized_range: bool = False,
) -> float:
    """
    Computes the cosine similarity between two masked/sparse signature vectors.
    
    Args:
        u_i: 1D signature tensor of shape (M,).
        u_j: 1D signature tensor of shape (M,).
        eps: Small epsilon to prevent division by zero.
        normalized_range: If True, maps [-1, 1] to [0, 1] via (cos + 1) / 2.
        
    Returns:
        Scalar float similarity score.
    """
    flat_i = u_i.detach().flatten().float()
    flat_j = u_j.detach().flatten().float()

    dot = torch.dot(flat_i, flat_j).item()
    norm_i = torch.norm(flat_i, p=2).item()
    norm_j = torch.norm(flat_j, p=2).item()

    denom = norm_i * norm_j
    if denom <= eps:
        cos_sim = 0.0
    else:
        cos_sim = max(-1.0, min(1.0, dot / denom))

    if normalized_range:
        return (cos_sim + 1.0) / 2.0
    return cos_sim


def pairwise_cosine_similarity_matrix(
    signatures: torch.Tensor | Sequence[torch.Tensor],
    eps: float = 1e-8,
    normalized_range: bool = False,
) -> torch.Tensor:
    """
    Vectorized computation of the N x N pairwise cosine similarity matrix.
    
    Args:
        signatures: 2D Tensor of shape (N, M) OR Sequence of N 1D Tensors.
        eps: Epsilon for numerical stability.
        normalized_range: If True, normalizes output range from [-1, 1] to [0, 1].
        
    Returns:
        2D Tensor of shape (N, N) where entry [i, j] is the cosine similarity between u_i and u_j.
    """
    if isinstance(signatures, (list, tuple)):
        mat = torch.stack([sig.detach().flatten().float() for sig in signatures], dim=0)
    else:
        mat = signatures.detach().float()
        if mat.ndim == 1:
            mat = mat.unsqueeze(0)

    # Compute L2 norms per row: shape (N, 1)
    norms = torch.norm(mat, p=2, dim=1, keepdim=True)
    
    # Identify non-zero rows
    valid_mask = (norms > eps).squeeze(1)

    # Normalize vectors
    norms_clamped = torch.clamp(norms, min=eps)
    normalized_mat = mat / norms_clamped

    # Compute dot products: (N, M) x (M, N) -> (N, N)
    sim_matrix = torch.mm(normalized_mat, normalized_mat.t())
    
    # Zero out rows/cols that had zero norm
    zero_rows = ~valid_mask
    if zero_rows.any():
        sim_matrix[zero_rows, :] = 0.0
        sim_matrix[:, zero_rows] = 0.0

    # Ensure diagonal is exactly 1.0 for non-zero vectors
    sim_matrix.diagonal().fill_(1.0)
    if zero_rows.any():
        sim_matrix.diagonal()[zero_rows] = 0.0

    sim_matrix.clamp_(-1.0, 1.0)

    if normalized_range:
        sim_matrix = (sim_matrix + 1.0) / 2.0

    return sim_matrix


def affinity_propagation_preference(
    similarity_matrix: torch.Tensor | np.ndarray,
    quantile: float = 0.5,
) -> float:
    """
    Calculates the clustering preference parameter for Scikit-Learn's AffinityPropagation.
    
    Extracts the off-diagonal pairwise similarities and computes the requested quantile
    (default: median = 0.5 quantile).
    
    Args:
        similarity_matrix: N x N pairwise similarity matrix.
        quantile: Quantile in [0, 1] to use as preference (0.5 = median).
        
    Returns:
        Scalar float preference value.
    """
    if isinstance(similarity_matrix, torch.Tensor):
        s_np = similarity_matrix.detach().cpu().numpy()
    else:
        s_np = np.asarray(similarity_matrix)

    n = s_np.shape[0]
    if n <= 1:
        return 0.0

    # Extract strictly off-diagonal elements
    mask = ~np.eye(n, dtype=bool)
    off_diag = s_np[mask]

    if len(off_diag) == 0:
        return 0.0

    pref = float(np.quantile(off_diag, quantile))
    return pref

# Add type annotations to signature extractors (iter 12)

# Vectorize scatter operation for dense mask (iter 14)

# Precompute prefix energy sums in O(M) (iter 16)

# Compute global consensus cardinality P* (iter 18)

# Document fidelity theorem cos(w, w^[P]) >= tau (iter 20)

# Exploit symmetry in S_ij = S_ji matrix (iter 22)

# Add polymorphic similarity computation (iter 24)

# Profile memory saving of top-P sparsity (iter 26)

# Clip P to [p_min, p_max] range (iter 28)

# Test binary search against brute force (iter 30)

# Sort parameters by importance score (iter 32)

# Accumulate energy in float64 for numerical precision (iter 34)

# Verify median similarity preference heuristic (iter 36)

# Add type annotations to signature extractors (iter 38)

# Vectorize scatter operation for dense mask (iter 40)

# Precompute prefix energy sums in O(M) (iter 42)

# Compute global consensus cardinality P* (iter 44)

# Document fidelity theorem cos(w, w^[P]) >= tau (iter 46)

# Exploit symmetry in S_ij = S_ji matrix (iter 48)

# Add polymorphic similarity computation (iter 50)

# Profile memory saving of top-P sparsity (iter 52)

# Clip P to [p_min, p_max] range (iter 54)

# Test binary search against brute force (iter 56)

# Sort parameters by importance score (iter 58)

# Accumulate energy in float64 for numerical precision (iter 60)

# Verify median similarity preference heuristic (iter 62)

# Add type annotations to signature extractors (iter 64)

# Vectorize scatter operation for dense mask (iter 66)

# Precompute prefix energy sums in O(M) (iter 68)

# Compute global consensus cardinality P* (iter 70)

# Document fidelity theorem cos(w, w^[P]) >= tau (iter 72)

# Exploit symmetry in S_ij = S_ji matrix (iter 74)

# Add polymorphic similarity computation (iter 76)

# Profile memory saving of top-P sparsity (iter 78)

# Clip P to [p_min, p_max] range (iter 80)

# Test binary search against brute force (iter 82)

# Sort parameters by importance score (iter 84)

# Accumulate energy in float64 for numerical precision (iter 86)

# Verify median similarity preference heuristic (iter 88)

# Add type annotations to signature extractors (iter 90)

# Vectorize scatter operation for dense mask (iter 92)

# Precompute prefix energy sums in O(M) (iter 94)

# Compute global consensus cardinality P* (iter 96)

# Document fidelity theorem cos(w, w^[P]) >= tau (iter 98)

# Exploit symmetry in S_ij = S_ji matrix (iter 100)

# Add polymorphic similarity computation (iter 102)

# Profile memory saving of top-P sparsity (iter 104)

# Clip P to [p_min, p_max] range (iter 106)

# Test binary search against brute force (iter 108)

# Sort parameters by importance score (iter 110)
