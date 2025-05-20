"""
Unit Tests for Top-P Signature Extraction, Binary Search Cardinality, and Sparse Metrics.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np

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


class TestSignatureExtractor:
    """Tests for SignatureExtractor EMA tracking and Top-P selection."""

    def test_ema_scoring_beta_zero(self):
        extractor = SignatureExtractor(beta=0.0)
        w1 = torch.tensor([1.0, -2.0, 3.0])
        s1 = extractor.update_scores(w1)
        assert torch.allclose(s1, torch.tensor([1.0, 2.0, 3.0]))

        w2 = torch.tensor([-5.0, 0.0, 1.0])
        s2 = extractor.update_scores(w2)
        assert torch.allclose(s2, torch.tensor([5.0, 0.0, 1.0]))

    def test_ema_scoring_beta_smoothing(self):
        extractor = SignatureExtractor(beta=0.9)
        w1 = torch.tensor([10.0, 20.0])
        s1 = extractor.update_scores(w1)
        # First update: initializes to |w1|
        assert torch.allclose(s1, torch.tensor([10.0, 20.0]))

        w2 = torch.tensor([0.0, 10.0])
        s2 = extractor.update_scores(w2)
        # s2 = 0.9 * [10, 20] + 0.1 * [0, 10] = [9.0, 19.0]
        expected = torch.tensor([9.0, 19.0])
        assert torch.allclose(s2, expected)

    def test_extract_top_p_dense_and_sparse(self):
        extractor = SignatureExtractor(beta=0.9)
        w = torch.tensor([1.5, -4.0, 0.2, 8.0, -1.0])
        extractor.update_scores(w)

        # Top 2 coordinates by magnitude are indices 3 (val 8.0) and 1 (val -4.0)
        dense_masked = extractor.extract_top_p(w, p=2, return_dense_masked=True)
        assert dense_masked.shape == w.shape
        assert dense_masked[3] == 8.0
        assert dense_masked[1] == -4.0
        assert dense_masked[0] == 0.0
        assert dense_masked[2] == 0.0
        assert dense_masked[4] == 0.0

        indices, values = extractor.extract_top_p(w, p=2, return_dense_masked=False)
        assert set(indices.tolist()) == {3, 1}
        assert set(values.abs().tolist()) == {8.0, 4.0}

    def test_extract_top_p_edge_cases(self):
        extractor = SignatureExtractor()
        w = torch.tensor([1.0, 2.0, 3.0])
        
        # p = 0
        dense_0 = extractor.extract_top_p(w, p=0, return_dense_masked=True)
        assert torch.all(dense_0 == 0.0)

        # p >= M
        dense_all = extractor.extract_top_p(w, p=10, return_dense_masked=True)
        assert torch.allclose(dense_all, w)

        # reset
        extractor.reset()
        assert extractor.scores is None


class TestBinarySearchCardinality:
    """Tests for Algorithm 1 and network cardinality aggregation."""

    def test_binary_search_matches_linear_brute_force(self):
        """Validates that Algorithm 1 (binary search) strictly matches linear brute-force."""
        torch.manual_seed(42)
        m = 200
        w = torch.randn(m)
        scores = w.abs() + 0.1 * torch.randn(m).abs()

        # Precompute brute force optimal P for several tau_min thresholds
        sorted_indices = torch.argsort(scores, descending=True)

        for tau_min in [0.5, 0.7, 0.85, 0.95, 0.99]:
            # Brute force linear scan:
            brute_force_p = m
            for p_candidate in range(1, m + 1):
                mask = torch.zeros_like(w)
                mask[sorted_indices[:p_candidate]] = w[sorted_indices[:p_candidate]]
                cos_val = F.cosine_similarity(w.unsqueeze(0), mask.unsqueeze(0)).item()
                if cos_val >= tau_min - 1e-6:
                    brute_force_p = p_candidate
                    break

            # Binary search:
            bs_p = find_minimal_cardinality(w, tau_min=tau_min, scores=scores)
            assert bs_p == brute_force_p, f"Mismatch for tau_min={tau_min}: BS={bs_p}, BF={brute_force_p}"

    def test_binary_search_edge_cases(self):
        w_zeros = torch.zeros(100)
        assert find_minimal_cardinality(w_zeros, tau_min=0.8) == 1

        w_single = torch.tensor([5.0])
        assert find_minimal_cardinality(w_single, tau_min=0.99) == 1

        w = torch.randn(50)
        assert find_minimal_cardinality(w, tau_min=0.0) == 1
        assert find_minimal_cardinality(w, tau_min=1.0) == 50

    def test_compute_network_p(self):
        client_ps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        # 0.5 quantile (median)
        p_50 = compute_network_p(client_ps, quantile=0.5)
        assert p_50 == 55

        # 0.85 quantile
        p_85 = compute_network_p(client_ps, quantile=0.85)
        assert p_85 == int(np.ceil(np.quantile(client_ps, 0.85)))

        # with p_max cap
        p_capped = compute_network_p(client_ps, quantile=0.99, p_max=45)
        assert p_capped == 45


class TestSimilarityMetrics:
    """Tests for Sparse Cosine Similarity and Vectorized Pairwise Matrix."""

    def test_sparse_cosine_similarity(self):
        u = torch.tensor([1.0, 2.0, 0.0, 0.0])
        v = torch.tensor([1.0, 2.0, 0.0, 0.0])
        assert pytest.approx(sparse_cosine_similarity(u, v), rel=1e-5) == 1.0

        w = torch.tensor([0.0, 0.0, 3.0, 4.0])
        # Orthogonal non-overlapping signatures
        assert pytest.approx(sparse_cosine_similarity(u, w), abs=1e-5) == 0.0

        # Normalized range [0, 1]
        assert pytest.approx(sparse_cosine_similarity(u, w, normalized_range=True), abs=1e-5) == 0.5

    def test_pairwise_cosine_similarity_matrix_consistency(self):
        torch.manual_seed(42)
        n = 5
        m = 50
        signatures = [torch.randn(m) for _ in range(n)]

        # Pairwise matrix
        mat = pairwise_cosine_similarity_matrix(signatures)
        assert mat.shape == (n, n)

        # Check diagonal is 1.0
        for i in range(n):
            assert pytest.approx(mat[i, i].item(), abs=1e-5) == 1.0

        # Check symmetry and match with individual sparse_cosine_similarity calls
        for i in range(n):
            for j in range(n):
                expected = sparse_cosine_similarity(signatures[i], signatures[j])
                assert pytest.approx(mat[i, j].item(), abs=1e-4) == expected
                assert pytest.approx(mat[i, j].item(), abs=1e-4) == mat[j, i].item()

    def test_affinity_propagation_preference(self):
        mat = torch.tensor([
            [1.0, 0.8, 0.2],
            [0.8, 1.0, 0.4],
            [0.2, 0.4, 1.0],
        ])
        # Off-diagonal elements: [0.8, 0.2, 0.8, 0.4, 0.2, 0.4] -> sorted: [0.2, 0.2, 0.4, 0.4, 0.8, 0.8]
        # Median should be 0.4
        pref = affinity_propagation_preference(mat, quantile=0.5)
        assert pytest.approx(pref, abs=1e-5) == 0.4
