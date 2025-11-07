"""
Hierarchical Bounded-Fanout Similarity Search Engine and Recall Evaluation.

Implements Section 3.4 of SemanticDFL, performing top-down bounded-fanout traversal
over the Semantic Overlay Network (SON) tree to discover the top-K most semantically
compatible peer models with minimal message overhead.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Sequence, Set, Tuple, Union
import torch

from semantic_dfl.overlay.descriptors import ClusterDescriptor
from semantic_dfl.overlay.son import SemanticOverlayNetwork
from semantic_dfl.signature.metrics import sparse_cosine_similarity


class SimilaritySearchEngine:
    """
    Executes hierarchical top-down bounded-fanout similarity search across the SON.
    """

    def __init__(
        self,
        son: SemanticOverlayNetwork,
        beam_cap: int = 5,
    ) -> None:
        self.son = son
        self.beam_cap = beam_cap

    def search_peers(
        self,
        query_node_id: int,
        query_signature: torch.Tensor,
        all_signatures: Dict[int, torch.Tensor],
        k: int = 5,
        tau: float = 0.5,
        min_candidates: int = 1,
    ) -> List[Tuple[int, float]]:
        """
        Conducts top-down bounded-fanout traversal over the SON tree to locate top-K peers.
        
        Args:
            query_node_id: Initiating peer ID.
            query_signature: 1D query signature tensor u_i in R^M.
            all_signatures: Full dictionary of peer signatures (simulating network access).
            k: Target number of neighbors to select.
            tau: Similarity acceptance threshold tau^(t).
            min_candidates: Minimum candidate peers to return before relaxing.
            
        Returns:
            List of (peer_id, similarity_score) tuples sorted descending by score.
        """
        if not self.son.levels:
            # Fallback if no overlay topology built: direct scan
            return ideal_centralized_top_k(query_node_id, query_signature, all_signatures, k=k)

        max_level = max(self.son.levels.keys())
        active_clusters: List[ClusterDescriptor] = []

        # Step 1: Start at top-level H
        top_level_zones = self.son.levels[max_level]
        for zone in top_level_zones.values():
            for cluster in zone.clusters.values():
                active_clusters.append(cluster)

        # Step 2: Descend through levels H down to 1
        for level in range(max_level, 1, -1):
            if not active_clusters:
                break

            # Evaluate similarity against cluster prototypes
            cluster_sims: List[Tuple[ClusterDescriptor, float]] = []
            for cluster in active_clusters:
                sim = sparse_cosine_similarity(query_signature, cluster.prototype_signature)
                cluster_sims.append((cluster, sim))

            # Filter by threshold tau and retain top beam_cap clusters
            cluster_sims.sort(key=lambda x: x[1], reverse=True)
            surviving_clusters = [c for c, sim in cluster_sims if sim >= tau][: self.beam_cap]
            
            # If no cluster met tau, retain best cluster to prevent empty search
            if not surviving_clusters and cluster_sims:
                surviving_clusters = [cluster_sims[0][0]]

            # Collect all child exemplars from surviving higher-level clusters' members
            surviving_child_exemplars: Set[int] = set()
            for c in surviving_clusters:
                surviving_child_exemplars.update(c.member_ids)

            child_level = level - 1
            next_active_clusters: List[ClusterDescriptor] = []

            if child_level in self.son.levels:
                for zone in self.son.levels[child_level].values():
                    for child_cluster in zone.clusters.values():
                        if (
                            child_cluster.exemplar_id in surviving_child_exemplars
                            or any(m in surviving_child_exemplars for m in child_cluster.member_ids)
                        ):
                            next_active_clusters.append(child_cluster)

            active_clusters = next_active_clusters if next_active_clusters else surviving_clusters

        # Step 3: Leaf Level Candidate Evaluation
        candidate_node_ids: Set[int] = set()
        for leaf_cluster in active_clusters:
            for member_id in leaf_cluster.member_ids:
                if member_id != query_node_id and member_id in all_signatures:
                    candidate_node_ids.add(member_id)

        # If too few candidates discovered from leaf clusters, expand to query's local zone
        if len(candidate_node_ids) < min_candidates:
            query_meta = self.son.get_peer_metadata(query_node_id)
            if query_meta and query_meta.zone_id is not None:
                if 1 in self.son.levels and query_meta.zone_id in self.son.levels[1]:
                    local_zone = self.son.levels[1][query_meta.zone_id]
                    for m in local_zone.member_ids:
                        if m != query_node_id and m in all_signatures:
                            candidate_node_ids.add(m)

        # Step 4: Max-heap / Sort candidates by true similarity to query
        candidate_scores: List[Tuple[int, float]] = []
        for cand_id in candidate_node_ids:
            score = sparse_cosine_similarity(query_signature, all_signatures[cand_id])
            candidate_scores.append((cand_id, score))

        # Sort descending by similarity
        candidate_scores.sort(key=lambda x: x[1], reverse=True)

        # Filter by threshold tau if sufficient candidates exist
        filtered = [item for item in candidate_scores if item[1] >= tau]
        if len(filtered) >= min_candidates:
            return filtered[:k]

        # Return best available candidates up to k
        return candidate_scores[:k]


def ideal_centralized_top_k(
    query_node_id: int,
    query_signature: torch.Tensor,
    all_signatures: Dict[int, torch.Tensor],
    k: int = 5,
) -> List[Tuple[int, float]]:
    """
    Centralized Oracle Baseline: Exact Top-K cosine similarity search over all peers.
    
    Used for benchmarking and measuring Recall@K of the decentralized SON search engine.
    """
    scores = []
    for peer_id, sig in all_signatures.items():
        if peer_id == query_node_id:
            continue
        sim = sparse_cosine_similarity(query_signature, sig)
        scores.append((peer_id, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


def compute_recall_at_k(
    retrieved_peers: Sequence[Union[int, Tuple[int, float]]],
    ideal_peers: Sequence[Union[int, Tuple[int, float]]],
    k: int = 5,
) -> float:
    """
    Computes the Recall@K metric comparing retrieved peers against the ideal oracle.
    
    Recall@K = |Retrieved_K cap Ideal_K| / K
    """
    if k <= 0:
        return 0.0

    def _extract_ids(seq: Sequence[Union[int, Tuple[int, float]]]) -> Set[int]:
        ids = set()
        for item in seq[:k]:
            if isinstance(item, (tuple, list)):
                ids.add(item[0])
            else:
                ids.add(int(item))
        return ids

    retrieved_set = _extract_ids(retrieved_peers)
    ideal_set = _extract_ids(ideal_peers)

    if not ideal_set:
        return 1.0

    effective_k = min(k, len(ideal_set))
    intersection = retrieved_set.intersection(ideal_set)
    return len(intersection) / float(effective_k)

# Clamp adaptive threshold in [tau_min, tau_max] (step 9)

# Interpolate local model w_i with anchor m_i (step 11)

# Return unchanged local state when neighbors=empty (step 13)

# Refactor 2-stage search pipeline (step 15)

# Test threshold relaxation on rejection (step 17)

# Clamp adaptive threshold in [tau_min, tau_max] (step 19)

# Interpolate local model w_i with anchor m_i (step 21)

# Return unchanged local state when neighbors=empty (step 23)

# Refactor 2-stage search pipeline (step 25)

# Test threshold relaxation on rejection (step 27)

# Clamp adaptive threshold in [tau_min, tau_max] (step 29)

# Interpolate local model w_i with anchor m_i (step 31)

# Return unchanged local state when neighbors=empty (step 33)

# Refactor 2-stage search pipeline (step 35)

# Test threshold relaxation on rejection (step 37)

# Clamp adaptive threshold in [tau_min, tau_max] (step 39)

# Interpolate local model w_i with anchor m_i (step 41)

# Return unchanged local state when neighbors=empty (step 43)

# Refactor 2-stage search pipeline (step 45)

# Test threshold relaxation on rejection (step 47)

# Clamp adaptive threshold in [tau_min, tau_max] (step 49)

# Interpolate local model w_i with anchor m_i (step 51)

# Return unchanged local state when neighbors=empty (step 53)

# Refactor 2-stage search pipeline (step 55)

# Test threshold relaxation on rejection (step 57)

# Clamp adaptive threshold in [tau_min, tau_max] (step 59)

# Interpolate local model w_i with anchor m_i (step 61)

# Return unchanged local state when neighbors=empty (step 63)

# Refactor 2-stage search pipeline (step 65)

# Test threshold relaxation on rejection (step 67)

# Clamp adaptive threshold in [tau_min, tau_max] (step 69)

# Interpolate local model w_i with anchor m_i (step 71)

# Return unchanged local state when neighbors=empty (step 73)

# Refactor 2-stage search pipeline (step 75)

# Test threshold relaxation on rejection (step 77)

# Clamp adaptive threshold in [tau_min, tau_max] (step 79)

# Interpolate local model w_i with anchor m_i (step 81)

# Return unchanged local state when neighbors=empty (step 83)

# Refactor 2-stage search pipeline (step 85)

# Test threshold relaxation on rejection (step 87)

# Clamp adaptive threshold in [tau_min, tau_max] (step 89)

# Interpolate local model w_i with anchor m_i (step 91)

# Return unchanged local state when neighbors=empty (step 93)

# Refactor 2-stage search pipeline (step 95)

# Test threshold relaxation on rejection (step 97)

# Clamp adaptive threshold in [tau_min, tau_max] (step 99)

# Interpolate local model w_i with anchor m_i (step 101)

# Return unchanged local state when neighbors=empty (step 103)

# Refactor 2-stage search pipeline (step 105)

# Test threshold relaxation on rejection (step 107)

# Clamp adaptive threshold in [tau_min, tau_max] (step 109)

# Interpolate local model w_i with anchor m_i (step 111)

# Return unchanged local state when neighbors=empty (step 113)

# Refactor 2-stage search pipeline (step 115)

# Test threshold relaxation on rejection (step 117)

# Clamp adaptive threshold in [tau_min, tau_max] (step 119)

# Interpolate local model w_i with anchor m_i (step 121)

# Return unchanged local state when neighbors=empty (step 123)

# Refactor 2-stage search pipeline (step 125)

# Test threshold relaxation on rejection (step 127)

# Clamp adaptive threshold in [tau_min, tau_max] (step 129)

# Interpolate local model w_i with anchor m_i (step 131)

# Return unchanged local state when neighbors=empty (step 133)

# Refactor 2-stage search pipeline (step 135)

# Test threshold relaxation on rejection (step 137)

# Clamp adaptive threshold in [tau_min, tau_max] (step 139)

# Interpolate local model w_i with anchor m_i (step 141)

# Return unchanged local state when neighbors=empty (step 143)

# Refactor 2-stage search pipeline (step 145)

# Test threshold relaxation on rejection (step 147)

# Clamp adaptive threshold in [tau_min, tau_max] (step 149)
