"""
Hierarchical Bounded-Fanout Similarity Search Engine and Recall Evaluation.

Implements Section 3.4 and Section 3.5 of SemanticDFL, performing top-down
bounded-fanout traversal over the Semantic Overlay Network (SON) tree to discover
the top-K most semantically compatible peer models with minimal control message overhead.
"""

from __future__ import annotations

import heapq
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
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
        relax_delta: float = 0.02,
        max_relax_steps: int = 3,
    ) -> None:
        self.son = son
        self.beam_cap = beam_cap
        self.relax_delta = relax_delta
        self.max_relax_steps = max_relax_steps

    def search_peers(
        self,
        query_node_id: int,
        query_signature: torch.Tensor,
        all_signatures: Dict[int, torch.Tensor],
        k: int = 5,
        tau: float = 0.5,
        min_candidates: int = 1,
        return_stats: bool = False,
    ) -> Union[List[Tuple[int, float]], Tuple[List[Tuple[int, float]], Dict[str, Any]]]:
        """
        Conducts top-down bounded-fanout traversal over the SON tree to locate top-K peers.
        
        Args:
            query_node_id: Initiating peer ID.
            query_signature: 1D query signature tensor u_i in R^M.
            all_signatures: Full dictionary of peer signatures.
            k: Target number of neighbors to select.
            tau: Similarity acceptance threshold tau^(t).
            min_candidates: Minimum candidate peers to return before relaxing.
            return_stats: If True, returns (candidates, stats_dict).
            
        Returns:
            List of (peer_id, similarity_score) tuples sorted descending by score,
            or (candidates, stats_dict) if return_stats is True.
        """
        stats: Dict[str, Any] = {
            "scored_edges": 0,
            "accepted_edges": 0,
            "acceptance_rate": 1.0,
            "control_bytes": 0,
            "relax_steps_applied": 0,
            "final_tau": tau,
        }

        if not self.son.levels:
            # Fallback if no overlay topology built: direct scan
            oracle = ideal_centralized_top_k(query_node_id, query_signature, all_signatures, k=k)
            return (oracle, stats) if return_stats else oracle

        current_tau = tau
        total_scored = 0
        total_accepted = 0
        p_cardinality = int(torch.count_nonzero(query_signature).item()) if query_signature.numel() > 0 else 100
        bytes_per_sig = p_cardinality * 6  # 4-byte index + 2-byte value (Section 3.4)

        for step in range(self.max_relax_steps + 1):
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

                cluster_sims: List[Tuple[ClusterDescriptor, float]] = []
                for cluster in active_clusters:
                    sim = sparse_cosine_similarity(query_signature, cluster.prototype_signature)
                    cluster_sims.append((cluster, sim))
                    total_scored += 1
                    stats["control_bytes"] += bytes_per_sig

                # Filter by threshold current_tau and retain top beam_cap clusters
                cluster_sims.sort(key=lambda x: x[1], reverse=True)
                surviving_clusters = [c for c, sim in cluster_sims if sim >= current_tau][: self.beam_cap]
                total_accepted += len(surviving_clusters)

                # Fallback if no branch passed filter: retain best to avoid dead-ends (Section 3.4)
                if not surviving_clusters and cluster_sims:
                    surviving_clusters = [cluster_sims[0][0]]

                # Collect child exemplars from surviving clusters
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

            # If too few candidates discovered, expand to query's local zone
            if len(candidate_node_ids) < min_candidates:
                query_meta = self.son.get_peer_metadata(query_node_id)
                if query_meta and query_meta.zone_id is not None:
                    if 1 in self.son.levels and query_meta.zone_id in self.son.levels[1]:
                        local_zone = self.son.levels[1][query_meta.zone_id]
                        for m in local_zone.member_ids:
                            if m != query_node_id and m in all_signatures:
                                candidate_node_ids.add(m)

            # Step 4: Max-heap candidate similarity ranking
            candidate_scores: List[Tuple[int, float]] = []
            for cand_id in candidate_node_ids:
                score = sparse_cosine_similarity(query_signature, all_signatures[cand_id])
                candidate_scores.append((cand_id, score))
                total_scored += 1
                stats["control_bytes"] += bytes_per_sig

            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            filtered = [item for item in candidate_scores if item[1] >= current_tau]
            total_accepted += len(filtered)

            # If we collected enough candidates or reached max relaxation, break
            if len(filtered) >= min(k, len(candidate_scores)) or step == self.max_relax_steps:
                candidates = filtered[:k] if len(filtered) >= min_candidates else candidate_scores[:k]
                break
            else:
                # Relax threshold for retry (Section 3.5)
                current_tau -= self.relax_delta
                stats["relax_steps_applied"] += 1

        stats["scored_edges"] = total_scored
        stats["accepted_edges"] = total_accepted
        stats["acceptance_rate"] = (total_accepted / float(max(1, total_scored)))
        stats["final_tau"] = current_tau

        if return_stats:
            return candidates, stats
        return candidates


def ideal_centralized_top_k(
    query_node_id: int,
    query_signature: torch.Tensor,
    all_signatures: Dict[int, torch.Tensor],
    k: int = 5,
) -> List[Tuple[int, float]]:
    """
    Centralized Oracle Baseline: Exact Top-K cosine similarity search over all peers.
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
    
    Recall@K = |Retrieved_K cap Ideal_K| / min(k, |Ideal_K|)
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
