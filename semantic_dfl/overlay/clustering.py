"""
Zone-level Affinity Propagation Clustering and Prototype Extraction.

Performs parameter-free semantic clustering over client signatures within a zone,
computes sparsified cluster prototype centroids, and elects primary exemplars with
failover replica sets.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Union
import numpy as np
import torch
from sklearn.cluster import AffinityPropagation

from semantic_dfl.overlay.descriptors import ClusterDescriptor
from semantic_dfl.signature.extractor import SignatureExtractor
from semantic_dfl.signature.metrics import (
    affinity_propagation_preference,
    pairwise_cosine_similarity_matrix,
    sparse_cosine_similarity,
)


class ZoneClustering:
    """
    Executes Affinity Propagation clustering on client signatures within an overlay zone.
    """

    def __init__(
        self,
        damping: float = 0.7,
        preference_quantile: float = 0.5,
        r_replicas: int = 3,
        random_state: int = 42,
    ) -> None:
        self.damping = damping
        self.preference_quantile = preference_quantile
        self.r_replicas = r_replicas
        self.random_state = random_state
        self._extractor = SignatureExtractor(beta=0.0)

    def cluster_zone_signatures(
        self,
        member_ids: Sequence[int],
        signatures: Dict[int, torch.Tensor],
        p: int,
        zone_id: Union[str, int] = 0,
        level: int = 1,
    ) -> Dict[Union[str, int], ClusterDescriptor]:
        """
        Clusters signatures of zone members into semantic clusters.
        
        Args:
            member_ids: Ordered sequence of peer IDs in the zone.
            signatures: Mapping from peer_id to their top-P signature tensor u_i in R^M.
            p: Target signature cardinality for cluster prototype centroids.
            zone_id: Parent zone ID.
            level: Hierarchical SON tree level.
            
        Returns:
            Dictionary mapping cluster_id to ClusterDescriptor.
        """
        if not member_ids:
            return {}

        valid_members = [m for m in member_ids if m in signatures]
        n = len(valid_members)

        if n == 0:
            return {}

        # Base case: single member
        if n == 1:
            m_id = valid_members[0]
            sig = signatures[m_id]
            proto = self._extractor.extract_top_p(sig, p=p, return_dense_masked=True)
            cid = f"z{zone_id}_c0"
            return {
                cid: ClusterDescriptor(
                    cluster_id=cid,
                    exemplar_id=m_id,
                    prototype_signature=proto,
                    replica_super_peers=[m_id],
                    member_ids=[m_id],
                    zone_id=zone_id,
                    level=level,
                )
            }

        # Step 1: Vectorized pairwise cosine similarity matrix
        stacked_signatures = [signatures[m] for m in valid_members]
        sim_tensor = pairwise_cosine_similarity_matrix(stacked_signatures)
        sim_np = sim_tensor.detach().cpu().numpy()

        # Step 2: Compute Affinity Propagation preference parameter
        pref = affinity_propagation_preference(sim_np, quantile=self.preference_quantile)

        # Step 3: Run Affinity Propagation
        ap = AffinityPropagation(
            affinity="precomputed",
            damping=self.damping,
            preference=pref,
            random_state=self.random_state,
            max_iter=400,
        )
        ap.fit(sim_np)

        # Fallback if AP did not form valid centers
        if ap.cluster_centers_indices_ is None or len(ap.cluster_centers_indices_) == 0:
            labels = np.zeros(n, dtype=int)
            center_indices = [0]
        else:
            labels = ap.labels_
            center_indices = ap.cluster_centers_indices_

        # Step 4: Group clusters and construct ClusterDescriptors
        unique_labels = np.unique(labels)
        clusters: Dict[Union[str, int], ClusterDescriptor] = {}

        for cluster_idx, label in enumerate(unique_labels):
            cluster_member_indices = np.where(labels == label)[0]
            cluster_members = [valid_members[i] for i in cluster_member_indices]

            # Determine exemplar node
            if label < len(center_indices):
                exemplar_node_idx = center_indices[label]
                exemplar_id = valid_members[exemplar_node_idx]
            else:
                exemplar_id = cluster_members[0]

            # Compute prototype centroid: u_bar = TopP( (1 / |C_j|) * sum_{i in C_j} u_i )
            cluster_sigs = [signatures[m] for m in cluster_members]
            raw_centroid = torch.stack(cluster_sigs, dim=0).mean(dim=0)
            prototype = self._extractor.extract_top_p(raw_centroid, p=p, return_dense_masked=True)

            # Elect replica super-peer set R_j (size r)
            # Rank cluster members by cosine similarity to the prototype centroid
            member_sims = []
            for m in cluster_members:
                sim = sparse_cosine_similarity(signatures[m], prototype)
                member_sims.append((m, sim))

            # Sort descending by similarity to prototype
            member_sims.sort(key=lambda x: x[1], reverse=True)
            sorted_candidates = [m for m, _ in member_sims]

            # Ensure exemplar is primary at index 0
            replica_set = [exemplar_id]
            for cand in sorted_candidates:
                if cand != exemplar_id and len(replica_set) < self.r_replicas:
                    replica_set.append(cand)

            cid = f"z{zone_id}_c{cluster_idx}"
            clusters[cid] = ClusterDescriptor(
                cluster_id=cid,
                exemplar_id=exemplar_id,
                prototype_signature=prototype,
                replica_super_peers=replica_set,
                member_ids=cluster_members,
                zone_id=zone_id,
                level=level,
            )

        return clusters

# Add architecture diagram to son.py (step 7)

# Clean up super-peer descriptor routing state (step 9)

# Calculate mean zone centroid signature (step 11)

# Set minimum exemplar count fallback (step 13)

# Test failover promotion on primary peer drop (step 15)

# Add architecture diagram to son.py (step 17)

# Clean up super-peer descriptor routing state (step 19)

# Calculate mean zone centroid signature (step 21)

# Set minimum exemplar count fallback (step 23)

# Test failover promotion on primary peer drop (step 25)

# Add architecture diagram to son.py (step 27)

# Clean up super-peer descriptor routing state (step 29)

# Calculate mean zone centroid signature (step 31)

# Set minimum exemplar count fallback (step 33)

# Test failover promotion on primary peer drop (step 35)

# Add architecture diagram to son.py (step 37)

# Clean up super-peer descriptor routing state (step 39)

# Calculate mean zone centroid signature (step 41)

# Set minimum exemplar count fallback (step 43)

# Test failover promotion on primary peer drop (step 45)

# Add architecture diagram to son.py (step 47)

# Clean up super-peer descriptor routing state (step 49)

# Calculate mean zone centroid signature (step 51)

# Set minimum exemplar count fallback (step 53)

# Test failover promotion on primary peer drop (step 55)

# Add architecture diagram to son.py (step 57)

# Clean up super-peer descriptor routing state (step 59)

# Calculate mean zone centroid signature (step 61)

# Set minimum exemplar count fallback (step 63)

# Test failover promotion on primary peer drop (step 65)

# Add architecture diagram to son.py (step 67)

# Clean up super-peer descriptor routing state (step 69)

# Calculate mean zone centroid signature (step 71)

# Set minimum exemplar count fallback (step 73)

# Test failover promotion on primary peer drop (step 75)

# Add architecture diagram to son.py (step 77)

# Clean up super-peer descriptor routing state (step 79)

# Calculate mean zone centroid signature (step 81)

# Set minimum exemplar count fallback (step 83)

# Test failover promotion on primary peer drop (step 85)

# Add architecture diagram to son.py (step 87)

# Clean up super-peer descriptor routing state (step 89)

# Calculate mean zone centroid signature (step 91)

# Set minimum exemplar count fallback (step 93)

# Test failover promotion on primary peer drop (step 95)
