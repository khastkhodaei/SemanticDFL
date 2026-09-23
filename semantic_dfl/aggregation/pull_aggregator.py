"""
Row-Stochastic Softmax Pull Aggregator, Personalization Mixing, and Freshness Guard.

Implements Section 2.1, Section 3.6, and Appendix E.3 of the SemanticDFL framework:
1. Calculates temperature-scaled softmax weights W_ij over candidate peer similarities.
2. Applies Freshness Guard sim(u_i^(t), u_j^(t')) >= T_fresh for delayed neighbor replies.
3. Forms a collaborative semantic neighbor anchor m_i^(t).
4. Interpolates local client weights with anchor using personalization coefficient alpha_i = (eta * psi_i) / (1 + eta * psi_i).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn.functional as F

from semantic_dfl.signature.metrics import sparse_cosine_similarity


def compute_alpha_from_psi(psi: float = 2.0, eta: float = 1.0) -> float:
    """
    Computes the personalization mixing weight alpha_i from coupling strength psi_i (Appendix E.3):
        alpha_i = (eta * psi_i) / (1 + eta * psi_i)
        
    Example:
        psi = 2.0, eta = 1.0 => alpha = 2 / 3 = 0.667
        psi = 0.0 => alpha = 0.0 (pure local learning)
    """
    if psi < 0.0:
        raise ValueError(f"Personalization coupling strength psi must be >= 0, got {psi}")
    return float((eta * psi) / (1.0 + eta * psi))


class PullAggregator:
    """
    Computes personalized pull-based model updates with freshness guards for decentralized learning.
    """

    def __init__(
        self,
        temperature: float = 0.5,
        default_alpha: float = 0.67,
        psi: Optional[float] = None,
        eta: float = 1.0,
        t_fresh: float = 0.6,
        max_staleness: int = 2,
    ) -> None:
        if temperature <= 0.0:
            raise ValueError(f"Temperature kappa must be strictly positive, got {temperature}")

        self.temperature = temperature
        self.eta = eta
        self.psi = psi
        if psi is not None:
            self.default_alpha = compute_alpha_from_psi(psi=psi, eta=eta)
        else:
            if not (0.0 <= default_alpha <= 1.0):
                raise ValueError(f"Personalization alpha must be in [0, 1], got {default_alpha}")
            self.default_alpha = default_alpha

        self.t_fresh = t_fresh
        self.max_staleness = max_staleness

    def compute_pull_weights(
        self,
        similarities: Sequence[float] | Dict[int, float] | torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes row-stochastic softmax weights W_ij with temperature scaling:
            W_{ij} = exp(s_{ij} / kappa) / sum_k exp(s_{ik} / kappa)
            
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
        weights = F.softmax(sim_tensor / self.temperature, dim=0)
        return weights

    def filter_fresh_replies(
        self,
        query_signature: torch.Tensor,
        candidate_signatures: Dict[int, torch.Tensor],
        candidate_rounds: Dict[int, int],
        current_round: int,
    ) -> Set[int]:
        """
        Section 3.6 Freshness Guard:
        Accepts replies from rounds t' in [t - Delta_max, t) only if sim(u_i^(t), u_j^(t')) >= T_fresh.
        """
        accepted_peers: Set[int] = set()

        for peer_id, sig in candidate_signatures.items():
            r = candidate_rounds.get(peer_id, current_round)
            staleness = current_round - r

            if staleness < 0:
                continue
            elif staleness == 0:
                accepted_peers.add(peer_id)
            elif staleness <= self.max_staleness:
                # Check freshness guard
                sim = sparse_cosine_similarity(query_signature, sig)
                if sim >= self.t_fresh:
                    accepted_peers.add(peer_id)
            else:
                # Exceeds max staleness -> discard (Assumption 5)
                pass

        return accepted_peers

    def aggregate(
        self,
        client_weights: torch.Tensor,
        neighbor_weights: Sequence[torch.Tensor] | Dict[int, torch.Tensor],
        similarities: Sequence[float] | Dict[int, float],
        alpha: Optional[float] = None,
        psi: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Aggregates neighbor models into a collaborative anchor and mixes with local client state:
            m_i^(t) = sum_j W_{ij} * w_j^(t)
            w_i^(new) = (1 - alpha_i) * w_tilde_i + alpha_i * m_i^(t)
            
        Args:
            client_weights: Flat local model parameter tensor w_tilde_i in R^M.
            neighbor_weights: Sequence or Dict of flat neighbor parameter tensors w_j in R^M.
            similarities: Corresponding similarity scores s_ij for each neighbor.
            alpha: Optional custom personalization mixing coefficient.
            psi: Optional custom coupling strength (computes alpha = (eta*psi)/(1 + eta*psi)).
            
        Returns:
            Updated local model parameter tensor w_i^(new) in R^M.
        """
        if psi is not None:
            interp_alpha = compute_alpha_from_psi(psi=psi, eta=self.eta)
        elif alpha is not None:
            interp_alpha = alpha
        else:
            interp_alpha = self.default_alpha

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

        # Form collaborative anchor m_i^(t) = sum_j W_{ij} * w_j on client device
        stacked_neighbors = torch.stack([w.detach().float().to(flat_client.device) for w in neighbors_list], dim=0)
        
        # Reshape weights for broadcasting: (num_neighbors, 1)
        weights_expanded = pull_weights.unsqueeze(1).to(flat_client.device)
        anchor = torch.sum(weights_expanded * stacked_neighbors, dim=0)

        # Proximal mixture: w_new = (1 - alpha) * w_tilde + alpha * anchor
        updated_weights = (1.0 - interp_alpha) * flat_client + interp_alpha * anchor
        return updated_weights
