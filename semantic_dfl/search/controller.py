"""
Adaptive Similarity Threshold Controller.

Implements Section 3.5 of the SemanticDFL framework, providing dynamic online tuning
of the cosine similarity threshold tau^(t) to balance candidate selectivity and network drift.
"""

from __future__ import annotations


class AdaptiveThresholdController:
    """
    Dynamically adjusts the local similarity acceptance threshold tau^(t).
    
    Formula:
        tau^{t+1} = clip_{[-1, 1]}(tau^t + eta_tau * (acceptance_rate - phi) - gamma * (1 - drift_proxy))
    where:
        - phi is the target peer acceptance rate in (0, 1)
        - eta_tau is the learning rate step size
        - gamma is the drift compensation slack
        - drift_proxy in [0, 1] captures local model trajectory alignment: rho_i^(t) = cos(u_i^(t), u_i^(t-1))
    """

    def __init__(
        self,
        tau_0: float = 0.5,
        phi: float = 0.5,
        eta_tau: float = 0.05,
        gamma: float = 0.02,
        relax_delta: float = 0.02,
        tau_min_clip: float = -1.0,
        tau_max_clip: float = 1.0,
    ) -> None:
        if not (0.0 < phi < 1.0):
            raise ValueError(f"Target acceptance rate phi must be in (0, 1), got {phi}")

        self.tau = float(tau_0)
        self.phi = phi
        self.eta_tau = eta_tau
        self.gamma = gamma
        self.relax_delta = relax_delta
        self.tau_min_clip = tau_min_clip
        self.tau_max_clip = tau_max_clip

    def update_threshold(
        self,
        acceptance_rate: float,
        drift_proxy: float = 1.0,
    ) -> float:
        """
        Updates the internal threshold based on candidate acceptance rate and model drift.
        
        Args:
            acceptance_rate: Fraction of discovered candidate peers that met the threshold in [0, 1].
            drift_proxy: Measure of model trajectory stability in [0, 1] (1.0 = stable, 0.0 = high drift).
            
        Returns:
            Updated threshold tau^{t+1}.
        """
        # If acceptance is higher than target phi -> increase threshold (be more selective)
        # If model drift is high (drift_proxy < 1.0) -> decrease threshold (allow wider candidate pool)
        delta_rate = self.eta_tau * (acceptance_rate - self.phi)
        delta_drift = self.gamma * (1.0 - max(0.0, min(1.0, drift_proxy)))

        new_tau = self.tau + delta_rate - delta_drift
        self.tau = max(self.tau_min_clip, min(self.tau_max_clip, new_tau))
        return self.tau

    def relax_threshold(self, delta: float | None = None) -> float:
        """
        Relaxes (decreases) the threshold when insufficient candidate peers (< K) are discovered.
        
        Args:
            delta: Optional custom relaxation step (defaults to self.relax_delta = 0.02).
            
        Returns:
            Relaxed threshold.
        """
        step = delta if delta is not None else self.relax_delta
        self.tau = max(self.tau_min_clip, self.tau - step)
        return self.tau

    def reset(self, tau_0: float | None = None) -> None:
        """Resets the controller to the initial threshold."""
        if tau_0 is not None:
            self.tau = float(tau_0)
        else:
            self.tau = 0.5
