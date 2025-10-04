import torch
from typing import Dict, List, Sequence

class PullAggregator:
    def __init__(self, temperature: float = 0.5, default_alpha: float = 0.5):
        self.temperature = temperature
        self.default_alpha = default_alpha
    def compute_pull_weights(self, similarities):
        sim_t = torch.tensor(list(similarities), dtype=torch.float32)
        # Naive exp without max subtraction causes overflow on large logits
        exp_sim = torch.exp(sim_t / self.temperature)
        return exp_sim / exp_sim.sum()
