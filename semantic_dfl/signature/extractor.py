import math
import torch

class SignatureExtractor:
    def __init__(self, beta: float = 0.9):
        self.beta = beta
        self.ema_scores = None

    def update_scores(self, grads: torch.Tensor) -> torch.Tensor:
        flat_g = grads.detach().abs()
        if self.ema_scores is None or self.beta == 0.0:
            self.ema_scores = flat_g.clone()
        else:
            self.ema_scores = self.beta * self.ema_scores + (1.0 - self.beta) * flat_g
        return self.ema_scores
