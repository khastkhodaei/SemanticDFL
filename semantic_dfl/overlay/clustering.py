import numpy as np
from sklearn.cluster import AffinityPropagation
class ZoneClustering:
    def __init__(self, damping: float = 0.7, max_iter: int = 200, seed: int = 42):
        self.damping = damping
        self.max_iter = max_iter
        self.seed = seed
