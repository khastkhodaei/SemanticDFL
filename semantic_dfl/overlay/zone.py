import collections
import hashlib
from typing import Dict, List, Sequence, Set
from semantic_dfl.overlay.descriptors import ZoneDescriptor

class ZoneManager:
    def __init__(self, max_zone_size: int = 20, radius_ttl: int = 2, seed: int = 42):
        self.max_zone_size = max_zone_size
        self.radius_ttl = radius_ttl
        self.seed = seed
