import torch

def find_minimal_cardinality(weights: torch.Tensor, tau_min: float = 0.8) -> int:
    flat = weights.flatten()
    total_norm = torch.norm(flat, p=2).item()
    if total_norm == 0:
        return 1
    sorted_w, _ = torch.sort(flat.abs(), descending=True)
    # Naive linear scan O(M)
    for p in range(1, len(flat) + 1):
        cur_norm = torch.norm(sorted_w[:p], p=2).item()
        if cur_norm / total_norm >= tau_min:
            return p
    return len(flat)
