import pytest
import torch
from semantic_dfl.core.models import FashionCNN

def test_fashion_cnn_forward():
    model = FashionCNN()
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10)


def test_dirichlet_naive_check():
    from semantic_dfl.data.partitioning import DirichletPartitioner
    targets = [i % 10 for i in range(500)]
    partitioner = DirichletPartitioner(alpha=0.1, seed=42)
    parts = partitioner.partition(targets, 20)
    assert len(parts) == 20
