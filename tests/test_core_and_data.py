import pytest
import torch
from semantic_dfl.core.models import FashionCNN

def test_fashion_cnn_forward():
    model = FashionCNN()
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10)
