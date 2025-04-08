"""
Unit Tests for SemanticDFL Core Neural Networks and Data Partitioning Modules.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset

from semantic_dfl.core.models import (
    FashionCNN,
    ResNet18Small,
    MLP2NN,
    get_flattened_params,
    set_flattened_params,
    get_flattened_grads,
    get_model_fingerprint,
)
from semantic_dfl.data.partitioning import (
    DirichletPartitioner,
    PathologicalPartitioner,
    IIDPartitioner,
)
from semantic_dfl.data.datasets import create_client_dataloaders


class TestCoreModels:
    """Tests for neural network architectures and parameter manipulation."""

    @pytest.mark.parametrize(
        "model_cls, input_shape, num_classes",
        [
            (FashionCNN, (4, 1, 28, 28), 10),
            (FashionCNN, (2, 1, 28, 28), 5),
            (ResNet18Small, (2, 3, 32, 32), 10),
            (ResNet18Small, (2, 1, 28, 28), 10),
            (MLP2NN, (4, 784), 10),
            (MLP2NN, (3, 1, 28, 28), 10),
        ],
    )
    def test_model_forward_shapes(self, model_cls, input_shape, num_classes):
        if model_cls == FashionCNN:
            model = FashionCNN(in_channels=input_shape[1], num_classes=num_classes)
        elif model_cls == ResNet18Small:
            model = ResNet18Small(in_channels=input_shape[1], num_classes=num_classes)
        elif model_cls == MLP2NN:
            model = MLP2NN(input_dim=784, num_classes=num_classes)

        x = torch.randn(*input_shape)
        output = model(x)
        assert output.shape == (input_shape[0], num_classes)

    def test_parameter_flatten_unflatten_consistency(self):
        model = FashionCNN(in_channels=1, num_classes=10)
        original_params = get_flattened_params(model)
        
        # Verify total elements match parameter sum
        expected_numel = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert original_params.numel() == expected_numel

        # Perturb parameters with known random tensor
        new_params = torch.randn_like(original_params)
        set_flattened_params(model, new_params)

        # Extract flattened parameters again and assert exact match
        extracted_params = get_flattened_params(model)
        assert torch.allclose(new_params, extracted_params)

    def test_parameter_unflatten_dimension_mismatch(self):
        model = FashionCNN()
        bad_params = torch.randn(10)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            set_flattened_params(model, bad_params)

    def test_flattened_gradients_extraction(self):
        model = FashionCNN(in_channels=1, num_classes=10)
        x = torch.randn(2, 1, 28, 28)
        y = torch.tensor([0, 1])
        criterion = nn.CrossEntropyLoss()

        # Before backward, grads should be zero vectors
        grads_before = get_flattened_grads(model)
        assert grads_before.shape == (sum(p.numel() for p in model.parameters()),)
        assert torch.all(grads_before == 0.0)

        # Perform forward and backward pass
        out = model(x)
        loss = criterion(out, y)
        loss.backward()

        # Extract grads after backward
        grads_after = get_flattened_grads(model)
        assert grads_after.shape == grads_before.shape
        assert not torch.all(grads_after == 0.0)

    def test_model_fingerprint(self):
        torch.manual_seed(42)
        model1 = FashionCNN()
        
        torch.manual_seed(42)
        model2 = FashionCNN()

        torch.manual_seed(999)
        model3 = FashionCNN()

        assert get_model_fingerprint(model1) == get_model_fingerprint(model2)
        assert get_model_fingerprint(model1) != get_model_fingerprint(model3)


class TestDataPartitioning:
    """Tests for IID and Non-IID dataset partitioners."""

    @pytest.fixture
    def synthetic_dataset(self):
        num_samples = 1000
        num_classes = 10
        # Equal distribution of 100 samples per class
        targets = np.repeat(np.arange(num_classes), num_samples // num_classes)
        return targets

    def test_iid_partitioner(self, synthetic_dataset):
        num_clients = 10
        partitioner = IIDPartitioner(seed=42)
        splits = partitioner.partition(synthetic_dataset, num_clients=num_clients)

        assert len(splits) == num_clients
        
        # Verify complete coverage and no duplicate sample indices
        all_indices = []
        for client_id, idxs in splits.items():
            assert len(idxs) == len(synthetic_dataset) // num_clients
            all_indices.extend(idxs)

        assert sorted(all_indices) == list(range(len(synthetic_dataset)))

    @pytest.mark.parametrize("alpha", [0.1, 0.5, 5.0])
    def test_dirichlet_partitioner(self, synthetic_dataset, alpha):
        num_clients = 8
        partitioner = DirichletPartitioner(alpha=alpha, seed=42, min_require_size=5)
        splits = partitioner.partition(synthetic_dataset, num_clients=num_clients)

        assert len(splits) == num_clients

        all_indices = []
        for client_id, idxs in splits.items():
            assert len(idxs) >= 5
            all_indices.extend(idxs)

        # Complete coverage check
        assert len(all_indices) == len(synthetic_dataset)
        assert set(all_indices) == set(range(len(synthetic_dataset)))

    def test_pathological_partitioner(self, synthetic_dataset):
        num_clients = 5
        classes_per_client = 2
        partitioner = PathologicalPartitioner(classes_per_client=classes_per_client, seed=42)
        splits = partitioner.partition(synthetic_dataset, num_clients=num_clients)

        assert len(splits) == num_clients

        all_indices = []
        for client_id, idxs in splits.items():
            client_labels = set(synthetic_dataset[idxs])
            # Each client must have at most classes_per_client unique labels
            assert len(client_labels) <= classes_per_client
            all_indices.extend(idxs)

        assert set(all_indices) == set(range(len(synthetic_dataset)))

    def test_create_client_dataloaders(self):
        x = torch.randn(100, 1, 28, 28)
        y = torch.randint(0, 10, (100,))
        dataset = TensorDataset(x, y)

        splits = {
            0: list(range(0, 50)),
            1: list(range(50, 100)),
        }

        loaders = create_client_dataloaders(dataset, splits, batch_size=16)
        assert len(loaders) == 2
        
        batch_count_0 = sum(1 for _ in loaders[0])
        batch_count_1 = sum(1 for _ in loaders[1])
        
        # 50 samples with batch_size 16 => ceil(50/16) = 4 batches
        assert batch_count_0 == 4
        assert batch_count_1 == 4
