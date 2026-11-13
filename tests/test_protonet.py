import torch
import torch.nn as nn
from src.models.proto_net import (compute_prototypes, prototypical_loss)

def test_compute_prototypes_shape():
    embeddings = torch.randn(20, 128)
    prototypes = compute_prototypes(embeddings, n_way=4, k_shot=5)

    assert prototypes.shape == (4, 128)


def test_compute_prototypes_calculation():
    embeddings = torch.tensor([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0]
    ])

    prototypes = compute_prototypes(embeddings, n_way=2, k_shot=2)

    expected = torch.tensor([
        [2.0, 3.0],
        [6.0, 7.0]
    ])

    assert torch.allclose(prototypes, expected)

class SimpleEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.projection = nn.Linear(3 * 8 * 8, 16)

    def forward(self, x):
        return self.projection(self.flatten(x))

def test_prototypical_loss_returns_valid_outputs():

    model = SimpleEmbeddingModel()

    support = torch.randn(3, 2, 3, 8, 8)
    query = torch.randn(12, 3, 8, 8)

    labels = torch.tensor([0] * 4 + [1] * 4 + [2] * 4, dtype=torch.long)

    loss, accuracy, prototypes = prototypical_loss(model, support, query, labels, n_way=3,k_shot=2)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert 0.0 <= accuracy <= 1.0
    assert prototypes.shape == (3, 16)


def test_prototypical_loss_supports_backpropagation():
    model = SimpleEmbeddingModel()

    support = torch.randn(3, 2, 3, 8, 8)
    query = torch.randn(12, 3, 8, 8)

    labels = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)

    loss, _, _ = prototypical_loss(model, support, query, labels, 3, 2)

    loss.backward()

    assert model.projection.weight.grad is not None
    assert torch.isfinite(model.projection.weight.grad).all()
