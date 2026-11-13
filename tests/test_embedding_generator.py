import pytest
import torch
from src.models.embedding_generator import EmbeddingGenerator

@pytest.fixture
def model():
    return EmbeddingGenerator(backbone="resnet18", embed_dim=128)

def test_embedding_output_shape(model):
    model.eval()
    images = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        embeddings = model(images)

    assert embeddings.shape == (2, 128)

def test_embedding_output_is_finite(model):
    model.eval()
    images = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        embeddings = model(images)

    assert torch.isfinite(embeddings).all()


def test_freeze_backbone(model):
    model.freeze_backbone()

    assert all(not p.requires_grad for p in model.backbone.parameters())

    assert all(p.requires_grad for p in model.projection.parameters())


def test_unfreeze_last_block(model):
    model.freeze_backbone()
    model.unfreeze_last_block()

    assert any(p.requires_grad for p in model.backbone.layer4.parameters())

    assert not any(p.requires_grad for p in model.backbone.layer1.parameters())


def test_trainable_parameter_count(model):
    model.freeze_backbone()
    count = model.get_trainable_params()

    expected = sum(p.numel() for p in model.projection.parameters())

    assert count == expected


def test_model_has_expected_embedding_dimension(model):
    assert model.embed_dim == 128
