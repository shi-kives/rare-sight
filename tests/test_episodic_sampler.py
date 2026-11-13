import numpy as np
import torch
import pytest
from medmnist import PathMNIST
from torchvision import transforms
from src.models.episodic_sampler import sample_episode


@pytest.fixture(scope="module")
def pathmnist_data():

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = PathMNIST(split="train", root="data/raw", download=True, transform=transform)

    # Build global class -> dataset indices mapping.
    labels = np.asarray(dataset.labels).reshape(-1)

    class_index = {int(class_id): np.where(labels == class_id)[0].tolist() for class_id in np.unique(labels)}

    return dataset, class_index


def test_pathmnist_episode_shapes(pathmnist_data):
    dataset, class_index = pathmnist_data

    base_classes = [0, 1, 2, 3, 4]

    support, query, labels, episode_classes = sample_episode(dataset=dataset,class_index=class_index, base_classes=base_classes, n_way=5, k_shot=5, q_query=15, device="cpu", return_classes=True)

    assert support.shape == (5, 5, 3, 28, 28)
    assert query.shape == (75, 3, 28, 28)
    assert labels.shape == (75,)

    assert len(episode_classes) == 5
    assert set(episode_classes).issubset(set(base_classes))

    assert labels.min().item() == 0
    assert labels.max().item() == 4


def test_pathmnist_query_class_balance(pathmnist_data):
    dataset, class_index = pathmnist_data

    support, query, labels, episode_classes = sample_episode(dataset=dataset,class_index=class_index, base_classes=[0, 1, 2, 3, 4], n_way=5, k_shot=5, q_query=15, device="cpu", return_classes=True)

    for local_class in range(5):
        assert (labels == local_class).sum().item() == 15


def test_pathmnist_samples_are_valid(pathmnist_data):
    dataset, class_index = pathmnist_data

    support, query, labels, episode_classes = sample_episode(dataset=dataset, class_index=class_index, base_classes=[0, 1, 2, 3, 4], n_way=5, k_shot=5, q_query=15, device="cpu", return_classes=True)

    assert torch.isfinite(support).all()
    assert torch.isfinite(query).all()

    assert support.min() >= 0
    assert support.max() <= 1

    assert query.min() >= 0
    assert query.max() <= 1


def test_pathmnist_classes_have_enough_samples(pathmnist_data):
    dataset, class_index = pathmnist_data

    required_samples = 5 + 15

    for class_id in [0, 1, 2, 3, 4]:
        assert class_id in class_index
        assert len(class_index[class_id]) >= required_samples
