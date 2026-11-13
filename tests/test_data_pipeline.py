import json
import pytest
import numpy as np
import torch
from pathlib import Path

from src.data.datasets import PathMNISTDataset, build_class_index
from src.data.preprocessing import get_transform

EXPECTED_CLASSES = set(range(9))
EXPECTED_IMAGE_SHAPE = (3, 224, 224)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUPPORT_DIR = PROJECT_ROOT / "data" / "support"
MANIFEST_PATH = PROJECT_ROOT / "data" / "support_manifest.json"

@pytest.fixture(scope="module")
def actual_datasets():

    datasets = {}
    for split in ["train", "val", "test"]:

        mode = "train" if split == "train" else "val"
        transform = get_transform("path", mode)

        try:
            dataset = PathMNISTDataset(split=split, transform=transform)

        except Exception as exc:
            pytest.fail(f"Could not load PathMNIST {split} dataset. Check that the dataset is downloaded and the configured root is correct.\nError: {exc}")

        datasets[split] = dataset

    return datasets

# 1. Dataset

@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_dataset_is_not_empty(actual_datasets, split):

    dataset = actual_datasets[split]
    assert len(dataset) > 0


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_dataset_has_expected_classes(actual_datasets, split):

    dataset = actual_datasets[split]
    labels = np.asarray(dataset.base.labels).flatten()

    assert set(np.unique(labels)) == EXPECTED_CLASSES


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_dataset_image_dimensions(actual_datasets, split):

    dataset = actual_datasets[split]
    images = dataset.base.imgs

    assert images.ndim == 4
    assert images.shape[1:] == (224, 224, 3)


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_dataset_image_count_matches_labels(actual_datasets, split):

    dataset = actual_datasets[split]

    assert len(dataset.base.imgs) == len(dataset.base.labels)
    assert len(dataset) == len(dataset.base.labels)

# 2. Class distribution

@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_all_classes_have_samples(actual_datasets, split):

    dataset = actual_datasets[split]
    labels = np.asarray(dataset.base.labels).flatten()

    unique, counts = np.unique(labels,return_counts=True)
    distribution = dict(zip(unique.tolist(), counts.tolist()))

    assert set(distribution.keys()) == EXPECTED_CLASSES

    for _, count in distribution.items():
        assert count > 0

# 3. Actual image retrieval

@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_actual_image_retrieval(actual_datasets, split):

    dataset = actual_datasets[split]
    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert isinstance(label, int)

    assert image.shape == EXPECTED_IMAGE_SHAPE
    assert image.dtype == torch.float32

    assert 0 <= label <= 8
    assert torch.isfinite(image).all()

# 4. Class-index

@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_class_index_contains_all_classes(actual_datasets, split):

    dataset = actual_datasets[split]
    class_index = build_class_index(dataset)

    assert set(class_index.keys()) == EXPECTED_CLASSES


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_class_index_matches_original_labels(actual_datasets, split):

    dataset = actual_datasets[split]
    class_index = build_class_index(dataset)

    original_labels = np.asarray(dataset.base.labels).flatten()

    for cls, indices in class_index.items():
        for idx in indices:
            assert int(original_labels[idx]) == cls


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_class_index_has_no_missing_or_duplicate_indices(actual_datasets, split):

    dataset = actual_datasets[split]
    class_index = build_class_index(dataset)

    all_indices = [idx for indices in class_index.values() for idx in indices]

    assert len(all_indices) == len(dataset)
    assert len(set(all_indices)) == len(dataset)

    assert sorted(all_indices) == list(range(len(dataset)))

# 5. Preprocessing

def test_training_preprocessing(actual_datasets):

    dataset = actual_datasets["train"]
    image, _ = dataset[0]

    assert image.shape == EXPECTED_IMAGE_SHAPE
    assert image.dtype == torch.float32
    assert torch.isfinite(image).all()


@pytest.mark.parametrize("split", ["val", "test"])
def test_evaluation_preprocessing(actual_datasets, split):

    dataset = actual_datasets[split]
    image, _ = dataset[0]

    assert image.shape == EXPECTED_IMAGE_SHAPE
    assert image.dtype == torch.float32
    assert torch.isfinite(image).all()


def test_evaluation_transform_is_deterministic(actual_datasets):
    dataset = actual_datasets["test"]

    # Retrieve the same underlying image twice.
    image_1, label_1 = dataset[0]
    image_2, label_2 = dataset[0]

    assert label_1 == label_2
    assert torch.equal(image_1, image_2)

# 6. Support manifest audit

def test_support_manifest_if_available(actual_datasets):

    if not MANIFEST_PATH.exists():
        pytest.skip("Support manifest has not been generated yet.")

    assert SUPPORT_DIR.exists(), "Support directory is missing."

    with open(MANIFEST_PATH, "r") as file:
        manifest = json.load(file)

    assert manifest["dataset"] == "PathMNIST"
    assert manifest["source_split"] == "train"

    train_dataset = actual_datasets["train"]
    train_labels = np.asarray(train_dataset.base.labels).flatten()

    for cls_str, samples in manifest["classes"].items():

        cls = int(cls_str)
        assert cls in EXPECTED_CLASSES

        class_dir = SUPPORT_DIR / f"class_{cls}"
        assert class_dir.exists()

        for sample in samples:
            assert sample["source_split"] == "train"

            source_idx = sample["source_index"]
            assert 0 <= source_idx < len(train_dataset)

            assert int(train_labels[source_idx]) == cls

            image_path = class_dir / sample["file_name"]
            assert image_path.exists()


# 7. Summary

def test_dataset_split_sizes_are_consistent(actual_datasets):

    train = actual_datasets["train"]
    val = actual_datasets["val"]
    test = actual_datasets["test"]

    print("\nPathMNIST dataset sizes:")
    print("Train:", len(train))
    print("Validation:", len(val))
    print("Test:", len(test))

    assert len(train) > len(val)
    assert len(train) > len(test)