import numpy as np
import pytest
import torch
import torch.nn as nn

from PIL import Image

from src.retrieval.retriever import prototype_classification, retrieve_similar, hybrid_decision, run_inference
from src.retrieval.index_refresher import refresh_class
from src.retrieval.add_class import add_class_pure, _measure_backward_transfer

class FakeCollection:
    def __init__(self):
        self.data = {}

    def count(self):
        return len(self.data)

    def upsert(self, embeddings, documents, ids, metadatas):
        for emb, doc, entry_id, metadata in zip(embeddings, documents, ids, metadatas):
            self.data[entry_id] = {'embedding': np.asarray(emb), 'document': doc, 'metadata': metadata}

    def add(self, embeddings, documents, ids, metadatas):
        for entry_id in ids:
            if entry_id in self.data:
                raise ValueError("Duplicate ID")

        self.upsert(embeddings, documents, ids, metadatas)

    def update(self, ids, embeddings):
        for entry_id, emb in zip(ids, embeddings):
            if entry_id in self.data:
                self.data[entry_id]['embedding'] = np.asarray(emb)

    def delete(self, ids=None, where=None):
        if ids is not None:
            for entry_id in ids:
                self.data.pop(entry_id, None)

        if where is not None:
            to_delete = []

            for entry_id, item in self.data.items():
                if all(item['metadata'].get(k) == v for k, v in where.items()):
                    to_delete.append(entry_id)

            for entry_id in to_delete:
                self.data.pop(entry_id, None)

    def get(self, where=None, include=None):
        items = []

        for entry_id, item in self.data.items():
            if where is not None and not all(item['metadata'].get(k) == v for k, v in where.items()):
                continue

            items.append((entry_id, item))

        return {
            'ids': [entry_id for entry_id, item in items],
            'embeddings': [item['embedding'].tolist() for entry_id, item in items],
            'documents': [item['document'] for entry_id, item in items],
            'metadatas': [item['metadata'] for entry_id, item in items]
        }

    def query(self, query_embeddings, n_results, include, where=None):
        query = np.asarray(query_embeddings[0])
        candidates = []

        for entry_id, item in self.data.items():
            metadata = item['metadata']

            if where is not None:
                key, condition = next(iter(where.items()))
                if '$ne' in condition and metadata.get(key) == condition['$ne']:
                    continue

            distance = float(np.sum((query - item['embedding']) ** 2))
            candidates.append((distance, entry_id, item))

        candidates.sort(key=lambda x: x[0])
        candidates = candidates[:n_results]

        return {
            'ids': [[entry_id for distance, entry_id, item in candidates]],
            'documents': [[item['document'] for distance, entry_id, item in candidates]],
            'distances': [[distance for distance, entry_id, item in candidates]],
            'metadatas': [[item['metadata'] for distance, entry_id, item in candidates]]
        }


class DummyModel(nn.Module):
    def forward(self, x):
        return x.mean(dim=(2, 3))


class DummyReplayBuffer:
    def __init__(self):
        self.classes = {}

    def add_class(self, class_name, images_tensor, model, device, label=None):
        self.classes[class_name] = images_tensor


@pytest.fixture
def collections():
    proto = FakeCollection()
    support = FakeCollection()

    proto.upsert(
        embeddings=[[0.0, 0.0], [5.0, 5.0]],
        documents=['class_0', 'class_1'],
        ids=['proto_0', 'proto_1'],
        metadatas=[{'class': 'class_0'}, {'class': 'class_1'}]
    )

    support.upsert(
        embeddings=[[0.0, 0.0], [0.1, 0.1], [5.0, 5.0]],
        documents=['class_0', 'class_0', 'class_1'],
        ids=['s0', 's1', 's2'],
        metadatas=[
            {'class': 'class_0', 'image_path': 'image0.jpg'},
            {'class': 'class_0', 'image_path': 'image1.jpg'},
            {'class': 'class_1', 'image_path': 'image2.jpg'}
        ]
    )

    return proto, support


# PROTOTYPE CLASSIFICATION TESTS

def test_prototype_classification_predicts_nearest_class(collections):
    proto, support = collections

    result = prototype_classification(np.array([0.1, 0.1]), proto)

    assert result['predicted_class'] == 'class_0'
    assert result['confidence'] > 0.5
    assert result['min_distance'] >= 0


def test_prototype_classification_returns_all_class_scores(collections):
    proto, support = collections

    result = prototype_classification(np.array([0.1, 0.1]), proto)

    assert len(result['all_class_scores']) == 2
    assert sum(result['all_class_scores'].values()) == pytest.approx(1.0)


def test_prototype_classification_returns_margin(collections):
    proto, support = collections

    result = prototype_classification(np.array([0.1, 0.1]), proto)

    assert 0 <= result['margin'] <= 1


def test_prototype_classification_handles_single_class():
    proto = FakeCollection()
    proto.upsert([[0.0, 0.0]], ['class_0'], ['p0'], [{'class': 'class_0'}])

    result = prototype_classification(np.array([0.1, 0.1]), proto)

    assert result['predicted_class'] == 'class_0'
    assert result['margin'] == 1.0


def test_prototype_classification_rejects_empty_collection():
    proto = FakeCollection()

    with pytest.raises(ValueError):
        prototype_classification(np.array([0.1, 0.1]), proto)


def test_prototype_classification_rejects_invalid_embedding(collections):
    proto, support = collections

    with pytest.raises(ValueError):
        prototype_classification(np.array([[0.1, 0.1]]), proto)


def test_prototype_classification_rejects_nonfinite_embedding(collections):
    proto, support = collections

    with pytest.raises(ValueError):
        prototype_classification(np.array([float('nan'), 0.1]), proto)


# SIMILARITY RETRIEVAL TESTS

def test_retrieve_similar_returns_nearest_images(collections):
    proto, support = collections

    results = retrieve_similar(np.array([0.0, 0.0]), support, n_results=2)

    assert len(results) == 2
    assert results[0]['class'] == 'class_0'
    assert results[0]['distance'] == pytest.approx(0.0)


def test_retrieve_similar_returns_metadata(collections):
    proto, support = collections

    results = retrieve_similar(np.array([0.0, 0.0]), support, n_results=1)

    assert results[0]['image_path'] == 'image0.jpg'
    assert results[0]['class'] == 'class_0'


def test_retrieve_similar_excludes_query_image(collections):
    proto, support = collections

    results = retrieve_similar(np.array([0.0, 0.0]), support, n_results=2, exclude_image_path='image0.jpg')

    assert all(r['image_path'] != 'image0.jpg' for r in results)


def test_retrieve_similar_handles_empty_collection():
    support = FakeCollection()

    assert retrieve_similar(np.array([0.0, 0.0]), support) == []


def test_retrieve_similar_rejects_invalid_n_results(collections):
    proto, support = collections

    with pytest.raises(ValueError):
        retrieve_similar(np.array([0.0, 0.0]), support, n_results=0)


# HYBRID DECISION TESTS

def test_hybrid_decision_when_models_agree():
    proto_result = {'predicted_class': 'class_0'}
    retrieval_results = [{'class': 'class_0'}, {'class': 'class_0'}, {'class': 'class_1'}]

    result = hybrid_decision(proto_result, retrieval_results)

    assert result['agrees'] is True
    assert result['final_prediction'] == 'class_0'
    assert result['agreement_count'] == 2


def test_hybrid_decision_when_models_disagree():
    proto_result = {'predicted_class': 'class_0'}
    retrieval_results = [{'class': 'class_1'}, {'class': 'class_1'}]

    result = hybrid_decision(proto_result, retrieval_results)

    assert result['agrees'] is False
    assert result['retrieval_majority'] == 'class_1'


def test_hybrid_decision_handles_empty_retrieval():
    proto_result = {'predicted_class': 'class_0'}

    result = hybrid_decision(proto_result, [])

    assert result['final_prediction'] == 'class_0'
    assert result['agrees'] is None
    assert result['retrieval_majority'] is None


def test_run_inference_returns_complete_output(collections):
    proto, support = collections

    result = run_inference(np.array([0.0, 0.0]), proto, support, n_similar=2)

    assert result['prediction'] == 'class_0'
    assert 'confidence' in result
    assert 'similar_cases' in result
    assert 'margin' in result
    assert 'min_distance' in result


# INDEX REFRESH TESTS

def test_refresh_class_updates_support_and_prototype(monkeypatch, tmp_path):
    import src.retrieval.index_refresher as refresher

    image_dir = tmp_path / 'class_0'
    image_dir.mkdir()

    for i in range(2):
        Image.fromarray(np.full((2, 2, 3), i * 100, dtype=np.uint8)).save(image_dir / f'{i}.png')

    monkeypatch.setattr(refresher, 'get_transform', lambda modality, split: lambda image: {'image': torch.tensor(image).permute(2, 0, 1).float()})

    model = DummyModel()
    support = FakeCollection()
    proto = FakeCollection()

    refresh_class('class_0', 'PathMNIST', str(image_dir), 'path', model, proto, support, torch.device('cpu'))

    assert support.count() == 2
    assert proto.count() == 1
    assert proto.get()['metadatas'][0]['n_support'] == 2


def test_refresh_class_rejects_empty_directory(monkeypatch, tmp_path):
    import src.retrieval.index_refresher as refresher

    image_dir = tmp_path / 'empty'
    image_dir.mkdir()

    monkeypatch.setattr(refresher, 'get_transform', lambda modality, split: lambda image: {'image': torch.tensor(image).permute(2, 0, 1).float()})

    with pytest.raises(ValueError):
        refresh_class('class_0', 'PathMNIST', str(image_dir), 'path', DummyModel(), FakeCollection(), FakeCollection(), torch.device('cpu'))


def test_refresh_class_removes_stale_support_ids(monkeypatch, tmp_path):
    import src.retrieval.index_refresher as refresher

    image_dir = tmp_path / 'class_0'
    image_dir.mkdir()
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(image_dir / '0.png')

    monkeypatch.setattr(refresher, 'get_transform', lambda modality, split: lambda image: {'image': torch.tensor(image).permute(2, 0, 1).float()})

    support = FakeCollection()
    proto = FakeCollection()

    support.upsert([[0, 0, 0]], ['class_0'], ['support_PathMNIST_class_0_9'], [{'class': 'class_0', 'dataset': 'PathMNIST'}])

    refresh_class('class_0', 'PathMNIST', str(image_dir), 'path', DummyModel(), proto, support, torch.device('cpu'))

    assert support.count() == 1
    assert support.get()['ids'] == ['support_PathMNIST_class_0_0']


# CLASS ADDITION TESTS

def test_add_class_pure_adds_support_and_prototype(monkeypatch, tmp_path):
    import src.retrieval.add_class as add_module

    image_paths = []

    for i in range(2):
        path = tmp_path / f'image_{i}.png'
        Image.fromarray(np.full((2, 2, 3), i * 100, dtype=np.uint8)).save(path)
        image_paths.append(str(path))

    monkeypatch.setattr(add_module, 'get_transform', lambda modality, split: lambda image: {'image': torch.tensor(image).permute(2, 0, 1).float()})

    proto = FakeCollection()
    support = FakeCollection()
    replay = DummyReplayBuffer()

    result = add_class_pure('class_0', image_paths, 'PathMNIST', 'path', DummyModel(), proto, support, replay, str(tmp_path / 'support'), torch.device('cpu'))

    assert result['n_support'] == 2
    assert proto.count() == 1
    assert support.count() == 2
    assert 'class_0' in replay.classes


def test_add_class_pure_rejects_empty_images():
    with pytest.raises(ValueError):
        add_class_pure('class_0', [], 'PathMNIST', 'path', DummyModel(), FakeCollection(), FakeCollection(), DummyReplayBuffer(), 'support', torch.device('cpu'))


def test_add_class_pure_can_be_repeated(monkeypatch, tmp_path):
    import src.retrieval.add_class as add_module

    image_path = tmp_path / 'image.png'
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(image_path)

    monkeypatch.setattr(add_module, 'get_transform', lambda modality, split: lambda image: {'image': torch.tensor(image).permute(2, 0, 1).float()})

    proto = FakeCollection()
    support = FakeCollection()
    replay = DummyReplayBuffer()
    model = DummyModel()
    save_dir = str(tmp_path / 'support')

    add_class_pure('class_0', [str(image_path)], 'PathMNIST', 'path', model, proto, support, replay, save_dir, torch.device('cpu'))
    add_class_pure('class_0', [str(image_path)], 'PathMNIST', 'path', model, proto, support, replay, save_dir, torch.device('cpu'))

    assert proto.count() == 1
    assert support.count() == 1


# BACKWARD TRANSFER TESTS

def test_backward_transfer_excludes_query_image(monkeypatch):
    import src.retrieval.retriever as retriever_module

    support = FakeCollection()
    support.upsert(
        [[0.0, 0.0], [0.1, 0.1]],
        ['class_0', 'class_0'],
        ['s0', 's1'],
        [{'class': 'class_0', 'image_path': 'a.png'}, {'class': 'class_0', 'image_path': 'b.png'}]
    )

    calls = []

    def fake_retrieve(embedding, collection, n_results=5, exclude_image_path=None):
        calls.append(exclude_image_path)
        return [{'class': 'class_0'}]

    monkeypatch.setattr(retriever_module, 'retrieve_similar', fake_retrieve)

    result = _measure_backward_transfer(DummyModel(), FakeCollection(), support, torch.device('cpu'), n_samples=2)

    assert result == 1.0
    assert set(calls) == {'a.png', 'b.png'}