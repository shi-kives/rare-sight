import os
import json
import torch
import numpy as np
import chromadb
from tqdm import tqdm
from PIL import Image
from src.models.embedding_generator import EmbeddingGenerator
from src.data.preprocessing import get_transform


def build_support_index(model, support_root, class_names, modality, chroma_path, dataset_tag, device):
    client = chromadb.PersistentClient(path=chroma_path)
    proto_col = client.get_or_create_collection(name='prototypes', metadata={'hnsw:space': 'l2'})
    support_col = client.get_or_create_collection(name='support_images', metadata={'hnsw:space': 'l2'})
    transform = get_transform(modality, 'test')

    original_modes = {module: module.training for module in model.modules()}

    try:
        model.eval()

        for raw_class_name in tqdm(class_names, desc=f'building {dataset_tag} index'):
            if dataset_tag == 'PathMNIST':
                class_name = f'class_{raw_class_name}'
            else:
                class_name = str(raw_class_name)

            class_dir = os.path.join(support_root, class_name)

            if not os.path.isdir(class_dir):
                print(f"class not found: {class_dir}. skipping this class")
                continue

            image_files = sorted([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

            if not image_files:
                print(f"no images found in: {class_dir}. skipping this class")
                continue

            class_embeddings = []
            print(f"working on class: {class_name}")

            for i, fname in enumerate(image_files):
                path = os.path.join(class_dir, fname)
                image = Image.open(path).convert('RGB')
                np_image = np.array(image)
                tensor = transform(image=np_image)['image']
                tensor = tensor.unsqueeze(0).to(device)

                with torch.inference_mode():
                    emb = model(tensor)

                if emb.ndim != 2 or emb.shape[0] != 1:
                    raise ValueError(f"Expected embedding shape (1, D), got {tuple(emb.shape)}.")

                np_emb = emb[0].cpu().numpy().astype(np.float32)

                if not np.all(np.isfinite(np_emb)):
                    raise ValueError(f"Non-finite embedding generated for: {path}")

                class_embeddings.append(np_emb)

                entry_id = f"support_{dataset_tag}_{class_name}_{i}"
                support_col.upsert(
                    embeddings=[np_emb.tolist()],
                    documents=[class_name],
                    ids=[entry_id],
                    metadatas=[{
                        'class': class_name,
                        'image_path': os.path.abspath(path),
                        'dataset': dataset_tag,
                        'index': i,
                        'filename': fname
                    }]
                )

            prototype = np.mean(np.stack(class_embeddings), axis=0).astype(np.float32)

            if not np.all(np.isfinite(prototype)):
                raise ValueError(f"Non-finite prototype generated for: {class_name}")

            prototype_id = f"proto_{dataset_tag}_{class_name}"

            proto_col.upsert(
                embeddings=[prototype.tolist()],
                documents=[class_name],
                ids=[prototype_id],
                metadatas=[{
                    'class': class_name,
                    'dataset': dataset_tag,
                    'n_support': len(class_embeddings)
                }]
            )

            # Remove support records left over from an older, larger version of this class.
            existing = support_col.get(
                where={'$and': [{'dataset': dataset_tag}, {'class': class_name}]},
                include=['metadatas']
            )

            expected_ids = {f"support_{dataset_tag}_{class_name}_{i}" for i in range(len(image_files))}
            stale_ids = [
                entry_id for entry_id, metadata in zip(existing['ids'], existing['metadatas'])
                if entry_id not in expected_ids
            ]

            if stale_ids:
                support_col.delete(ids=stale_ids)

    finally:
        for module, training_mode in original_modes.items():
            module.training = training_mode

    print("-- index built --")
    print("prototypes:", proto_col.count())
    print("support images:", support_col.count())
    return proto_col, support_col


if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pathmnist_split_path = os.path.join(project_root, 'data', 'processed', 'splits', 'medmnist_split.json')

    with open(pathmnist_split_path, 'r', encoding='utf-8') as file:
        data_pathmnist = json.load(file)
        print("data loaded:", data_pathmnist)

    # The backbone must match the checkpoint architecture.
    model = EmbeddingGenerator('resnet18', embed_dim=128)

    # Replace this with the actual ResNet18 checkpoint filename in your models directory.
    checkpoint_path = os.path.join(project_root, 'models', 'best_model_resnet18.pt')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    chroma_path = os.path.join(project_root, 'chroma_store')

    pathmnist_support_root = os.path.join(project_root, 'data', 'processed', 'support_images', 'pathmnist')

    build_support_index(
        model=model,
        support_root=pathmnist_support_root,
        class_names=data_pathmnist['novel_classes'],
        modality='path',
        chroma_path=chroma_path,
        dataset_tag='PathMNIST',
        device=device
    )