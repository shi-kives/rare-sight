import os
import shutil
import torch
import numpy as np
from PIL import Image
from src.data.preprocessing import get_transform
from src.retrieval.index_refresher import refresh_all


def _embed_images(model, image_paths, modality, device):
    if not image_paths:
        raise ValueError("image_paths cannot be empty.")

    transform = get_transform(modality, 'test')
    embeddings = []

    original_modes = {module: module.training for module in model.modules()}

    try:
        model.eval()

        with torch.inference_mode():
            for path in image_paths:
                image = Image.open(path).convert('RGB')
                image_np = np.array(image)
                tensor = transform(image=image_np)['image'].unsqueeze(0).to(device)
                emb = model(tensor)

                if emb.ndim != 2 or emb.shape[0] != 1:
                    raise ValueError(f"Expected embedding shape (1, D), got {tuple(emb.shape)}.")

                emb_np = emb[0].cpu().numpy().astype(np.float32)

                if not np.all(np.isfinite(emb_np)):
                    raise ValueError(f"Non-finite embedding generated for: {path}")

                embeddings.append(emb_np)

    finally:
        for module, training_mode in original_modes.items():
            module.training = training_mode

    return np.stack(embeddings)


def add_class_pure(class_name, image_paths, dataset_tag, modality, model, proto_collection, support_collection, replay_buffer, save_dir, device):
    if not image_paths:
        raise ValueError("image_paths cannot be empty.")

    class_save_dir = os.path.join(save_dir, class_name)
    os.makedirs(class_save_dir, exist_ok=True)

    # Remove previously generated copies so repeated additions do not leave stale files.
    for fname in os.listdir(class_save_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            os.remove(os.path.join(class_save_dir, fname))

    saved_paths = []

    for i, src_path in enumerate(image_paths):
        ext = os.path.splitext(src_path)[1] or '.jpg'
        dst_path = os.path.join(class_save_dir, f"{class_name}_{i:04d}{ext}")
        shutil.copy2(src_path, dst_path)
        saved_paths.append(dst_path)

    embeddings = _embed_images(model, saved_paths, modality, device)
    prototype = embeddings.mean(axis=0).astype(np.float32)

    proto_id = f"proto_{dataset_tag}_{class_name}"
    proto_collection.upsert(
        embeddings=[prototype.tolist()],
        documents=[class_name],
        ids=[proto_id],
        metadatas=[{
            'class': class_name,
            'dataset': dataset_tag,
            'n_support': len(embeddings)
        }]
    )

    # Remove old support records for this class before inserting the current set.
    existing = support_collection.get(
        where={'$and': [{'dataset': dataset_tag}, {'class': class_name}]},
        include=['metadatas']
    )

    if existing['ids']:
        support_collection.delete(ids=existing['ids'])

    for i, (emb, path) in enumerate(zip(embeddings, saved_paths)):
        support_collection.upsert(
            embeddings=[emb.tolist()],
            documents=[class_name],
            ids=[f"support_{dataset_tag}_{class_name}_{i}"],
            metadatas=[{
                'class': class_name,
                'image_path': os.path.abspath(path),
                'dataset': dataset_tag,
                'index': i,
                'filename': os.path.basename(path)
            }]
        )

    images_tensor = torch.stack([
        get_transform(modality, 'test')(image=np.array(Image.open(p).convert('RGB')))['image'] for p in saved_paths
    ])

    replay_buffer.add_class(class_name, images_tensor, model, device, label=None)

    print(f"added {class_name} ({len(embeddings)} images) - pure prototype addition")

    return {
        'class': class_name,
        'finetuned': False,
        'n_support': len(embeddings),
        'saved_to': class_save_dir
    }

def _measure_backward_transfer(model, proto_collection, support_collection, device, n_samples=20, seed=42):
    from src.retrieval.retriever import retrieve_similar

    original_modes = {module: module.training for module in model.modules()}
    precisions = []

    all_supports = support_collection.get(include=['embeddings', 'metadatas'])

    if not all_supports['embeddings']:
        return 0.0

    n_total = len(all_supports['embeddings'])
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=min(n_samples, n_total), replace=False)

    try:
        model.eval()

        with torch.inference_mode():
            for idx in indices:
                emb_np = np.asarray(all_supports['embeddings'][idx], dtype=np.float32)
                metadata = all_supports['metadatas'][idx]
                true_class = metadata['class']
                image_path = metadata.get('image_path')

                results = retrieve_similar(emb_np, support_collection, n_results=5, exclude_image_path=image_path)

                if not results:
                    continue

                n_correct = sum(1 for r in results if r['class'] == true_class)
                precisions.append(n_correct / len(results))

    finally:
        for module, training_mode in original_modes.items():
            module.training = training_mode

    return float(np.mean(precisions)) if precisions else 0.0
