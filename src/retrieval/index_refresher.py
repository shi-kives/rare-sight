import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from src.data.preprocessing import get_transform

def refresh_class(class_name, dataset_tag, image_dir, modality, model, proto_collection, support_collection, device):
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Class directory not found: {image_dir}")

    transform = get_transform(modality, 'test')
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    if not image_files:
        raise ValueError(f"No images found in: {image_dir}")

    original_modes = {module: module.training for module in model.modules()}
    new_embeddings = []

    try:
        model.eval()

        for i, fname in enumerate(image_files):
            path = os.path.join(image_dir, fname)
            image = Image.open(path).convert('RGB')
            np_image = np.array(image)
            tensor = transform(image=np_image)['image'].unsqueeze(0).to(device)

            with torch.inference_mode():
                emb = model(tensor)

            if emb.ndim != 2 or emb.shape[0] != 1:
                raise ValueError(f"Expected embedding shape (1, D), got {tuple(emb.shape)}.")

            emb_np = emb[0].cpu().numpy().astype(np.float32)

            if not np.all(np.isfinite(emb_np)):
                raise ValueError(f"Non-finite embedding generated for: {path}")

            new_embeddings.append(emb_np)

            entry_id = f"support_{dataset_tag}_{class_name}_{i}"
            support_collection.upsert(
                ids=[entry_id],
                embeddings=[emb_np.tolist()],
                documents=[class_name],
                metadatas=[{
                    'class': class_name,
                    'image_path': os.path.abspath(path),
                    'dataset': dataset_tag,
                    'index': i,
                    'filename': fname
                }]
            )

        new_prototype = np.mean(np.stack(new_embeddings), axis=0).astype(np.float32)

        if not np.all(np.isfinite(new_prototype)):
            raise ValueError(f"Non-finite prototype generated for: {class_name}")

        proto_id = f"proto_{dataset_tag}_{class_name}"
        proto_collection.upsert(
            ids=[proto_id],
            embeddings=[new_prototype.tolist()],
            documents=[class_name],
            metadatas=[{
                'class': class_name,
                'dataset': dataset_tag,
                'n_support': len(new_embeddings)
            }]
        )

        # Remove support records left over from an older, larger version of this class.
        existing = support_collection.get(
            where={'$and': [{'dataset': dataset_tag}, {'class': class_name}]},
            include=['metadatas']
        )

        expected_ids = {f"support_{dataset_tag}_{class_name}_{i}" for i in range(len(image_files))}
        stale_ids = [
            entry_id for entry_id, metadata in zip(existing['ids'], existing['metadatas'])
            if entry_id not in expected_ids
        ]

        if stale_ids:
            support_collection.delete(ids=stale_ids)

    finally:
        for module, training_mode in original_modes.items():
            module.training = training_mode


def refresh_all(model, support_root_map, proto_collection, support_collection, device):
    for dataset_tag, config in support_root_map.items():
        print("refreshing ", dataset_tag)

        for class_name in tqdm(config['classes']):
            image_dir = os.path.join(config['root'], class_name)

            if not os.path.isdir(image_dir):
                print(f"class directory not found: {image_dir}. skipping this class")
                continue

            refresh_class(class_name=class_name, dataset_tag=dataset_tag, image_dir=image_dir, modality=config['modality'], model=model, proto_collection=proto_collection, support_collection=support_collection, device=device)

    print("refresh complete.")
    print(f"prototypes: {proto_collection.count()}")
    print(f"support images: {support_collection.count()}")