from medmnist import PathMNIST
import os
from pathlib import Path
import numpy as np
import json
import pandas as pd

SEED = 42
SAMPLES_PER_CLASS = 10

current_dir = Path(__file__).resolve().parents
ROOT = current_dir[1]

DATA_DIR = ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
SUPPORT_DIR = DATA_DIR / "support"
SPLIT_PATH = DATA_DIR / "split.json"
MANIFEST_PATH = DATA_DIR / "manifests" / "support_manifest.json"

def load_datasets():
    print("starting downloads")

    train_dataset = PathMNIST(split='train', root= RAW_DIR, download=True, size=224)
    print("train dataset loaded")

    val_dataset = PathMNIST(split='val', root= RAW_DIR, download=True, size=224)
    print("val dataset loaded")

    test_dataset = PathMNIST(split='test', root= RAW_DIR, download=True, size=224)
    print("test dataset loaded")

    return train_dataset, val_dataset, test_dataset

def create_support(train):
    
    with open(SPLIT_PATH, 'r') as file:
        data = json.load(file)

    base_classes = data['base_classes']
    novel_classes = data['novel_classes']
    support_classes = base_classes + novel_classes

    add_classes = data["add_classes"]

    assert set(base_classes).isdisjoint(novel_classes)
    assert set(base_classes).isdisjoint(add_classes)
    assert set(novel_classes).isdisjoint(add_classes)

    all_classes = base_classes + novel_classes + add_classes

    assert sorted(all_classes) == list(range(9))

    rng = np.random.default_rng(SEED)
    labels = np.array(train.labels).flatten()

    SUPPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": "PathMNIST",
        "source_split": "train",
        "seed": SEED,
        "samples_per_class": SAMPLES_PER_CLASS,
        "classes": {}
    }

    for cls in support_classes:
        class_indices = np.where(labels == cls)[0]

        if len(class_indices) < SAMPLES_PER_CLASS:
            raise ValueError(f"Not enough training samples for class {cls}!")
        
        chosen_indices = rng.choice(class_indices, size=SAMPLES_PER_CLASS, replace=False)

        class_dir = SUPPORT_DIR / f"class_{cls}"
        os.makedirs(class_dir, exist_ok=True)

        manifest["classes"][str(cls)] = []

        for idx in chosen_indices:
            pil_img, _ = train[int(idx)]
            
            file_name = f"img_{idx}.png"
            save_path = os.path.join(class_dir, file_name)
            
            pil_img.save(save_path)

            manifest["classes"][str(cls)].append({
                "file_name": file_name,
                "source_split": "train",
                "source_index": int(idx)
            })

        with open(MANIFEST_PATH, "w") as file:
            json.dump(manifest, file, indent=4)
      
        print(f"saved 10 images for class {cls} to {class_dir}")
        print(f"manifest saved at: {MANIFEST_PATH}")

def inspect(dataset, type):
    print(f"dataset {type} details: ")
    print(dataset)

    print(f"shape of {type}: ", dataset.imgs.shape)

    series = pd.Series(dataset.labels.flatten())
    series = series.sort_values()
    print("all the classes: ", series.unique())
    print("number of unique classes: ", series.nunique())
    print("class distribution:\n", series.value_counts().sort_index())

if __name__ == '__main__':
    train, val, test = load_datasets()

    inspect(train, "train")
    inspect(val, "val")
    inspect(test, "test")

    create_support(train)