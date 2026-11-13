from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np
import os
from medmnist import PathMNIST
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "raw"

class PathMNISTDataset(Dataset):

    def __init__(self, split, transform = None):
        os.makedirs(DATA_ROOT, exist_ok=True)
        file_exists = os.path.exists(os.path.join(DATA_ROOT, 'pathmnist_224.npz'))

        assert file_exists == True, "PathMNIST NPZ file doesn't exist."

        self.base = PathMNIST(split = split, download = False, size = 224, root = DATA_ROOT, mmap_mode='r')

        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, label = self.base[idx]
        np_image = np.array(image) # shape = (224, 224, 3)

        if self.transform: 
            np_image = self.transform(image = np_image)['image'] # torch tensor, shape (3, 224, 224)
        else:
            raise Exception("Transform function unknown!")

        label_int = int(label[0]) if hasattr(label, '__len__') else int(label) # convert [3] (array obj) -> 3 (plain int)

        return np_image, label_int

def build_class_index(dataset): # creates a mapping of class -> sample indices for episodic training
    class_index = defaultdict(list)

    if hasattr(dataset, 'base') and hasattr(dataset.base, 'labels'):
        labels = dataset.base.labels.flatten()
        for idx, label in enumerate(labels):
            class_index[int(label)].append(idx)
        return dict(class_index)

    # Fallback for any other dataset wrapper
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_index[int(label)].append(idx)

    return dict(class_index)