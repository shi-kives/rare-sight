import io
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from src.models.embedding_generator import EmbeddingGenerator

logger = logging.getLogger("uvicorn")

# ImageNet normalization stats — standard for a pretrained backbone
# (resnet / efficientnet / vit etc. all typically expect these)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_preprocess_pipeline = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Convert raw uploaded image bytes into a model-ready tensor.

    Returns a (1, 3, 224, 224) float tensor, batch dimension included.
    """
    if not image_bytes:
        raise ValueError("Image data is empty.")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise ValueError("Uploaded file is not a valid image.") from e

    tensor = _preprocess_pipeline(image)
    return tensor.unsqueeze(0)  # add batch dimension -> (1, 3, 224, 224)


def embed_image(image_tensor: torch.Tensor, model: torch.nn.Module, device: torch.device) -> np.ndarray:
    """
    Run a preprocessed image tensor through the model and return a
    128d embedding as a numpy array (detached from the graph, off-GPU).
    """
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    if image_tensor.ndim != 4 or image_tensor.shape[0] != 1:
        raise ValueError("Expected one image with shape (1, 3, H, W).")

    model.eval()
    with torch.inference_mode():
        embedding = model(image_tensor.to(device))
        # Some models return (embedding, logits) or a dict — unwrap if so.
        if isinstance(embedding, (tuple, list)):
            embedding = embedding[0]
        elif isinstance(embedding, dict):
            if "embedding" not in embedding:
                raise ValueError("Model output dictionary must contain an 'embedding' key.")
            embedding = embedding["embedding"]

    if not isinstance(embedding, torch.Tensor) or embedding.ndim != 2 or embedding.shape[0] != 1:
        raise ValueError("Model must return an embedding tensor with shape (1, embed_dim).")

    emb_np = embedding[0].detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(emb_np).all():
        raise ValueError("Model produced a non-finite embedding.")
    return emb_np


def load_model(path: str, config: dict, device: torch.device) -> torch.nn.Module:
    """
    Instantiate the embedding model from config and load trained weights
    from `path`. Returns the model in eval mode, moved to `device`.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    model = EmbeddingGenerator(
        backbone=config.get("backbone", "resnet18"),
        embed_dim=config.get("embed_dim", config.get("embedding_dim", 128)),
    )

    state_dict = torch.load(checkpoint_path, map_location=device)
    # Some checkpoints save {"model_state_dict": ...} rather than the raw state dict
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint must contain a model state dictionary.")

    # Handle checkpoints saved from DataParallel.
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info(f"Loaded {config.get('backbone', 'resnet18')} weights from {checkpoint_path} onto {device}")
    return model
