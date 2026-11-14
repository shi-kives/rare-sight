import numpy as np
import torch
import torch.nn as nn


def enable_dropout(model: nn.Module):
    """
    Enable dropout layers while leaving BatchNorm and other
    modules in evaluation mode.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

def mc_predict(model: nn.Module, image_tensor: torch.Tensor, n_runs: int = 30, device: torch.device = None):
    if n_runs < 2:
        raise ValueError("n_runs must be at least 2.")

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    image_tensor = torch.as_tensor(image_tensor)

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    if image_tensor.ndim != 4:
        raise ValueError("Expected image tensor with shape (C,H,W) or (N,C,H,W).")

    if image_tensor.shape[0] != 1:
        raise ValueError("mc_predict currently accepts one image at a time.")

    image_tensor = image_tensor.to(device)

    original_modes = {module: module.training for module in model.modules()}
    run_embeddings = []

    try:
        model.eval()
        enable_dropout(model)

        with torch.inference_mode():
            for _ in range(n_runs):
                embedding = model(image_tensor)

                if embedding.ndim != 2 or embedding.shape[0] != 1:
                    raise ValueError("Model must return an embedding of shape (1, embed_dim).")

                run_embeddings.append(embedding[0].detach().cpu())

    finally:
        for module, training_mode in original_modes.items():
            module.training = training_mode

    embeddings = torch.stack(run_embeddings, dim=0)

    mean_embedding = embeddings.mean(dim=0)
    variance_per_dim = embeddings.var(dim=0, unbiased=False)
    std_per_dim = embeddings.std(dim=0, unbiased=False)

    scalar_variance = float(variance_per_dim.mean().item())

    return {
        "mean_embedding": mean_embedding.numpy(),
        "variance": scalar_variance,
        "std_per_dim": std_per_dim.numpy(),
        "epistemic_uncertain": None
    }


def calibrate_mc_threshold(model: nn.Module, val_images, n_runs: int = 30, percentile: float = 95.0, device: torch.device = None):
    if not 0 < percentile < 100:
        raise ValueError("percentile must be between 0 and 100.")

    variances = []

    for item in val_images:
        image_tensor = item[0] if isinstance(item, (tuple, list)) else item

        result = mc_predict(model, image_tensor, n_runs=n_runs, device=device)

        variances.append(result["variance"])

    if not variances:
        raise ValueError("No validation images were provided.")

    threshold = float(np.percentile(variances, percentile))

    print(f"Monte Carlo dropout threshold ({percentile}th percentile): {threshold:.6f}")
    return threshold