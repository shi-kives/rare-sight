import numpy as np
import scipy.stats as stats
import torch
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from tqdm import tqdm
from src.models.episodic_sampler import sample_episode
from src.models.proto_net import prototypical_loss, compute_prototypes
from data.mapping import CLASS_MAPPING_NAMES

from sklearn.metrics import (
    f1_score,
    recall_score,
    confusion_matrix
)

def evaluate(model, dataset, class_index, available_classes, n_way, k_shot, q_query, n_episodes, device):
    was_training = model.training

    if len(available_classes) < n_way:
        raise Exception(f"number of classes available ({available_classes}) is less than n_way ({n_way})!")
    if not available_classes:
        raise ValueError("available_classes cannot be empty.")

    try:
        model.eval()

        episode_accuracies = []
        all_true_labels = []
        all_pred_labels = []

        class_labels = sorted(available_classes)

        with torch.no_grad():
            for _ in tqdm(range(n_episodes), desc=f"{n_way}-way {k_shot}-shot"):
                support, query, labels, episode_classes = sample_episode(dataset, class_index, available_classes, n_way, k_shot, q_query, device, return_classes=True)
                _, accuracy, prototypes = prototypical_loss(model, support, query, labels, n_way, k_shot)

                support_flat = support.reshape(n_way * k_shot, *support.shape[2:])

                support_embeddings = model(support_flat)
                prototypes = support_embeddings.reshape(n_way, k_shot, -1).mean(dim=1)

                query_embeddings = model(query)
                distances = torch.cdist(query_embeddings, prototypes, p=2)

                local_predictions = distances.argmin(dim=1)

                # Map local labels back to original PathMNIST class IDs.
                local_true = labels.cpu().numpy()
                local_pred = local_predictions.cpu().numpy()

                global_true = np.asarray(episode_classes)[local_true]
                global_pred = np.asarray(episode_classes)[local_pred]

                episode_accuracies.append(accuracy)
                all_true_labels.extend(global_true.tolist())
                all_pred_labels.extend(global_pred.tolist())


    finally:
        model.train(was_training)
    
    mean_accuracy = np.mean(episode_accuracies)

    sem = stats.sem(episode_accuracies)
    ci_low, ci_high = stats.t.interval(confidence=0.95, df=len(episode_accuracies) - 1, loc=mean_accuracy, scale=sem)

    ci_half_width = float((ci_high - ci_low) / 2)

    macro_f1 = f1_score(all_true_labels, all_pred_labels, labels=class_labels, average="macro", zero_division=0)

    recalls = recall_score(all_true_labels, all_pred_labels, labels=class_labels, average=None,zero_division=0)

    precisions = precision_score(all_true_labels, all_pred_labels, labels=class_labels, average=None,zero_division=0)


    matrix = confusion_matrix(all_true_labels, all_pred_labels, labels=class_labels)

    metrics = {
        "macro_f1": float(macro_f1),
        "per_class_recall": {int(class_id): float(recall) for class_id, recall in zip(class_labels, recalls)},
        "per_class_precision": {int(class_id): float(precision) for class_id, precision in zip(class_labels, precisions)},
        "confusion_matrix": matrix,
        "class_labels": [int(c) for c in class_labels]
    }

    return mean_accuracy, ci_half_width, metrics


def k_retrieval_precision(model, test_images, test_labels, support_collection, k, device):

    was_training = model.training
    model.eval()

    precisions = []

    with torch.no_grad():
        for image_tensor, true_label in zip(test_images, test_labels):
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(device)
            embedding = model(image_tensor)
            np_embedding = embedding.squeeze().cpu().numpy().tolist()

            results = support_collection.query(query_embeddings = [np_embedding], n_results = k, include = ['metadatas'])
            retrieved_classes = [m['class'] for m in CLASS_MAPPING_NAMES[results['metadatas'][0]]]

            assert isinstance(retrieved_classes[0], int), "Retrieved classes are not integers labels."
        
            correct = sum(1 for c in retrieved_classes if c == int(true_label))

            retrieved_count = len(retrieved_classes)
            if retrieved_count == 0:
                continue

            precisions.append(correct / retrieved_count)

    model.train(was_training)
    return float(np.mean(precisions))
    
def distance_threshold(model, dataset, class_index, available_classes, n_way, k_shot, q_query, proto_collection, n_episodes = 200, percentile = 95, device = None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    was_training = model.training
    model.eval()
    min_distances = []

    with torch.no_grad():
        for _ in range(n_episodes):
            support, query, labels = sample_episode(dataset, class_index, available_classes, n_way = n_way, k_shot = k_shot, q_query = q_query, device = device)
            C, H, W = support.shape[2:]

            support_embeddings = model(support.view(n_way * k_shot, C, H, W))
            prototypes = compute_prototypes(support_embeddings, n_way, k_shot)
            query_embeddings = model(query)

            dists = torch.cdist(query_embeddings, prototypes, p = 2)
            min_d = dists.min(dim = 1).values

            predictions = dists.argmin(dim = 1)
            correct = predictions == labels
            min_distances.extend(min_d[correct].cpu().numpy().tolist())
    
    if len(min_distances) == 0:
        raise ValueError("No correctly classified query distances available for threshold estimation.")

    if not 0 <= percentile <= 100:
        raise ValueError("Percentile must be between 0 and 100.")

    threshold = float(np.percentile(min_distances, percentile))       
    print(f"distance threshold ({percentile}%): {threshold:.4f}")
    model.train(was_training)
    return threshold