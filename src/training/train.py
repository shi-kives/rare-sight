import json
import torch
import torch.nn as nn
import mlflow
from tqdm import tqdm
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from src.data.datasets import PathMNISTDataset, build_class_index
from src.data.preprocessing import get_transform
from src.models.episodic_sampler import EpisodeLoader
from src.models.embedding_generator import EmbeddingGenerator
from src.models.proto_net import prototypical_loss
from src.evaluation.evaluate import evaluate


SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

with open('data/processed/splits/medmnist_split.json', 'r', encoding = 'utf-8') as file:
    data = json.load(file)
    print("data loaded: ", data)

CONFIG = {
    'backbone': 'resnet18',
    'embed_dim': 128,
    'n_way': 5,
    'k_shot': 5,
    'q_query': 15,
    'episodes_per_epoch': 100,
    'val_episodes': 50,
    'n_epochs': 30,
    'lr_head': 1e-3,
    'lr_backbone': 1e-5,
    'unfreeze_epoch': 5, # unfreeze backbone after this number of epochs
    'grad_clip': 1.0,
    'base_classes': data['base_classes'],
    'novel_classes': data['novel_classes'],
}

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("training on device: ", device)

    train_dataset = PathMNISTDataset('train', transform = get_transform('path', 'train'))
    val_dataset = PathMNISTDataset('val', transform = get_transform('path', 'test'))

    train_class_idx = build_class_index(train_dataset)
    val_class_idx = build_class_index(val_dataset)

    train_loader = EpisodeLoader(train_dataset, train_class_idx, CONFIG['base_classes'], CONFIG['n_way'], CONFIG['k_shot'], CONFIG['q_query'], CONFIG['episodes_per_epoch'], device)

    model = EmbeddingGenerator(CONFIG['backbone'], CONFIG['embed_dim']).to(device)
    model.freeze_backbone()

    optimizer = torch.optim.Adam(model.projection.parameters(), lr=CONFIG['lr_head'])

    with mlflow.start_run(run_name = f"{CONFIG['backbone']}_baseline"):
        mlflow.log_params(CONFIG)

        best_val_accuracy = 0.0

        for epoch in range(CONFIG['n_epochs']):
            if epoch == CONFIG['unfreeze_epoch']:
                model.unfreeze_last_block()
                optimizer.add_param_group({
                    'params': [p for p in model.backbone.parameters() if p.requires_grad],
                    'lr': CONFIG['lr_backbone']
                })
                print(f"epoch {epoch}: unfroze last block of backbone {CONFIG['backbone']}")

            model.train()

            epoch_losses, epoch_accuracies = [], []

            for support, query, labels in tqdm(train_loader, desc = f"epoch {epoch + 1}"):
                optimizer.zero_grad()

                loss, accuracy, _ = prototypical_loss(model, support, query, labels, CONFIG['n_way'], CONFIG['k_shot'])

                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])

                optimizer.step()

                epoch_losses.append(loss.item())
                epoch_accuracies.append(accuracy)

            train_loss = sum(epoch_losses) / len(epoch_losses)
            train_accuracy = sum(epoch_accuracies) / len(epoch_accuracies)

            val_accuracy, val_conf_int, val_metrics = evaluate(model, val_dataset, val_class_idx, CONFIG['base_classes'], CONFIG['n_way'], CONFIG['k_shot'], CONFIG['q_query'], CONFIG['val_episodes'], device)

            print(f"-- epoch {epoch + 1}/{CONFIG['n_epochs']} --")
            print(f"training loss: {train_loss:.4f}, training accuracy: {train_accuracy:.4f}, validation accuracy: {val_accuracy:.4f} +- {val_conf_int:.4f}")
            print("classification metrics: ")
            print("F1-Score: ", val_metrics['macro_f1'])
            print("Per-Class Recall: ", val_metrics['per_class_recall'])
            print("Per-Class Precision: ", val_metrics['per_class_recall'])
            print("Confusion Matrix: ")
            print(val_metrics['confusion_matrix'])

            cm_png = ConfusionMatrixDisplay(confusion_matrix=val_metrics['confusion_matrix'])
            fig, ax = plt.subplots(figsize=(5, 5))
            cm_png.plot(ax=ax)

            mlflow.log_metric('train_loss', train_loss, step = epoch)
            mlflow.log_metric('train_accuracy', train_accuracy, step = epoch)
            mlflow.log_metric('val_accuracy', val_accuracy, step = epoch)
            mlflow.log_metric('val_ci_half_width', val_conf_int, step = epoch)
            mlflow.log_metric('val_f1', val_metrics['macro_f1'], step = epoch)
            mlflow.log_metrics(val_metrics['per_class_recall'])
            mlflow.log_metrics(val_metrics['per_class_precision'])
            mlflow.log_figure(fig, "val confusion matrix")

            model_path = f"models/best_model_{CONFIG['backbone']}.pt"
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy

                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "config": CONFIG,
                    "epoch": epoch,
                    "best_val_accuracy": best_val_accuracy,
                    "seed": 42
                }
                
                os.makedirs("models", exist_ok=True)
                torch.save(checkpoint, model_path)
                print(f"saved best model with val accuracy: {val_accuracy:.4f}")


def evaluate_test():
    print("test evaluation started")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("testing on device: ", device)

    model = EmbeddingGenerator(CONFIG['backbone'], CONFIG['embed_dim']).to(device)

    with mlflow.start_run(run_name = f"{CONFIG['backbone']}_baseline_test"):
        test_dataset = PathMNISTDataset(split = 'test', transform = get_transform('path', 'test'))
        test_class_idx = build_class_index(dataset = test_dataset)

        model_path = f"models/best_model_{CONFIG['backbone']}.pt"
        model.load_state_dict(torch.load(model_path, map_location=device))

        test_acc_5shot, test_ci_5shot, test_metrics_5shot = evaluate(model = model, dataset = test_dataset, class_index = test_class_idx, available_classes = CONFIG['novel_classes'], n_way = 3, k_shot = 5, q_query = 15, n_episodes = 600, device = device)

        test_acc_1shot, test_ci_1shot, test_metrics_1shot = evaluate(model = model, dataset = test_dataset, class_index = test_class_idx, available_classes = CONFIG['novel_classes'], n_way = 3, k_shot = 1, q_query = 15, n_episodes = 600, device = device)

        print("\n---- final results ---")
        print("1. Accuracy")
        print(f"3-way 5-shot: {test_acc_5shot:.4f} +- {test_ci_5shot:.4f}")
        print(f"3-way 1-shot: {test_acc_1shot:.4f} +- {test_ci_1shot:.4f}")

        print("2. F1 Score")
        print(f"3-way 5-shot: {test_metrics_5shot['macro_f1']}")
        print(f"3-way 1-shot: {test_metrics_1shot['macro_f1']}")

        print("3. Per-Class Recall")
        print(f"3-way 5-shot: {test_metrics_5shot['per_class_recall']}")
        print(f"3-way 1-shot: {test_metrics_1shot['per_class_recall']}")

        print("3. Per-Class Recall")
        print(f"3-way 5-shot: {test_metrics_5shot['per_class_precision']}")
        print(f"3-way 1-shot: {test_metrics_1shot['per_class_precision']}")

        print("3. Per-Class Recall")
        print(f"3-way 5-shot: \n{test_metrics_5shot['confusion_matrix']}")
        print(f"3-way 1-shot: \n{test_metrics_1shot['confusion_matrix']}")

        cm_png = ConfusionMatrixDisplay(confusion_matrix=test_metrics_5shot['confusion_matrix'])
        fig1, ax = plt.subplots(figsize=(5, 5))
        cm_png.plot(ax=ax)

        cm_png = ConfusionMatrixDisplay(confusion_matrix=test_metrics_1shot['confusion_matrix'])
        fig2, ax = plt.subplots(figsize=(5, 5))
        cm_png.plot(ax=ax)
        
        mlflow.log_metric('test_acc_3way_5shot', test_acc_5shot)
        mlflow.log_metric('test_ci_3way_5shot', test_ci_5shot)
        mlflow.log_metric('test_acc_3way_1shot', test_acc_1shot)
        mlflow.log_metric('test_ci_3way_1shot', test_ci_1shot)
        mlflow.log_metric('test_mf1_3way_5shot', test_metrics_5shot['macro_f1'])
        mlflow.log_metric('test_mf1_3way_5shot', test_metrics_1shot['macro_f1'])
        mlflow.log_metrics(test_metrics_5shot['per_class_recall'])
        mlflow.log_metrics(test_metrics_5shot['per_class_precision'])
        mlflow.log_figure(fig1, "test 5 shot confusion matrix")
        mlflow.log_metrics(test_metrics_1shot['per_class_recall'])
        mlflow.log_metrics(test_metrics_1shot['per_class_precision'])
        mlflow.log_figure(fig2, "test 1 shot confusion matrix")


if __name__ == '__main__':
    train()
    evaluate_test()