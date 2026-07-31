import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.torchsparse import SparseTensor


def compute_metrics(
    targets: np.ndarray,
    probs: np.ndarray,
    preds: Optional[np.ndarray] = None
) -> Dict[str, float]:
    if preds is None:
        preds = (probs >= 0.5).astype(int)

    acc = float(accuracy_score(targets, preds))
    f1 = float(f1_score(targets, preds, zero_division=0))
    prec = float(precision_score(targets, preds, zero_division=0))
    rec = float(recall_score(targets, preds, zero_division=0))

    try:
        auc = float(roc_auc_score(targets, probs))
    except ValueError:
        auc = 0.5

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "roc_auc": auc
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for coords, feats, labels in loader:
        coords = coords.to(device)
        feats = feats.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        input_tensor = SparseTensor(feats=feats, coords=coords)

        optimizer.zero_grad()
        logits = model(input_tensor)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    return avg_loss, acc


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for coords, feats, labels in loader:
            coords = coords.to(device)
            feats = feats.to(device)
            labels = labels.to(device)
            batch_size = labels.size(0)

            input_tensor = SparseTensor(feats=feats, coords=coords)

            logits = model(input_tensor)
            loss = criterion(logits, labels)

            total_loss += loss.item() * batch_size
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    targets_np = np.array(all_targets)
    probs_np = np.array(all_probs)

    metrics = compute_metrics(targets_np, probs_np)
    return avg_loss, metrics["accuracy"], metrics, probs_np, targets_np


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    step_size: int = 3,
    gamma: float = 0.1,
    class_weights: Optional[torch.Tensor] = None,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True
) -> Tuple[Dict[str, list], Dict[str, float]]:

    model = model.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "val_f1": [], "val_auc": [],
        "lr": []
    }

    start_time = time.time()
    best_f1 = -1.0
    best_metrics = {}

    if verbose:
        print(f"Starting TorchSparse model training on {device} ({num_epochs} epochs)...")

    for epoch in range(1, num_epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_metrics, _, _ = validate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_metrics["f1"])
        history["val_auc"].append(val_metrics["roc_auc"])
        history["lr"].append(current_lr)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_metrics = val_metrics.copy()
            best_metrics["val_loss"] = val_loss

        if verbose:
            print(
                f"Epoch [{epoch:02d}/{num_epochs:02d}] | LR: {current_lr:.1e} | "
                f"Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% "
                f"F1: {val_metrics['f1']:.4f} AUC: {val_metrics['roc_auc']:.4f}"
            )

    elapsed = time.time() - start_time
    if verbose:
        print(f"Training finished in {elapsed:.2f} s. Best Val F1: {best_f1:.4f} | Best AUC: {best_metrics.get('roc_auc', 0.0):.4f}")

    return history, best_metrics
