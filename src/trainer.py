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

try:
    from torch_geometric.data import HeteroData
except ImportError:
    HeteroData = None


def find_optimal_threshold(
    targets: np.ndarray,
    probs: np.ndarray,
    metric: str = "f1",
    num_thresholds: int = 200,
) -> Tuple[float, float]:
    """
    Scans candidate decision thresholds to find the threshold that maximizes the specified metric.
    Supported metrics: 'f1', 'accuracy', 'precision', 'recall', 'youden' (TPR - FPR).
    """
    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    best_thresh = 0.5
    best_score = -1.0

    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        if metric == "f1":
            score = float(f1_score(targets, preds, zero_division=0))
        elif metric == "accuracy":
            score = float(accuracy_score(targets, preds))
        elif metric == "precision":
            score = float(precision_score(targets, preds, zero_division=0))
        elif metric == "recall":
            score = float(recall_score(targets, preds, zero_division=0))
        elif metric == "youden":
            rec = recall_score(targets, preds, zero_division=0)
            tn = np.sum((targets == 0) & (preds == 0))
            fp = np.sum((targets == 0) & (preds == 1))
            fpr = fp / max(1, (fp + tn))
            score = float(rec - fpr)
        else:
            score = float(f1_score(targets, preds, zero_division=0))

        if score > best_score:
            best_score = score
            best_thresh = float(thresh)

    return best_thresh, best_score


def compute_metrics(
    targets: np.ndarray,
    probs: np.ndarray,
    preds: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    optimize_threshold: bool = True,
    optimize_metric: str = "f1",
) -> Dict[str, float]:
    if optimize_threshold and preds is None:
        effective_thresh, _ = find_optimal_threshold(targets, probs, metric=optimize_metric)
    elif threshold is not None:
        effective_thresh = float(threshold)
    else:
        effective_thresh = 0.5

    if preds is None:
        preds = (probs >= effective_thresh).astype(int)

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
        "roc_auc": auc,
        "threshold": effective_thresh,
    }



def compute_sample_weights(
    labels: torch.Tensor,
    energies: Optional[torch.Tensor] = None,
    class_weights: Optional[torch.Tensor] = None,
    use_energy_weighting: bool = False,
    energy_epsilon: float = 0.5,
    energy_alpha: float = 1.0,
) -> torch.Tensor:
    """
    Computes per-sample weights combining per-class inverse frequency weights
    and optional inverse true neutrino energy reweighting.

    Energy weighting formula:
      w_energy = (1.0 / (E_nu + epsilon)) ** alpha
      w_energy_norm = w_energy / mean(w_energy)   [batch normalized]
    Total sample weight:
      w_i = w_class[label_i] * w_energy_norm[i]
    """
    batch_size = labels.size(0)
    device = labels.device

    if class_weights is not None:
        w_class = class_weights[labels]
    else:
        w_class = torch.ones(batch_size, dtype=torch.float32, device=device)

    if use_energy_weighting and energies is not None:
        safe_energies = torch.clamp(energies.float(), min=0.0)
        raw_energy_w = (1.0 / (safe_energies + float(energy_epsilon))) ** float(energy_alpha)
        mean_w = torch.mean(raw_energy_w)
        if mean_w > 0:
            w_energy = raw_energy_w / mean_w
        else:
            w_energy = torch.ones_like(raw_energy_w)
    else:
        w_energy = torch.ones(batch_size, dtype=torch.float32, device=device)

    return w_class * w_energy


def compute_batch_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    energies: Optional[torch.Tensor] = None,
    class_weights: Optional[torch.Tensor] = None,
    use_energy_weighting: bool = False,
    energy_epsilon: float = 0.5,
    energy_alpha: float = 1.0,
) -> torch.Tensor:
    per_sample_loss = nn.functional.cross_entropy(logits, labels, reduction="none")
    sample_weights = compute_sample_weights(
        labels=labels,
        energies=energies,
        class_weights=class_weights,
        use_energy_weighting=use_energy_weighting,
        energy_epsilon=energy_epsilon,
        energy_alpha=energy_alpha,
    )
    return (per_sample_loss * sample_weights).mean()


def _forward_batch(model: nn.Module, batch, device: torch.device):
    model = model.to(device)
    energies = None
    non_blocking = (device.type == "cuda")

    if isinstance(batch, tuple):
        if len(batch) == 4:
            coords, feats, labels, energies = batch
        elif len(batch) == 3:
            coords, feats, labels = batch
        else:
            raise TypeError("Unsupported tuple batch length.")

        coords = coords.to(device, non_blocking=non_blocking)
        feats = feats.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        if energies is not None:
            energies = energies.to(device, non_blocking=non_blocking)
        input_tensor = SparseTensor(feats=feats, coords=coords)
        logits = model(input_tensor)
        return logits, labels, energies

    if HeteroData is not None and isinstance(batch, HeteroData):
        batch = batch.to(device, non_blocking=non_blocking)
        labels = batch.y
        energies = getattr(batch, "true_energy", None)
        logits = model(batch)
        return logits, labels, energies

    if hasattr(batch, "to") and hasattr(batch, "y"):
        batch = batch.to(device, non_blocking=non_blocking)
        labels = batch.y
        energies = getattr(batch, "true_energy", None)
        logits = model(batch)
        return logits, labels, energies

    raise TypeError(
        "Unsupported batch type. Expected SparseTensor collate output or a PyG HeteroData batch."
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Optional[nn.Module] = None,
    device: torch.device = torch.device("cpu"),
    class_weights: Optional[torch.Tensor] = None,
    use_energy_weighting: bool = False,
    energy_epsilon: float = 0.5,
    energy_alpha: float = 1.0,
) -> Tuple[float, float]:
    model = model.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in loader:
        logits, labels, energies = _forward_batch(model, batch, device)
        batch_size = labels.size(0)

        optimizer.zero_grad()
        if use_energy_weighting or class_weights is not None or criterion is None:
            loss = compute_batch_loss(
                logits, labels, energies,
                class_weights=class_weights,
                use_energy_weighting=use_energy_weighting,
                energy_epsilon=energy_epsilon,
                energy_alpha=energy_alpha,
            )
        else:
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
    criterion: Optional[nn.Module] = None,
    device: torch.device = torch.device("cpu"),
    class_weights: Optional[torch.Tensor] = None,
    use_energy_weighting: bool = False,
    energy_epsilon: float = 0.5,
    energy_alpha: float = 1.0,
    optimize_threshold: bool = True,
    threshold_metric: str = "f1",
) -> Tuple[float, float, Dict[str, float], np.ndarray, np.ndarray]:
    model = model.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            logits, labels, energies = _forward_batch(model, batch, device)
            batch_size = labels.size(0)

            if use_energy_weighting or class_weights is not None or criterion is None:
                loss = compute_batch_loss(
                    logits, labels, energies,
                    class_weights=class_weights,
                    use_energy_weighting=use_energy_weighting,
                    energy_epsilon=energy_epsilon,
                    energy_alpha=energy_alpha,
                )
            else:
                loss = criterion(logits, labels)

            total_loss += loss.item() * batch_size
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    targets_np = np.array(all_targets)
    probs_np = np.array(all_probs)

    metrics = compute_metrics(
        targets_np,
        probs_np,
        optimize_threshold=optimize_threshold,
        optimize_metric=threshold_metric,
    )
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
    use_energy_weighting: bool = False,
    energy_epsilon: float = 0.5,
    energy_alpha: float = 1.0,
    optimize_threshold: bool = True,
    threshold_metric: str = "f1",
    device: torch.device = torch.device("cpu"),
    verbose: bool = True
) -> Tuple[Dict[str, list], Dict[str, float]]:

    model = model.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "val_f1": [], "val_auc": [],
        "val_threshold": [],
        "lr": []
    }

    start_time = time.time()
    best_f1 = -1.0
    best_metrics = {}

    if verbose:
        print(f"Starting model training on {device} ({num_epochs} epochs)...")
        if use_energy_weighting:
            print(f"Inverse Energy Reweighting Enabled: w(E_nu) = 1/(E_nu + {energy_epsilon})^{energy_alpha} (batch normalized)")
        if optimize_threshold:
            print(f"Decision Threshold Optimization Enabled (Maximizing '{threshold_metric}')")

    for epoch in range(1, num_epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer,
            device=device, class_weights=class_weights,
            use_energy_weighting=use_energy_weighting,
            energy_epsilon=energy_epsilon, energy_alpha=energy_alpha,
        )
        val_loss, val_acc, val_metrics, _, _ = validate_epoch(
            model, val_loader,
            device=device, class_weights=class_weights,
            use_energy_weighting=use_energy_weighting,
            energy_epsilon=energy_epsilon, energy_alpha=energy_alpha,
            optimize_threshold=optimize_threshold, threshold_metric=threshold_metric,
        )
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_metrics["f1"])
        history["val_auc"].append(val_metrics["roc_auc"])
        history["val_threshold"].append(val_metrics["threshold"])
        history["lr"].append(current_lr)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_metrics = val_metrics.copy()
            best_metrics["val_loss"] = val_loss

        if verbose:
            print(
                f"Epoch [{epoch:02d}/{num_epochs:02d}] | LR: {current_lr:.1e} | "
                f"Train Loss (w): {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | "
                f"Val Loss (w): {val_loss:.4f} Acc: {val_acc*100:.2f}% "
                f"F1: {val_metrics['f1']:.4f} AUC: {val_metrics['roc_auc']:.4f} Thresh: {val_metrics['threshold']:.3f}"
            )

    elapsed = time.time() - start_time
    if verbose:
        print(
            f"Training finished in {elapsed:.2f} s. "
            f"Best Val F1: {best_f1:.4f} | Best AUC: {best_metrics.get('roc_auc', 0.0):.4f} | "
            f"Optimal Threshold: {best_metrics.get('threshold', 0.5):.3f}"
        )

    return history, best_metrics


