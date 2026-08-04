import time
import os
import copy
import csv
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
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
    best_state_dict = None

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
            best_state_dict = copy.deepcopy(model.state_dict())

        if verbose:
            print(
                f"Epoch [{epoch:02d}/{num_epochs:02d}] | LR: {current_lr:.1e} | "
                f"Train Loss (w): {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | "
                f"Val Loss (w): {val_loss:.4f} Acc: {val_acc*100:.2f}% "
                f"F1: {val_metrics['f1']:.4f} AUC: {val_metrics['roc_auc']:.4f} Thresh: {val_metrics['threshold']:.3f}"
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    elapsed = time.time() - start_time
    if verbose:
        print(
            f"Training finished in {elapsed:.2f} s. "
            f"Best Val F1: {best_f1:.4f} | Best AUC: {best_metrics.get('roc_auc', 0.0):.4f} | "
            f"Optimal Threshold: {best_metrics.get('threshold', 0.5):.3f}"
        )

    return history, best_metrics


def auto_commit_and_get_hash(commit_msg: Optional[str] = None) -> str:
    """
    Checks if the workspace has uncommitted git changes. If so, automatically stages
    and commits them with a timestamped message. Returns the current short git hash.
    """
    try:
        is_git = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True
        )
        if is_git.returncode != 0:
            print("Git repository not detected. Returning default hash.")
            return "no-git-repo"

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True
        )
        if status.stdout.strip():
            msg = commit_msg or f"Auto-commit before training ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"
            subprocess.run(["git", "add", "-A"], check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", msg], check=True, capture_output=True)
            print(f"Git auto-commit created: {msg}")

        git_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True
        ).stdout.strip()
        print(f"Git commit hash: {git_hash}")
        return git_hash
    except Exception as e:
        print(f"Git auto-commit note: {e}")
        try:
            res = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True
            )
            h = res.stdout.strip()
            return h if h else "git-error"
        except Exception:
            return "git-error"


def save_model_checkpoint(
    model: nn.Module,
    config: Dict[str, Any],
    best_metrics: Dict[str, float],
    git_hash: str,
    save_dir: str = "saved_models"
) -> str:
    """
    Saves the trained PyTorch model state_dict along with configuration metadata and metrics.
    Returns the path to the saved checkpoint file.
    """
    os.makedirs(save_dir, exist_ok=True)
    model_type = config.get("model_type", "model")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_type}_{timestamp}_{git_hash}.pt"
    filepath = os.path.join(save_dir, filename)

    checkpoint = {
        "model_type": model_type,
        "model_state_dict": model.state_dict(),
        "config": config,
        "best_metrics": best_metrics,
        "git_hash": git_hash,
        "timestamp": datetime.now().isoformat(),
    }
    torch.save(checkpoint, filepath)
    print(f"Saved trained model checkpoint to: {filepath}")
    return filepath


def log_experiment(
    config: Dict[str, Any],
    model_name: str,
    best_metrics: Dict[str, float],
    git_hash: str,
    model_path: str,
    csv_path: str = "model_leaderboard.csv"
) -> pd.DataFrame:
    """
    Appends experiment metrics and hyperparameters to a CSV leaderboard file.
    """
    fieldnames = [
        "timestamp",
        "git_hash",
        "model_type",
        "model_name",
        "roc_auc",
        "f1",
        "accuracy",
        "val_loss",
        "threshold",
        "num_epochs",
        "batch_size",
        "lr",
        "feature_mode",
        "max_events",
        "model_path",
    ]

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_hash": git_hash,
        "model_type": config.get("model_type", "unknown"),
        "model_name": model_name,
        "roc_auc": round(float(best_metrics.get("roc_auc", 0.0)), 4),
        "f1": round(float(best_metrics.get("f1", 0.0)), 4),
        "accuracy": round(float(best_metrics.get("accuracy", 0.0)), 4),
        "val_loss": round(float(best_metrics.get("val_loss", 0.0)), 4),
        "threshold": round(float(best_metrics.get("threshold", 0.5)), 3),
        "num_epochs": config.get("num_epochs", 0),
        "batch_size": config.get("batch_size", 0),
        "lr": config.get("lr", 0.0),
        "feature_mode": config.get("feature_mode", "dual_ph"),
        "max_events": config.get("max_events", "all"),
        "model_path": model_path,
    }

    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Logged experiment metrics to leaderboard: {csv_path}")
    return pd.read_csv(csv_path)


def display_leaderboard(
    csv_path: str = "model_leaderboard.csv",
    rank_by: str = "roc_auc",
    ascending: bool = False
) -> pd.DataFrame:
    """
    Reads the leaderboard CSV file, sorts models by the chosen metric,
    and displays/returns the ranked table.
    """
    if not os.path.exists(csv_path):
        print(f"No leaderboard CSV found at {csv_path} yet.")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Leaderboard is empty.")
        return df

    if rank_by not in df.columns:
        print(f"Warning: Metric '{rank_by}' not in leaderboard columns ({list(df.columns)}). Defaulting to 'roc_auc'.")
        rank_by = "roc_auc"

    df_sorted = df.sort_values(by=rank_by, ascending=ascending).reset_index(drop=True)
    df_sorted.index = df_sorted.index + 1
    df_sorted.index.name = "Rank"

    print(f"\n=================== MODEL LEADERBOARD (Ranked by {rank_by.upper()}) ===================")

    try:
        from IPython.display import display
        display(df_sorted)
    except ImportError:
        print(df_sorted.to_string())

    return df_sorted



