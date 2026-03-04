"""
evaluator.py
============
Evaluation utilities:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix (saved as PNG)
  - ROC Curve (saved as PNG)
  - Training curve (loss + accuracy per round)

All metrics match those reported in the paper (Section IV-C).
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Benign", "Malicious"]


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model on a DataLoader.

    Returns:
        dict with keys: accuracy, precision, recall, f1, auc
    """
    model.eval().to(device)
    all_preds  = []
    all_labels = []
    all_probs  = []

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1)[:, 1]   # prob of class=1 (Malicious)
        preds  = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_probs = np.array(all_probs)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    # AUC
    try:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
    except Exception:
        roc_auc = 0.0

    return {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "auc":       roc_auc,
        "y_true":    y_true,
        "y_pred":    y_pred,
        "y_probs":   y_probs,
    }


def print_metrics(metrics: dict, title: str = "Evaluation Results") -> None:
    """Print formatted evaluation metrics."""
    print("\n" + "=" * 50)
    print(f" {title}")
    print("=" * 50)
    print(f"  Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {metrics['precision']*100:.2f}%")
    print(f"  Recall   : {metrics['recall']*100:.2f}%")
    print(f"  F1-Score : {metrics['f1']*100:.2f}%")
    print(f"  AUC-ROC  : {metrics['auc']:.4f}")
    print("=" * 50 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Confusion Matrix
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Save a high-quality confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, fmt, subtitle in [
        (axes[0], cm,      "d",     "Counts"),
        (axes[1], cm_norm, ".2%",   "Normalized"),
    ]:
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label",      fontsize=12)
        ax.set_title(f"{title} ({subtitle})", fontsize=13, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# ROC Curve
# ──────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    save_path: str,
    label: str = "ResNet-50",
) -> None:
    """Save a ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2563EB", lw=2,
            label=f"{label} (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.12, color="#2563EB")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Lightweight-Fed-NIDS", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curve saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Training Curve
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    round_logs: List[dict],
    save_path: str,
    title_suffix: str = "",
) -> None:
    """
    Plot training loss and validation accuracy across FL rounds.
    """
    rounds      = [r["round"]      for r in round_logs]
    train_losses = [r["train_loss"] for r in round_logs]
    val_accs    = [r["val_acc"] * 100 for r in round_logs]
    comm_costs  = [r["comm_cost_mb"] for r in round_logs]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(rounds, train_losses, "o-", color="#EF4444", lw=2, markersize=6)
    axes[0].set_xlabel("FL Round", fontsize=12)
    axes[0].set_ylabel("Training Loss", fontsize=12)
    axes[0].set_title(f"Training Loss{title_suffix}", fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(rounds)

    # Val Accuracy
    axes[1].plot(rounds, val_accs, "s-", color="#10B981", lw=2, markersize=6)
    axes[1].set_xlabel("FL Round", fontsize=12)
    axes[1].set_ylabel("Validation Accuracy (%)", fontsize=12)
    axes[1].set_title(f"Validation Accuracy{title_suffix}", fontsize=13, fontweight="bold")
    axes[1].set_ylim([0, 100])
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(rounds)

    # Communication Cost
    axes[2].bar(rounds, comm_costs, color="#6366F1", alpha=0.8)
    axes[2].set_xlabel("FL Round", fontsize=12)
    axes[2].set_ylabel("Cumulative Comm. Cost (MB)", fontsize=12)
    axes[2].set_title(f"Communication Cost{title_suffix}", fontsize=13, fontweight="bold")
    axes[2].grid(True, alpha=0.3, axis="y")
    axes[2].set_xticks(rounds)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Training curves saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Client statistics
# ──────────────────────────────────────────────────────────────────────────────

def log_client_stats(client_loaders: dict) -> None:
    """Print per-client sample counts."""
    print("\nClient Statistics:")
    print(f"  {'Client ID':<12} {'# Samples':>10}")
    print("  " + "-" * 22)
    for cid, loader in client_loaders.items():
        n = len(loader.dataset)
        print(f"  {cid:<12} {n:>10}")
    print()
