"""
visualization.py
================
Aggregate visualization helpers for FL experiment results.
"""

import os
import logging
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_sparsity_comparison(
    results: Dict[str, dict],
    save_path: str,
) -> None:
    """
    Bar chart comparing Accuracy and F1 across sparsity levels.
    `results` is a dict: { '0%': {accuracy, f1, ...}, '50%': {...}, ... }
    """
    labels    = list(results.keys())
    accuracies = [v["accuracy"] * 100 for v in results.values()]
    f1_scores  = [v["f1"] * 100       for v in results.values()]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy (%)",
                   color="#2563EB", alpha=0.85)
    bars2 = ax.bar(x + width / 2, f1_scores,  width, label="F1-Score (%)",
                   color="#10B981", alpha=0.85)

    ax.set_xlabel("Pruning Sparsity", fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("ResNet-50: Accuracy & F1 vs Pruning Sparsity\n(USTC-TFC2016, Lightweight-Fed-NIDS)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([90, 101])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Sparsity comparison plot saved → {save_path}")


def plot_fl_metrics_summary(
    round_logs: List[dict],
    save_path: str,
    title_suffix: str = "",
) -> None:
    """FL metrics summary: loss, accuracy, comm cost, round time."""
    rounds      = [r["round"]         for r in round_logs]
    losses      = [r["train_loss"]    for r in round_logs]
    accs        = [r["val_acc"] * 100 for r in round_logs]
    comm_costs  = [r["comm_cost_mb"]  for r in round_logs]
    times       = [r["round_time_s"]  for r in round_logs]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Federated Learning Metrics{title_suffix}",
                 fontsize=14, fontweight="bold", y=1.01)

    configs = [
        (axes[0, 0], losses,     "#EF4444", "o-", "Training Loss",           "Loss"),
        (axes[0, 1], accs,       "#10B981", "s-", "Validation Accuracy (%)", "Acc (%)"),
        (axes[1, 0], comm_costs, "#6366F1", "^-", "Comm. Cost (MB)",         "MB"),
        (axes[1, 1], times,      "#F59E0B", "D-", "Round Time (s)",          "seconds"),
    ]
    for ax, data, color, marker, title, ylabel in configs:
        ax.plot(rounds, data, marker, color=color, lw=2, markersize=7)
        ax.set_xlabel("FL Round")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(rounds)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"FL metrics summary saved → {save_path}")
