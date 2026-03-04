"""
baseline.py
===========
Zero-shot structured pruning for ResNet-50 using `torch-pruning`.

Paper: "Lightweight Federated Learning for Efficient Network Intrusion Detection"
Algorithm:
  1. Server computes pruning mask ONCE before any FL training (zero-shot, data-independent).
  2. Importance score: L1-norm of each filter (channel).
  3. Low-importance filters are removed (structured — entire channels/filters).
  4. Pruned model + mask sent to all clients.

Sparsity levels tested: 0%, 50%, 70%, 90%
"""

import logging
import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def prune_model(
    model: nn.Module,
    sparsity: float,
    example_inputs: Optional[torch.Tensor] = None,
    image_size: int = 224,
) -> nn.Module:
    """
    Apply one-time zero-shot structured pruning to the model.

    Works by removing an entire `sparsity` fraction of filters from
    all convolutional layers (except the final fc head) using L1-norm
    importance scores (MagnitudeImportance with p=1).

    Args:
        model: The ResNet-50 model to prune.
        sparsity: Fraction of filters to remove (e.g., 0.5 → 50%).
        example_inputs: A dummy input tensor for dependency analysis.
        image_size: Used to create dummy input if not provided.

    Returns:
        pruned_model: A structurally pruned model (smaller, faster).
    """
    if sparsity <= 0.0:
        logger.info("Sparsity=0% — skipping pruning, returning original model.")
        return copy.deepcopy(model)

    try:
        import torch_pruning as tp  # type: ignore
    except ImportError:
        raise ImportError(
            "torch-pruning is required. Install it with: pip install torch-pruning"
        )

    pruned_model = copy.deepcopy(model)
    pruned_model.eval()

    if example_inputs is None:
        example_inputs = torch.randn(1, 3, image_size, image_size)

    # Identify layers to IGNORE (keep unchanged)
    # The fc head (Sequential with Linears) must not be pruned
    ignored_layers = []
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Linear):
            ignored_layers.append(module)

    # L1-norm importance (per paper's structured pruning approach)
    importance = tp.importance.MagnitudeImportance(p=1)

    pruner = tp.pruner.MagnitudePruner(
        pruned_model,
        example_inputs=example_inputs,
        importance=importance,
        iterative_steps=1,              # one-shot pruning (zero-shot in paper)
        pruning_ratio=sparsity,
        ignored_layers=ignored_layers,
        global_pruning=False,           # per-layer pruning ratio
    )

    # Execute pruning (removes channels in-place)
    pruner.step()

    logger.info(
        f"Structured pruning complete: sparsity={sparsity*100:.0f}%"
    )

    return pruned_model


def compute_pruning_stats(
    original_model: nn.Module,
    pruned_model: nn.Module,
    image_size: int = 224,
    device: str = "cpu",
) -> dict:
    """
    Compare original vs pruned model: params, FLOPs, size.

    Returns a dict with reduction statistics.
    """
    from models.model_utils import count_parameters, compute_flops, model_size_mb

    orig_params, _ = count_parameters(original_model)
    prnd_params, _ = count_parameters(pruned_model)

    orig_flops, _ = compute_flops(original_model, device=device)
    prnd_flops, _ = compute_flops(pruned_model, device=device)

    orig_size = model_size_mb(original_model)
    prnd_size = model_size_mb(pruned_model)

    param_red = (1 - prnd_params / max(orig_params, 1)) * 100
    flop_red  = (1 - prnd_flops / max(orig_flops, 1)) * 100
    size_red  = (1 - prnd_size / max(orig_size, 1)) * 100

    print("\n" + "─" * 55)
    print("Pruning Statistics")
    print("─" * 55)
    print(f"  {'Metric':<25} {'Original':>12}  {'Pruned':>12}  {'Reduction':>10}")
    print(f"  {'Parameters':<25} {orig_params:>12,}  {prnd_params:>12,}  {param_red:>9.1f}%")
    print(f"  {'GFLOPs':<25} {orig_flops:>12.3f}  {prnd_flops:>12.3f}  {flop_red:>9.1f}%")
    print(f"  {'Model size (MB)':<25} {orig_size:>12.2f}  {prnd_size:>12.2f}  {size_red:>9.1f}%")
    print("─" * 55 + "\n")

    return {
        "orig_params": orig_params,
        "pruned_params": prnd_params,
        "param_reduction_pct": param_red,
        "orig_flops_g": orig_flops,
        "pruned_flops_g": prnd_flops,
        "flop_reduction_pct": flop_red,
        "orig_size_mb": orig_size,
        "pruned_size_mb": prnd_size,
        "size_reduction_pct": size_red,
    }
