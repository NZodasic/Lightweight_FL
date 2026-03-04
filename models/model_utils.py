"""
model_utils.py
==============
Utilities for measuring and reporting model complexity:
  - Number of parameters (total and non-zero)
  - FLOPs (using thop)
  - Model size (MB)
  - Inference latency (ms)
"""

import time
import copy
import logging
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and non-zero (trainable) parameters.

    Returns:
        (total_params, nonzero_params)
    """
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum(
        (p != 0).sum().item() for p in model.parameters()
    )
    return int(total), int(nonzero)


def compute_flops(
    model: nn.Module,
    input_size: Tuple[int, int, int] = (3, 224, 224),
    device: str = "cpu",
) -> Tuple[float, int]:
    """
    Compute FLOPs and parameter count using `thop`.

    Returns:
        (flops_giga, params)
    """
    try:
        from thop import profile, clever_format  # type: ignore

        dummy = torch.randn(1, *input_size).to(device)
        model_copy = copy.deepcopy(model).to(device).eval()
        flops, params = profile(model_copy, inputs=(dummy,), verbose=False)
        return flops / 1e9, int(params)  # GFLOPs
    except ImportError:
        logger.warning("thop not installed — FLOPs estimation skipped.")
        return 0.0, 0


def model_size_mb(model: nn.Module) -> float:
    """Estimate model size in MB based on parameter count (float32)."""
    total, _ = count_parameters(model)
    return (total * 4) / (1024 ** 2)  # 4 bytes per float32


def measure_inference_latency(
    model: nn.Module,
    device: torch.device,
    input_size: Tuple[int, int, int] = (3, 224, 224),
    batch_size: int = 32,
    n_warmup: int = 10,
    n_runs: int = 100,
) -> float:
    """
    Measure average per-batch inference latency (ms).

    Returns:
        avg_latency_ms
    """
    model.eval().to(device)
    dummy = torch.randn(batch_size, *input_size).to(device)

    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            _ = model(dummy)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_runs):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

    avg_ms = (end - start) / n_runs * 1000
    return avg_ms


def print_model_summary(
    model: nn.Module,
    sparsity: float = 0.0,
    device: Optional[torch.device] = None,
    input_size: Tuple[int, int, int] = (3, 224, 224),
) -> Dict:
    """Print a full model complexity summary."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_params, nonzero_params = count_parameters(model)
    flops_g, _ = compute_flops(model, input_size, device=str(device))
    size_mb = model_size_mb(model)
    latency_ms = measure_inference_latency(model, device, input_size)

    print("\n" + "─" * 45)
    print("Model Complexity Report")
    print("─" * 45)
    print(f"  Architecture       : ResNet-50")
    print(f"  Pruning sparsity   : {sparsity * 100:.0f}%")
    print(f"  Total parameters   : {total_params:,}")
    print(f"  Non-zero parameters: {nonzero_params:,}")
    print(f"  Remaining params   : {nonzero_params / max(total_params, 1) * 100:.1f}%")
    print(f"  FLOPs              : {flops_g:.3f} GFLOPs")
    print(f"  Model size         : {size_mb:.2f} MB")
    print(f"  Inference latency  : {latency_ms:.2f} ms/batch")
    print("─" * 45 + "\n")

    return {
        "total_params": total_params,
        "nonzero_params": nonzero_params,
        "flops_g": flops_g,
        "size_mb": size_mb,
        "latency_ms": latency_ms,
    }



