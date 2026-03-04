"""
device_utils.py
===============
Auto-detects the best available device (CUDA / CPU)
and sets reproducibility seeds.
"""

import os
import random
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU (no CUDA device found)")
    return device


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    logger.info(f"All seeds fixed to {seed}")


def suggest_batch_size(device: torch.device, default: int = 32) -> int:
    """
    Suggest a safe batch size based on available GPU memory.
    Falls back to default on CPU or unknown GPU.
    """
    if device.type == "cpu":
        return min(default, 16)

    try:
        free_mb = torch.cuda.get_device_properties(device).total_memory / 1e6
        if free_mb >= 8000:   # >= 8 GB
            return 64
        elif free_mb >= 4000:  # >= 4 GB
            return 32
        else:
            return 16
    except Exception:
        return default
