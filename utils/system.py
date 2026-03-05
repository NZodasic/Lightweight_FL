import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    Ensure reproducibility by setting all relevant seeds.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Auto match device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_optimal_batch_size(base_batch_size=64):
    """Adjusts batch size if running on CPU or low VRAM GPU."""
    if not torch.cuda.is_available():
        return max(16, base_batch_size // 4)
    return base_batch_size
