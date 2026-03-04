"""
data_loader.py
==============
USTC-TFC2016 dataset loader for ResNet-50.

Pipeline:
  1. Decompress .7z archives (py7zr)
  2. Parse .pcap files (scapy) — extract first N bytes from each packet payload
  3. Assemble per-flow byte sequences → reshape to 2D grayscale image → save as PNG
  4. Build PyTorch Dataset with torchvision transforms for ResNet-50
  5. Partition into IID or Non-IID federated datasets

Paper reference: Section IV-A — USTC-TFC2016, 70%/30% train/test split.
"""

import os
import glob
import hashlib
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
CLASS_MAP = {"Benign": 0, "Malware": 1}   # Binary labels

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ──────────────────────────────────────────────────────────────────────────────
# Archive extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_archives(raw_root: str) -> None:
    """
    Decompress all .7z files under raw_root in-place.
    If py7zr is not installed, logs a warning and skips.
    """
    try:
        import py7zr
    except ImportError:
        archives = glob.glob(os.path.join(raw_root, "**", "*.7z"), recursive=True)
        if archives:
            logger.warning(
                f"py7zr not found -- skipping extraction of "
                f"{len(archives)} .7z archive(s). Run: pip install py7zr"
            )
        return

    for archive in glob.glob(os.path.join(raw_root, "**", "*.7z"), recursive=True):
        out_dir = os.path.splitext(archive)[0]
        if os.path.exists(out_dir):
            logger.info(f"[skip] already extracted: {archive}")
            continue
        logger.info(f"Extracting {archive} -> {out_dir}")
        with py7zr.SevenZipFile(archive, mode="r") as z:
            z.extractall(path=out_dir)


# ──────────────────────────────────────────────────────────────────────────────
# PCAP → Image conversion
# ──────────────────────────────────────────────────────────────────────────────

def _flow_bytes(pcap_path: str, max_bytes: int = 784) -> List[np.ndarray]:
    """
    Parse a pcap file and extract up to `max_bytes` payload bytes per packet.
    Returns a list of byte arrays (one per packet with non-empty payload).
    """
    try:
        import dpkt
    except ImportError:
        logger.warning("dpkt not installed -- cannot parse pcap. Run: pip install dpkt")
        return []

    samples = []
    try:
        with open(pcap_path, "rb") as f:
            pcap = dpkt.pcap.Reader(f)
            for ts, buf in pcap:
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    ip = eth.data
                    if hasattr(ip, 'data') and hasattr(ip.data, 'data'):
                        raw = bytes(ip.data.data)
                        if raw:
                            arr = np.frombuffer(raw[:max_bytes], dtype=np.uint8)
                            if len(arr) < max_bytes:
                                arr = np.pad(arr, (0, max_bytes - len(arr)), constant_values=0)
                            samples.append(arr)
                except Exception:
                    continue
    except Exception as e:
        logger.warning(f"Failed to read {pcap_path}: {e}")
        return []

    return samples


def _bytes_to_image(byte_arr: np.ndarray, image_size: int = 224) -> Image.Image:
    """
    Convert a 1D byte array to a square PIL grayscale image
    then convert to RGB for ResNet-50 (3-channel).
    """
    side = int(np.ceil(np.sqrt(len(byte_arr))))
    padded = np.pad(byte_arr, (0, side * side - len(byte_arr)), constant_values=0)
    gray = padded.reshape(side, side).astype(np.uint8)
    img = Image.fromarray(gray, mode="L").resize((image_size, image_size), Image.BILINEAR)
    return img.convert("RGB")


def preprocess_pcaps(
    raw_root: str,
    output_dir: str,
    image_size: int = 224,
    max_bytes: int = 784,
) -> None:
    """
    Walk raw_root (Benign/ and Malware/ subdirs), parse every .pcap,
    convert each packet's payload to an image, save to output_dir.

    Output structure:
        output_dir/
          Benign/<traffic_type>/<hash>.png
          Malware/<traffic_type>/<hash>.png
    """
    for class_name, label in CLASS_MAP.items():
        class_dir = os.path.join(raw_root, class_name)
        if not os.path.isdir(class_dir):
            logger.warning(f"Class directory not found: {class_dir}")
            continue

        pcap_files = glob.glob(os.path.join(class_dir, "**", "*.pcap"), recursive=True)
        logger.info(f"[{class_name}] Found {len(pcap_files)} pcap files")

        for pcap_path in pcap_files:
            traffic_type = Path(pcap_path).stem
            out_subdir = os.path.join(output_dir, class_name, traffic_type)
            os.makedirs(out_subdir, exist_ok=True)

            byte_arrays = _flow_bytes(pcap_path, max_bytes=max_bytes)
            for i, arr in enumerate(byte_arrays):
                img_name = f"pkt_{i:06d}.png"
                img_path = os.path.join(out_subdir, img_name)
                if not os.path.exists(img_path):
                    _bytes_to_image(arr, image_size).save(img_path)

    logger.info(f"Pre-processing complete -> {output_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────────────

class USTCDataset(Dataset):
    """
    Loads pre-processed PNG images from the output of preprocess_pcaps().
    Folder structure: processed_dir/{Benign,Malware}/<traffic_type>/<img>.png
    """

    def __init__(
        self,
        processed_dir: str,
        image_size: int = 224,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.transform = transform or _default_transform(image_size)
        self.samples: List[Tuple[str, int]] = []

        for class_name, label in CLASS_MAP.items():
            class_path = os.path.join(processed_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for img_path in glob.glob(
                os.path.join(class_path, "**", "*.png"), recursive=True
            ):
                self.samples.append((img_path, label))

        logger.info(
            f"USTCDataset loaded: {len(self.samples)} samples "
            f"(Benign={sum(1 for _,l in self.samples if l==0)}, "
            f"Malware={sum(1 for _,l in self.samples if l==1)})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def _default_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Train / Validation / Test split
# ──────────────────────────────────────────────────────────────────────────────

def split_dataset(
    dataset: USTCDataset,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    """
    Split into train / val / test.
    Paper: 70% train, 30% test; train further split 80/20 for val.
    """
    n = len(dataset)
    idx = list(range(n))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_test  = int(n * (1 - train_ratio))
    n_train = n - n_test
    n_val   = int(n_train * val_ratio)
    n_train -= n_val

    train_idx = idx[:n_train]
    val_idx   = idx[n_train : n_train + n_val]
    test_idx  = idx[n_train + n_val :]

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Federated Partitioning
# ──────────────────────────────────────────────────────────────────────────────

def iid_partition(
    dataset: Subset,
    num_clients: int,
    seed: int = 42,
) -> Dict[int, Subset]:
    """Shuffle and split equally among clients (IID)."""
    idx = list(dataset.indices)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    chunks = np.array_split(idx, num_clients)
    return {i: Subset(dataset.dataset, list(c)) for i, c in enumerate(chunks)}


def non_iid_partition(
    dataset: Subset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> Dict[int, Subset]:
    """
    Non-IID partition using Dirichlet distribution.
    alpha controls heterogeneity: smaller alpha → more skewed.
    """
    rng = np.random.default_rng(seed)

    # Gather labels for the subset
    labels = np.array([dataset.dataset.samples[i][1] for i in dataset.indices])
    num_classes = len(CLASS_MAP)
    class_indices = {c: np.where(labels == c)[0] for c in range(num_classes)}

    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    for c in range(num_classes):
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()
        splits = (np.cumsum(proportions) * len(class_indices[c])).astype(int)[:-1]
        shards = np.split(rng.permutation(class_indices[c]), splits)
        for client_id, shard in enumerate(shards):
            client_indices[client_id].extend(
                [dataset.indices[i] for i in shard]
            )

    return {
        i: Subset(dataset.dataset, idxs)
        for i, idxs in client_indices.items()
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main build function
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(
    raw_root: str,
    processed_root: str,
    image_size: int = 224,
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    num_clients: int = 10,
    partition: str = "iid",
    dirichlet_alpha: float = 0.5,
) -> Tuple[Dict[int, DataLoader], DataLoader, DataLoader, USTCDataset]:
    """
    Full pipeline:
      1. Extract archives (if needed)
      2. Pre-process pcaps → images (if needed)
      3. Build dataset, split train/val/test
      4. Partition train among clients (IID or Non-IID)
      5. Return client DataLoaders, val DataLoader, test DataLoader

    Returns:
        client_loaders: {client_id: DataLoader}
        val_loader: DataLoader
        test_loader: DataLoader
        full_dataset: USTCDataset (for inspection)
    """
    # Step 1: extract archives
    extract_archives(raw_root)

    # Step 2: preprocess pcaps -> images (if not already done)
    if not os.path.exists(processed_root) or len(
        glob.glob(os.path.join(processed_root, "**", "*.png"), recursive=True)
    ) == 0:
        # Check dpkt is available before attempting preprocessing
        try:
            import dpkt  # type: ignore  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "dpkt is required to convert pcap files to images.\n"
                "Install it with: .\\venv\\Scripts\\pip install dpkt\n"
                "Then re-run the script."
            )
        logger.info("Starting pcap -> image pre-processing ...")
        preprocess_pcaps(raw_root, processed_root, image_size=image_size)
    else:
        logger.info(f"Using existing processed images at: {processed_root}")

    # Step 3: dataset + splits
    full_dataset = USTCDataset(processed_root, image_size=image_size)

    if len(full_dataset) == 0:
        raise RuntimeError(
            f"No images found in '{processed_root}'.\n"
            "Possible causes:\n"
            "  1. dpkt not installed -> run: .\\venv\\Scripts\\pip install dpkt\n"
            "  2. .7z archives not extracted -> run: .\\venv\\Scripts\\pip install py7zr\n"
            "  3. Raw dataset path is wrong -> check --raw_data argument\n"
            f"     Current raw_data path: {raw_root}"
        )

    train_set, val_set, test_set = split_dataset(
        full_dataset, train_ratio, val_ratio, seed
    )

    # Step 4: partition
    if partition == "iid":
        client_subsets = iid_partition(train_set, num_clients, seed)
    else:
        client_subsets = non_iid_partition(
            train_set, num_clients, alpha=dirichlet_alpha, seed=seed
        )

    # Step 5: DataLoaders
    # Windows requires num_workers=0 unless inside if __name__ == '__main__'
    import platform
    effective_workers = 0 if platform.system() == "Windows" else num_workers

    def _make_loader(subset, shuffle=False):
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=effective_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=lambda wid: np.random.seed(seed + wid),
        )

    client_loaders = {i: _make_loader(s, shuffle=True) for i, s in client_subsets.items()}
    val_loader  = _make_loader(val_set, shuffle=False)
    test_loader = _make_loader(test_set, shuffle=False)

    # ── Print dataset description ──
    n_total = len(full_dataset)
    n_benign  = sum(1 for _, l in full_dataset.samples if l == 0)
    n_malware = sum(1 for _, l in full_dataset.samples if l == 1)
    n_train   = len(train_set)
    n_test    = len(test_set)
    samples_per_client = [len(s) for s in client_subsets.values()]

    print("\n" + "=" * 55)
    print("Dataset: USTC-TFC2016")
    print(f"Total samples : {n_total}")
    print(f"Classes       : 2 (Benign={n_benign}, Malicious={n_malware})")
    print(f"Training samples : {n_train}")
    print(f"Testing samples  : {n_test}")
    print(f"Number of clients: {num_clients}")
    print(f"Data distribution: {partition.upper()}")
    print(f"Samples per client: ~{int(np.mean(samples_per_client))}")
    print("=" * 55 + "\n")

    return client_loaders, val_loader, test_loader, full_dataset
