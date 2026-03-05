import os
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.system import set_seed

class PcapImageDataset(Dataset):
    """
    Reads PCAP files directly from a zip archive, chunks them into 
    fixed sizes to simulate 32x32 images (1024 bytes).
    """
    def __init__(self, zip_path, img_size=32, transform=None, max_samples_per_class=5000):
        self.zip_path = zip_path
        self.img_size = img_size
        self.transform = transform
        self.max_samples = max_samples_per_class
        
        self.data = []
        self.labels = []
        
        # Load from zip
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            benign_files = [f for f in z.namelist() if f.startswith('Benign/') and f.endswith('.pcap')]
            malware_files = [f for f in z.namelist() if f.startswith('Malware/') and f.endswith('.pcap')]
            
            # Load Benign (Label 0)
            self._process_files(z, benign_files, label=0, target_count=max_samples_per_class)
            
            # Load Malware (Label 1)
            self._process_files(z, malware_files, label=1, target_count=max_samples_per_class)
            
        self.data = np.array(self.data, dtype=np.float32) / 255.0  # normalize
        self.labels = np.array(self.labels, dtype=np.int64)

    def _process_files(self, z, file_list, label, target_count):
        chunk_size = self.img_size * self.img_size
        count = 0
        for fname in file_list:
            if count >= target_count:
                break
            with z.open(fname, 'r') as f:
                content = f.read()
                # Skip global pcap header (24 bytes)
                content = content[24:]
                # Chunk data
                for i in range(0, len(content) - chunk_size, chunk_size):
                    chunk = np.frombuffer(content[i:i+chunk_size], dtype=np.uint8)
                    if len(chunk) == chunk_size:
                        # ResNet 50 expects 3 channels usually. We tile 1 channel 3 times
                        img = chunk.reshape(1, self.img_size, self.img_size)
                        img = np.repeat(img, 3, axis=0) # Shape: (3, 32, 32)
                        self.data.append(img)
                        self.labels.append(label)
                        count += 1
                    if count >= target_count:
                        break

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.transform:
            x = self.transform(x)
        return x, y


def get_dataset(zip_path, img_size=32, max_samples=5000):
    """Returns the full parsed dataset."""
    print("Loading dataset...", flush=True)
    dataset = PcapImageDataset(zip_path, img_size=img_size, max_samples_per_class=max_samples)
    print(f"Dataset loaded. Total samples: {len(dataset)}, Classes: 2 (Benign, Malicious)", flush=True)
    return dataset


def partition_data(dataset, num_clients, partition_type="iid", dirichlet_alpha=0.5):
    """
    Partitions the dataset among clients.
    partition_type: 'iid' or 'noniid'
    returns: List of Subsets or DataLoaders
    """
    num_items = int(len(dataset) / num_clients)
    client_datasets = {}
    
    if partition_type == "iid":
        all_idxs = np.random.permutation(len(dataset))
        for i in range(num_clients):
            client_datasets[i] = all_idxs[i * num_items : (i + 1) * num_items]
            
    elif partition_type == "noniid":
        # Dirichlet distribution for non-IID
        min_size = 0
        min_require_size = 10
        N = len(dataset)
        labels = dataset.labels
        
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(2): # 2 classes
                idx_k = np.where(labels == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(dirichlet_alpha, num_clients))
                
                # Balance
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                
        for i in range(num_clients):
            client_datasets[i] = idx_batch[i]
            np.random.shuffle(client_datasets[i])
            
    return client_datasets

def get_dataloaders(dataset, client_datasets, batch_size=64):
    """Converts partitioned indices into DataLoaders."""
    dataloaders = []
    for i in range(len(client_datasets)):
        subset = torch.utils.data.Subset(dataset, client_datasets[i])
        dl = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=True)
        dataloaders.append(dl)
    return dataloaders
