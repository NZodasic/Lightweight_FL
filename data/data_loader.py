import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CANTabularDataset(Dataset):
    """
    Reads tabular CAN data from a Parquet file.
    Extracts ID and DATA0-7, normalizes them, and returns (features, label).
    """
    def __init__(self, data_path_or_df, max_samples=None):
        if isinstance(data_path_or_df, str):
            self.data_path = data_path_or_df
            df = pd.read_parquet(self.data_path)
        else:
            self.data_path = "DataFrame"
            df = data_path_or_df
            
        if max_samples is not None and max_samples < len(df):
            df = df.sample(n=max_samples, random_state=42)
            
        feature_cols = ['ID', 'DATA0', 'DATA1', 'DATA2', 'DATA3', 'DATA4', 'DATA5', 'DATA6', 'DATA7']
        
        # Verify columns exist
        for col in feature_cols:
            if col not in df.columns:
                raise ValueError(f"Missing expected column: {col}")
                
        X = df[feature_cols].values.astype(np.float32)
        
        # Normalization
        # ID max normalization (to avoid massive values skewing the network)
        max_id = np.max(X[:, 0]) if len(X) > 0 else 1.0
        if max_id == 0: max_id = 1.0
        X[:, 0] = X[:, 0] / max_id
        
        # DATA0-7 are bytes (0-255)
        X[:, 1:] = X[:, 1:] / 255.0
        
        self.data = X
        
        # Binary classification mapping
        # Label 0 is benign, > 0 is malicious
        if 'label' in df.columns:
            y = df['label'].values
            self.labels = (y > 0).astype(np.int64)
        else:
            self.labels = np.zeros(len(X), dtype=np.int64)
            
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def get_dataset(data_path, img_size=None, max_samples=None):
    """Returns the full parsed dataset using the tabular structure."""
    print(f"Loading dataset from {data_path}...", flush=True)
    dataset = CANTabularDataset(data_path, max_samples=max_samples)
    print(f"Dataset loaded. Total samples: {len(dataset)}, Classes: 2 (Benign, Malicious)", flush=True)
    return dataset

def get_prepartitioned_client_datasets(concept_parquet_path):
    """
    Loads a pre-partitioned concept parquet file.
    Returns a dictionary of Datasets grouped by 'client_id'.
    """
    df = pd.read_parquet(concept_parquet_path)
    
    if 'client_id' not in df.columns:
        raise ValueError("Missing 'client_id' column in the partition parquet file.")
        
    client_ids = df['client_id'].unique()
    num_clients = len(client_ids)
    
    print(f"Loaded concept file with {num_clients} clients.", flush=True)
    
    client_datasets = {}
    for cid in np.sort(client_ids):
        client_df = df[df['client_id'] == cid].copy()
        client_datasets[cid] = CANTabularDataset(client_df)
        
    return client_datasets, num_clients

def get_dataloaders(client_datasets_dict, batch_size=64):
    """Converts a dictionary of Datasets into DataLoaders."""
    dataloaders = []
    # Ensure ordered by client ID matches 0, ..., N-1
    sorted_cids = sorted(client_datasets_dict.keys())
    for cid in sorted_cids:
        dl = DataLoader(client_datasets_dict[cid], batch_size=batch_size, shuffle=True, drop_last=True)
        dataloaders.append(dl)
    return dataloaders
