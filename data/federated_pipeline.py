import os
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Tuple

# Set up robust logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def dirichlet_partition_indices(labels: np.ndarray, num_clients: int, alpha: float = 0.5, random_state: int = 42) -> List[np.ndarray]:
    """
    Partitions indices non-IID based on Dirichlet distribution.
    """
    np.random.seed(random_state)
    unique_labels = np.unique(labels)
    client_indices = [[] for _ in range(num_clients)]
    
    for label in unique_labels:
        # Get indices corresponding to this label
        idx = np.where(labels == label)[0]
        np.random.shuffle(idx)
        
        # Dirichlet proportions for this label across all clients
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Convert proportions to integer counts
        splits = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        label_splits = np.split(idx, splits)
        
        for i in range(num_clients):
            client_indices[i].extend(label_splits[i])
            
    # Shuffle indices within each client
    for i in range(num_clients):
        np.random.shuffle(client_indices[i])
        client_indices[i] = np.array(client_indices[i], dtype=int)
        
    return client_indices

def enforce_client_constraints_indices(client_indices: List[np.ndarray], min_samples: int = 20, random_state: int = 42) -> Tuple[List[np.ndarray], List[int]]:
    """
    Removes clients with fewer than min_samples and redistributes their indices evenly.
    Ensures no data loss.
    """
    valid_indices = []
    removed_data = []
    removed_clients = []
    
    for i, idx_arr in enumerate(client_indices):
        if len(idx_arr) < min_samples:
            removed_clients.append(i)
            removed_data.extend(idx_arr)
        else:
            valid_indices.append(idx_arr)
            
    if not removed_data:
        return client_indices, []
        
    logger.info(f"Removed {len(removed_clients)} clients due to size constraints (<{min_samples}). Redistributing {len(removed_data)} samples...")
    
    np.random.seed(random_state)
    removed_data = np.array(removed_data)
    np.random.shuffle(removed_data)
    
    num_valid = len(valid_indices)
    if num_valid == 0:
        raise ValueError("All clients were removed! Re-check parameters.")
        
    # Standard array_split redistributes elements evenly
    chunks = np.array_split(removed_data, num_valid)
    
    for i in range(num_valid):
        # Merge redistributed indices into the valid clients
        valid_indices[i] = np.concatenate([valid_indices[i], chunks[i]])
        np.random.shuffle(valid_indices[i])
        
    return valid_indices, removed_clients

def build_pipeline():
    # 1. Project Specifications
    dataset_path = r"road_multi_label.csv"
    output_base_dir = r"..\Dataset"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 2. Load Data securely
    logger.info(f"Loading raw dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    initial_len = len(df)
    
    # Preprocessing
    df.dropna(inplace=True)
    if len(df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(df)} rows with NaNs.")
    
    if 'label' not in df.columns:
        raise ValueError("Missing 'label' column. Validation failed.")
        
    # 3. Split 80/20 Using Indices 
    # Use random permutations for huge multi-label datastream to protect against stratify assertion errors
    logger.info("Splitting dataset 80% train / 20% test...")
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(df))
    split_point = int(0.8 * len(df))
    
    train_indices = shuffled_indices[:split_point]
    test_indices = shuffled_indices[split_point:]
    
    logger.info(f"Train size: {len(train_indices)}, Test size: {len(test_indices)}")
    
    logger.info("Saving global train and test sets to output (Parquet format)...")
    # Safe index subsets
    df.iloc[train_indices].to_parquet(os.path.join(output_base_dir, "train.parquet"), index=False)
    df.iloc[test_indices].to_parquet(os.path.join(output_base_dir, "test.parquet"), index=False)
    
    # Reference target metrics for Dirichlet Partitioning
    train_labels = df['label'].values[train_indices]
    
    # 4. Enforce Federated Rule Concepts
    concepts = {
        "concept_1": 10,
        "concept_2": 20,
        "concept_3": 50
    }
    
    metadata = {}
    
    for concept_name, num_clients in concepts.items():
        logger.info(f"\n--- Processing {concept_name} (Target: {num_clients} clients) ---")
        concept_dir = os.path.join(output_base_dir, concept_name)
        os.makedirs(concept_dir, exist_ok=True)
        
        # Map: Non-IID 
        client_idx_list = dirichlet_partition_indices(train_labels, num_clients, alpha=0.5, random_state=42)
        
        # Modulate: Min-Sample Redistribution
        valid_client_idx_list, removed_ids = enforce_client_constraints_indices(client_idx_list, min_samples=20)
        
        # Validation checks
        total_partitioned = sum(len(idx) for idx in valid_client_idx_list)
        assert total_partitioned == len(train_indices), "FATAL EXCEPTION: Data loss/Duplication detected during redistribution!"
        
        counts = []
        concept_dfs = []
        for c_id, idx in enumerate(valid_client_idx_list):
            assert len(idx) >= 20, f"FATAL EXCEPTION: Client {c_id} has < 20 samples!"
            counts.append(len(idx))
            
            # Subselect via global index arrays
            global_indices = train_indices[idx]
            client_df = df.iloc[global_indices].copy()
            client_df['client_id'] = c_id
            concept_dfs.append(client_df)
            
        # Optimization: Save as a single massively compressed parquet file instead of multiple CSVs
        logger.info(f"Saving single optimized {concept_name}.parquet...")
        combined_df = pd.concat(concept_dfs, ignore_index=True)
        out_path = os.path.join(output_base_dir, f"{concept_name}.parquet")
        combined_df.to_parquet(out_path, index=False)
        
        logger.info(f"{concept_name} saved with {len(valid_client_idx_list)} active clients to {out_path}.")
        
        # Accumulate Statistics
        metadata[concept_name] = {
            "initial_num_clients": num_clients,
            "removed_clients": removed_ids,
            "final_num_clients": len(valid_client_idx_list),
            "samples_per_client": counts,
            "total_samples": total_partitioned
        }
        
    # 5. Build Aggregated Stats File
    stats_path = os.path.join(output_base_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    logger.info(f"Exported overall metadata to {stats_path}")
    logger.info("Federated data pipeline execution precisely complete.")

if __name__ == "__main__":
    build_pipeline()
