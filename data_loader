import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from typing import List, Tuple, Dict

# Assuming we have utility functions; we'll import from utils.py later if needed
# For now, define placeholder for partitioning schemes based on paper (Yeo-7, Yeo-17, Schaefer-100)
# In practice, these would load actual atlas mappings (region to module)

def get_partition_mapping(scheme: str) -> Dict[int, int]:
    """
    Returns a dummy mapping of brain regions to functional modules based on scheme.
    In real implementation, load from atlas files (e.g., Yeo atlas with 7/17 networks, Schaefer with 100 parcels).
    Assuming 200 regions for simplicity (common in fMRI like AAL or Desikan).
    Maps region_id (0 to 199) to module_id (0 to num_modules-1).
    """
    num_regions = 200  # Placeholder; adjust based on actual data
    if scheme == 'Yeo-7':
        num_modules = 7
    elif scheme == 'Yeo-17':
        num_modules = 17
    elif scheme == 'Schaefer-100':
        num_modules = 100
    else:
        raise ValueError(f"Unknown partition scheme: {scheme}")
    
    # Dummy mapping: assign regions evenly to modules
    mapping = {i: i % num_modules for i in range(num_regions)}
    return mapping

def compute_module_time_series(bold_signals: np.ndarray, mapping: Dict[int, int], num_modules: int) -> np.ndarray:
    """
    Aggregate region-level BOLD signals to module-level averages.
    bold_signals: (num_regions, T)
    Returns: (num_modules, T)
    """
    module_signals = np.zeros((num_modules, bold_signals.shape[1]))
    module_counts = np.zeros(num_modules)
    
    for region, module in mapping.items():
        module_signals[module] += bold_signals[region]
        module_counts[module] += 1
    
    module_signals /= module_counts[:, np.newaxis] + 1e-8  # Avoid division by zero
    return module_signals

def compute_dynamic_paths(module_signals: np.ndarray, num_modules: int, T: int) -> np.ndarray:
    """
    Compute dynamic connection strengths (paths) between modules.
    For each pair (i,j), w(t)_{i,j} = correlation or similarity between module i and j at time t.
    Here, use simple Pearson correlation over a sliding window, but for simplicity, assume instantaneous (or precomputed).
    Returns: (num_paths, T) where num_paths = num_modules * (num_modules - 1) / 2 or full matrix flattened.
    Paper treats paths as between all pairs, including self? But likely inter-module.
    For simplicity, return (num_modules^2, T) with w(t)_{i,j} = dot product or correlation proxy.
    """
    paths = np.zeros((num_modules * num_modules, T))
    path_idx = 0
    for i in range(num_modules):
        for j in range(num_modules):
            # Simple: element-wise product as proxy for connection strength (not accurate, but placeholder)
            paths[path_idx] = module_signals[i] * module_signals[j]  # Replace with actual dynamic FC computation
            path_idx += 1
    return paths

class BrainDataset(Dataset):
    """
    Custom PyTorch Dataset for brain fMRI data based on paper.
    Loads preprocessed BOLD signals, applies partitioning, computes dynamic paths.
    Assumes data structure: data_path/subject_id/bold.npy (num_regions x T) and label.txt (int label).
    Labels: e.g., for ABIDE: 0=TD, 1=ASD
    """
    def __init__(self, data_path: str, partition_scheme: str, time_series_length: int, subject_list: List[str], preprocess: bool = True):
        self.data_path = data_path
        self.partition_scheme = partition_scheme
        self.time_series_length = time_series_length
        self.subject_list = subject_list
        self.preprocess = preprocess
        self.mapping = get_partition_mapping(partition_scheme)
        self.num_modules = max(self.mapping.values()) + 1  # From paper's partition counts

    def __len__(self) -> int:
        return len(self.subject_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        subject_id = self.subject_list[idx]
        bold_file = os.path.join(self.data_path, subject_id, 'bold.npy')
        label_file = os.path.join(self.data_path, subject_id, 'label.txt')
        
        # Load BOLD signals (num_regions x T)
        bold_signals = np.load(bold_file).astype(np.float32)
        
        if self.preprocess:
            # Normalize: z-score per region
            bold_signals = (bold_signals - bold_signals.mean(axis=1, keepdims=True)) / (bold_signals.std(axis=1, keepdims=True) + 1e-8)
        
        # Aggregate to modules
        module_signals = compute_module_time_series(bold_signals, self.mapping, self.num_modules)
        
        # Compute dynamic paths: (num_paths x T)
        paths = compute_dynamic_paths(module_signals, self.num_modules, self.time_series_length)
        paths = torch.from_numpy(paths)  # (num_paths, T)
        
        # Load label
        with open(label_file, 'r') as f:
            label = int(f.read().strip())
        label = torch.tensor(label, dtype=torch.long)
        
        return paths, label

def get_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare train and test DataLoaders with 5-fold CV (as per paper).
    For simplicity, here we do one fold; in trainer.py, loop over folds.
    Assumes all subjects in a list.
    """
    data_config = config['data']
    dataset_name = data_config['dataset_name']
    data_path = data_config['data_path']
    partition_scheme = data_config['partition_scheme']
    time_series_length = data_config['time_series_length']
    batch_size = config['training']['batch_size']
    num_folds = data_config['fold']
    
    # Placeholder: list all subjects (in real case, scan directory)
    subject_list = [f'subject_{i}' for i in range(945)]  # ABIDE has 945 subjects as per paper
    
    # For 5-fold CV
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=config['experiment']['seed'])
    train_idx, test_idx = next(iter(kf.split(subject_list)))  # One fold for example; loop in trainer
    
    train_subjects = [subject_list[i] for i in train_idx]
    test_subjects = [subject_list[i] for i in test_idx]
    
    train_dataset = BrainDataset(data_path, partition_scheme, time_series_length, train_subjects)
    test_dataset = BrainDataset(data_path, partition_scheme, time_series_length, test_subjects)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# If running directly, test the loader
if __name__ == "__main__":
    # Dummy config for testing
    dummy_config = {
        'data': {'dataset_name': 'ABIDE', 'data_path': './data/ABIDE/', 'partition_scheme': 'Schaefer-100', 'time_series_length': 200, 'fold': 5},
        'training': {'batch_size': 32},
        'experiment': {'seed': 42}
    }
    train_loader, test_loader = get_dataloaders(dummy_config)
    print(f"Train loader size: {len(train_loader.dataset)}, Test loader size: {len(test_loader.dataset)}")
