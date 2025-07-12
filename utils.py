import os
import numpy as np
import torch
import logging
from typing import Dict, List, Tuple
from scipy.stats import pearsonr
import nibabel as nib  # For loading atlas files if needed

def setup_logging(log_dir: str, log_file: str = 'training.log') -> logging.Logger:
    """
    Set up logging to file and console.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('NeuroPathNet')
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, log_file))
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def load_atlas_mapping(scheme: str, atlas_dir: str = './atlases/') -> Dict[int, int]:
    """
    Load actual atlas mapping from files.
    Assumes atlas files are NIfTI or text files mapping regions to modules.
    For simplicity, here we simulate loading; in practice, use nibabel to load parcellation.
    Returns mapping: region_id -> module_id
    """
    if scheme == 'Yeo-7':
        num_modules = 7
        # Simulate loading: e.g., load Yeo_7Networks.nii.gz and extract labels
        # mapping = {i: label for i, label in enumerate(nib.load(os.path.join(atlas_dir, 'Yeo_7Networks.nii.gz')).get_fdata().flatten())}
        mapping = {i: i % num_modules for i in range(200)}  # Dummy
    elif scheme == 'Yeo-17':
        num_modules = 17
        mapping = {i: i % num_modules for i in range(200)}  # Dummy
    elif scheme == 'Schaefer-100':
        num_modules = 100
        mapping = {i: i % num_modules for i in range(200)}  # Dummy
    else:
        raise ValueError(f"Unknown atlas scheme: {scheme}")
    
    return mapping

def normalize_bold_signals(bold_signals: np.ndarray) -> np.ndarray:
    """
    Normalize BOLD signals: z-score per region.
    bold_signals: (num_regions, T)
    """
    mean = bold_signals.mean(axis=1, keepdims=True)
    std = bold_signals.std(axis=1, keepdims=True) + 1e-8
    return (bold_signals - mean) / std

def compute_sliding_window_correlation(time_series: np.ndarray, window_size: int = 30) -> np.ndarray:
    """
    Compute dynamic functional connectivity using sliding window Pearson correlation.
    time_series: (num_modules, T)
    Returns: (num_modules, num_modules, T - window_size + 1) correlation matrices over time.
    """
    num_modules, T = time_series.shape
    num_windows = T - window_size + 1
    dyn_fc = np.zeros((num_modules, num_modules, num_windows))
    
    for t in range(num_windows):
        window = time_series[:, t:t+window_size]
        for i in range(num_modules):
            for j in range(i, num_modules):  # Symmetric
                corr, _ = pearsonr(window[i], window[j])
                dyn_fc[i, j, t] = corr
                dyn_fc[j, i, t] = corr
    
    return dyn_fc

def flatten_fc_to_paths(dyn_fc: np.ndarray) -> np.ndarray:
    """
    Flatten dynamic FC matrices to path time series.
    dyn_fc: (num_modules, num_modules, num_time_points)
    Returns: (num_paths, num_time_points) where num_paths = num_modules*(num_modules+1)/2 for upper triangle, or full.
    Here, full flatten including self and duplicates.
    """
    num_modules, _, num_time_points = dyn_fc.shape
    paths = dyn_fc.reshape(num_modules * num_modules, num_time_points)
    return paths

def compute_metrics(y_true: List[int], y_pred: List[int], y_prob: List[float]) -> Dict[str, float]:
    """
    Compute evaluation metrics as per paper.
    """
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
    
    metrics = {
        'ACC': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_prob),
        'F1': f1_score(y_true, y_pred),
        'SEN': recall_score(y_true, y_pred),
        'SPE': precision_score(y_true, y_pred, pos_label=0)  # Specificity as precision for negative class
    }
    return metrics

# If running directly, test utils
if __name__ == "__main__":
    logger = setup_logging('./logs/')
    logger.info("Utils test: Logging setup complete.")
    
    mapping = load_atlas_mapping('Yeo-7')
    print(f"Mapping for Yeo-7: {list(mapping.items())[:5]}")
    
    dummy_ts = np.random.randn(7, 100)
    dyn_fc = compute_sliding_window_correlation(dummy_ts, window_size=30)
    paths = flatten_fc_to_paths(dyn_fc)
    print(f"Paths shape: {paths.shape}")

