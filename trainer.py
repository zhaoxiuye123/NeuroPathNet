import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score
import numpy as np
import os
from typing import Dict, List, Tuple

# Assuming imports from other files; in practice, import Config, get_dataloaders, NeuroPathNet
# For this file, we'll assume they are available or define placeholders

# Placeholder for model and dataloader; replace with actual imports
# from model import NeuroPathNet
# from data_loader import get_dataloaders
# from config import Config

def train_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        paths, labels = batch
        paths, labels = paths.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(paths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, Dict[str, float]]:
    """Evaluate for one epoch and compute metrics."""
    model.eval()
    total_loss = 0.0
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            paths, labels = batch
            paths, labels = paths.to(device), labels.to(device)
            
            outputs = model(paths)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Probability for class 1
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels_np)
    
    avg_loss = total_loss / len(loader)
    
    metrics = {
        'ACC': accuracy_score(all_labels, all_preds),
        'AUC': roc_auc_score(all_labels, all_probs),
        'F1': f1_score(all_labels, all_preds),
        'SEN': recall_score(all_labels, all_preds),  # Sensitivity
        'SPE': precision_score(all_labels, all_preds, pos_label=0)  # Specificity: precision for class 0
    }
    return avg_loss, metrics

def run_training(config: Dict, fold: int = None) -> Dict[str, float]:
    """Run training and evaluation for one fold or full dataset."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get dataloaders (assuming get_dataloaders returns train/test for one fold if fold provided)
    train_loader, test_loader = get_dataloaders(config)  # Modify to pass fold if needed
    
    model = NeuroPathNet(config).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    scheduler = StepLR(optimizer, step_size=config['training']['scheduler_step'], gamma=config['training']['scheduler_gamma'])
    criterion = nn.CrossEntropyLoss()
    
    best_metrics = {}
    for epoch in range(config['training']['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, metrics = eval_epoch(model, test_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{config['training']['epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Metrics: {metrics}")
        
        # Save best model based on AUC or ACC
        if not best_metrics or metrics['AUC'] > best_metrics['AUC']:
            best_metrics = metrics
            torch.save(model.state_dict(), os.path.join(config['experiment']['log_dir'], f"best_model_fold{fold}.pth"))
    
    return best_metrics

def main(config: Dict):
    """Main training function handling CV, ablation, etc."""
    os.makedirs(config['experiment']['log_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    all_metrics = []
    
    if config['experiment']['run_ablation']:
        for abl in config['experiment']['ablation_sequence']:
            # Disable modules based on sequence
            enabled = set(abl.split('+'))
            for mod in config['model']['enable_modules']:
                config['model']['enable_modules'][mod] = mod in enabled
            print(f"Running ablation: {abl}")
            
            fold_metrics = run_cv(config)
            all_metrics.append((abl, fold_metrics))
    else:
        fold_metrics = run_cv(config)
        all_metrics.append(('Full', fold_metrics))
    
    # Log or print average metrics
    print("Final Metrics:")
    for name, mets in all_metrics:
        print(f"{name}: {mets}")

def run_cv(config: Dict) -> Dict[str, float]:
    """Run 5-fold CV and average metrics."""
    num_folds = config['data']['fold']
    subject_list = [f'subject_{i}' for i in range(945)]  # Placeholder; from data_loader
    
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=config['experiment']['seed'])
    
    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(subject_list)):
        # Update config or pass fold to get_dataloaders (need to modify data_loader accordingly)
        # For now, assume get_dataloaders handles it internally or modify here
        print(f"Fold {fold+1}/{num_folds}")
        metrics = run_training(config, fold=fold)
        fold_results.append(metrics)
    
    # Average metrics across folds
    avg_metrics = {k: np.mean([f[k] for f in fold_results]) for k in fold_results[0]}
    return avg_metrics

# If running directly
if __name__ == "__main__":
    # Dummy config for testing; replace with Config().get_config()
    dummy_config = {
        'data': {'fold': 5, 'dataset_name': 'ABIDE', 'data_path': './data/ABIDE/', 'partition_scheme': 'Schaefer-100', 'time_series_length': 200},
        'model': {'input_dim': 1, 'hidden_dim': 64, 'num_layers': 2, 'num_heads': 4, 'enable_modules': {'PM': True, 'GE': True, 'MHA': True, 'TP': True}, 'dropout': 0.1},
        'training': {'learning_rate': 0.1, 'epochs': 100, 'batch_size': 32, 'weight_decay': 1e-4, 'scheduler_step': 20, 'scheduler_gamma': 0.5},
        'experiment': {'run_ablation': False, 'ablation_sequence': ['PM', 'PM+GE', 'PM+GE+MHA', 'PM+GE+MHA+TP'], 'seed': 42, 'log_dir': './logs/'}
    }
    main(dummy_config)

