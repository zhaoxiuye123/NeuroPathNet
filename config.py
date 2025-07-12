import argparse
from typing import Dict, Any

class Config:
    """
    Configuration class for NeuroPathNet based on the paper:
    "NeuroPathNet: Dynamic Path Trajectory Learning for Brain Functional Connectivity Analysis"

    This class organizes hyperparameters, dataset settings, model architecture details,
    training parameters, and evaluation metrics as described in the paper.

    Sections:
    - Data: Dataset paths, preprocessing, partitioning schemes (Yeo-7, Yeo-17, Schaefer-100)
    - Model: Architecture hyperparameters (e.g., modules like PM, GE, MHA, TP for ablation)
    - Training: Optimizer, learning rate (0.1 as per paper), batch size, epochs
    - Evaluation: Metrics (ACC, AUC, F1, SEN, SPE), cross-validation (5-fold)
    - Experiment: Ablation studies, grid search parameters
    """

    def __init__(self):
        # Data-related configurations
        self.data = {
            'dataset_name': 'ABIDE',  # Primary dataset from paper, can be changed to HCP or others
            'data_path': './data/ABIDE/',  # Placeholder; update with actual path
            'partition_scheme': 'Schaefer-100',  # Default from ablation study (best performance)
            'available_partitions': ['Yeo-7', 'Yeo-17', 'Schaefer-100'],  # As tested in paper Table IV
            'num_partitions': {  # Number of partitions per scheme from paper
                'Yeo-7': 7,
                'Yeo-17': 17,
                'Schaefer-100': 10
            },
            'time_series_length': 200,  # Assumed typical fMRI sequence length; adjust as needed
            'preprocess': True,  # Whether to apply preprocessing (e.g., normalization)
            'fold': 5  # 5-fold cross-validation as per paper
        }

        # Model-related configurations
        self.model = {
            'input_dim': 1,  # Connection strength time series (scalar per path)
            'hidden_dim': 64,  # Hidden dimension for temporal modeling; tunable
            'num_layers': 2,  # Number of layers in temporal neural network to avoid deep network issues (gradient vanishing)
            'num_heads': 4,  # For multi-head attention (MHA) mechanism
            'enable_modules': {  # For ablation studies as in Table V
                'PM': True,  # Path Modeling
                'GE': True,  # Global Integration
                'MHA': True,  # Multi-Head Attention
                'TP': True   # Time Pooling
            },
            'dropout': 0.1  # Dropout rate to prevent overfitting
        }

        # Training-related configurations
        self.training = {
            'optimizer': 'Adam',  # Common choice; paper doesn't specify, but suitable for stability
            'learning_rate': 0.1,  # As specified in paper for fast convergence and stability
            'batch_size': 32,  # Tunable; grid search suggested in paper
            'epochs': 100,  # Reasonable default; adjust based on convergence
            'weight_decay': 1e-4,  # Regularization to prevent overfitting
            'scheduler': 'StepLR',  # To adjust LR; paper mentions initial high LR for convergence
            'scheduler_step': 20,
            'scheduler_gamma': 0.5
        }

        # Evaluation-related configurations
        self.evaluation = {
            'metrics': ['ACC', 'AUC', 'F1', 'SEN', 'SPE'],  # As used in paper Tables I, II, etc.
            'cross_validation': True,  # 5-fold as per paper
            'num_folds': 5
        }

        # Experiment-related configurations
        self.experiment = {
            'run_ablation': False,  # Set to True to run ablation on modules
            'ablation_sequence': ['PM', 'PM+GE', 'PM+GE+MHA', 'PM+GE+MHA+TP'],  # Gradual addition as in paper
            'grid_search': {  # For parameter analysis as mentioned in paper Section B
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'hidden_dim': [32, 64, 128],
                'num_layers': [1, 2, 3],
                'batch_size': [16, 32, 64]
            },
            'seed': 42,  # For reproducibility
            'log_dir': './logs/',  # Directory for saving logs and models
            'model_save_path': './models/neuropathnet.pth'
        }

    def get_config(self) -> Dict[str, Any]:
        """Return the entire configuration as a dictionary."""
        return {
            'data': self.data,
            'model': self.model,
            'training': self.training,
            'evaluation': self.evaluation,
            'experiment': self.experiment
        }

    def update_from_args(self, args: argparse.Namespace):
        """Update configurations based on command-line arguments."""
        # Example: Update learning rate if provided
        if args.learning_rate is not None:
            self.training['learning_rate'] = args.learning_rate
        # Add similar updates for other args as needed

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments to override default configs."""
    parser = argparse.ArgumentParser(description='NeuroPathNet Configuration')
    parser.add_argument('--dataset', type=str, default='ABIDE', help='Dataset name')
    parser.add_argument('--partition', type=str, default='Schaefer-100', help='Partition scheme')
    parser.add_argument('--lr', type=float, default=0.1, dest='learning_rate', help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--run_ablation', action='store_true', help='Run ablation study')
    # Add more arguments as needed based on paper's tunable params
    return parser.parse_args()

# If running config.py directly, print the default config
if __name__ == "__main__":
    config = Config()
    print(config.get_config())


