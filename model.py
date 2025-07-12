
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class NeuroPathNet(nn.Module):
    """
    NeuroPathNet model implementation based on the paper:
    "NeuroPathNet: Dynamic Path Trajectory Learning for Brain Functional Connectivity Analysis"

    The model processes dynamic path trajectories (time series of connection strengths between brain modules).
    Key components:
    - Path Modeling (PM): Temporal modeling of each path's sequence using LSTM.
    - Time Pooling (TP): Aggregates temporal features (e.g., mean pooling).
    - Multi-Head Attention (MHA): Cross-path attention to capture interactions.
    - Global Integration (GE): Combines features for final classification.

    Ablation support: Modules can be enabled/disabled via config.

    Input: (B, num_paths, T) - Batch of path time series.
    Output: (B, num_classes) - Logits for classification (e.g., 2 classes: TD vs ASD).
    """

    def __init__(self, config: Dict):
        super(NeuroPathNet, self).__init__()
        model_config = config['model']
        data_config = config['data']

        # Derive num_modules from partition scheme
        self.num_modules = data_config['num_partitions'][data_config['partition_scheme']]
        # Assuming full matrix including self-loops, num_paths = num_modules ** 2
        # Adjust if paper uses upper triangle (undirected)
        self.num_paths = self.num_modules ** 2
        self.input_dim = model_config['input_dim']  # 1 for scalar time series per path
        self.hidden_dim = model_config['hidden_dim']
        self.num_layers = model_config['num_layers']
        self.num_heads = model_config['num_heads']
        self.dropout = model_config['dropout']
        self.enable_modules = model_config['enable_modules']
        self.time_series_length = data_config['time_series_length']
        self.num_classes = 2  # Binary classification (e.g., ABIDE: TD=0, ASD=1); adjust if needed

        # Path Modeling (PM): LSTM for temporal sequence modeling per path
        if self.enable_modules['PM']:
            self.path_lstm = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=True,  # Bidirectional for better temporal capture
                dropout=self.dropout if self.num_layers > 1 else 0
            )
        else:
            # If PM disabled, use identity or simple embedding
            self.path_lstm = nn.Identity()

        # Multi-Head Attention (MHA): Attention across paths
        if self.enable_modules['MHA']:
            self.attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dim * 2,  # Bidirectional LSTM doubles hidden_dim
                num_heads=self.num_heads,
                dropout=self.dropout,
                batch_first=True
            )
        else:
            self.attention = nn.Identity()

        # Time Pooling (TP): After LSTM, pool over time dimension
        # Using mean pooling; could be max or attention-based
        self.time_pooling = nn.AdaptiveAvgPool1d(1) if self.enable_modules['TP'] else nn.Identity()

        # Global Integration (GE): Fully connected layers for classification
        if self.enable_modules['GE']:
            feature_dim = self.hidden_dim * 2 if self.enable_modules['PM'] else self.input_dim
            if self.enable_modules['TP']:
                feature_dim *= 1  # After pooling, time dim is 1, but flattened
            else:
                feature_dim *= self.time_series_length  # If no TP, flatten time
            self.fc1 = nn.Linear(self.num_paths * feature_dim, self.hidden_dim)
            self.fc2 = nn.Linear(self.hidden_dim, self.num_classes)
        else:
            # Minimal classifier if GE disabled
            self.fc1 = nn.Identity()
            self.fc2 = nn.Linear(self.num_paths * self.input_dim * self.time_series_length, self.num_classes)

        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: (B, num_paths, T) - Input path time series.

        Processing order based on paper: PM -> TP -> MHA -> GE
        """
        B, num_paths, T = x.shape
        assert num_paths == self.num_paths, f"Expected {self.num_paths} paths, got {num_paths}"

        # Reshape for LSTM: Treat each path independently, but batch them
        # (B * num_paths, T, input_dim)
        x = x.view(B * self.num_paths, T, self.input_dim)

        # Path Modeling (PM)
        if self.enable_modules['PM']:
            lstm_out, _ = self.path_lstm(x)  # (B*num_paths, T, hidden_dim*2)
            x = lstm_out
        # Else, x remains (B*num_paths, T, input_dim)

        # Time Pooling (TP): Pool over time to get (B*num_paths, 1, feature_dim)
        if self.enable_modules['TP']:
            x = x.permute(0, 2, 1)  # (B*num_paths, feature_dim, T) for pooling
            x = self.time_pooling(x)  # (B*num_paths, feature_dim, 1)
            x = x.squeeze(2)  # (B*num_paths, feature_dim)
        else:
            # If no TP, flatten time: (B*num_paths, T * feature_dim)
            x = x.view(x.size(0), -1)

        # Reshape back to (B, num_paths, feature_dim)
        feature_dim = x.size(1)
        x = x.view(B, self.num_paths, feature_dim)

        # Multi-Head Attention (MHA): Attention over paths (queries, keys, values are paths)
        if self.enable_modules['MHA']:
            attn_out, _ = self.attention(x, x, x)  # (B, num_paths, feature_dim)
            x = attn_out + x  # Residual connection
            x = self.dropout_layer(x)

        # Global Integration (GE): Flatten and classify
        x = x.view(B, -1)  # (B, num_paths * feature_dim)
        if self.enable_modules['GE']:
            x = F.relu(self.fc1(x))
            x = self.dropout_layer(x)
            x = self.fc2(x)  # (B, num_classes)
        else:
            x = self.fc2(x)

        return x

# If running directly, test the model with dummy input
if __name__ == "__main__":
    # Dummy config for testing
    dummy_config = {
        'data': {'partition_scheme': 'Schaefer-100', 'num_partitions': {'Schaefer-100': 100}, 'time_series_length': 200},
        'model': {
            'input_dim': 1,
            'hidden_dim': 64,
            'num_layers': 2,
            'num_heads': 4,
            'enable_modules': {'PM': True, 'GE': True, 'MHA': True, 'TP': True},
            'dropout': 0.1
        }
    }
    model = NeuroPathNet(dummy_config)
    dummy_input = torch.randn(4, 100**2, 200)  # B=4, num_paths=10000, T=200
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Should be (4, 2)
