import torch
import torch.nn as nn
import torch.nn.functional as F

class PathEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.pe = nn.Parameter(torch.randn(1, 100, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        b, t = x.size()
        x = x.view(b, t, 1)
        x = self.embedding(x) + self.pe[:, :t]
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = self.temporal_pool(x).squeeze(-1)
        return x

class CrossPathAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.att_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )

    def forward(self, H):
        H_prime = self.transformer(H)
        scores = self.att_pool(H_prime).squeeze(-1)
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)
        z = torch.sum(alpha * H_prime, dim=1)
        return z

class NeuroPathNet(nn.Module):
    def __init__(self, num_paths, d_model=64, n_heads=4, d_ff=128, num_classes=2):
        super().__init__()
        self.path_encoders = nn.ModuleList([PathEncoder(d_model, n_heads, d_ff) for _ in range(num_paths)])
        self.cross_path_attention = CrossPathAttention(d_model, n_heads, d_ff)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, X_paths):
        path_outputs = [encoder(path) for encoder, path in zip(self.path_encoders, X_paths)]
        H = torch.stack(path_outputs, dim=1)
        z = self.cross_path_attention(H)
        return self.classifier(z)
