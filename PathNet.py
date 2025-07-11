import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_path_trajectories(fmri_data, partitions, window_size, stride):
    """
    构建路径轨迹 Path_{i,j}。
    :param fmri_data: np.array, shape=(T_total, num_ROIs)
    :param partitions: list of list of int, each sublist is ROI indices for one partition
    :param window_size: int, sliding window length
    :param stride: int, step between windows
    :return: np.array, shape=(num_paths, num_windows)
    """
    from scipy.stats import pearsonr

    T_total, num_rois = fmri_data.shape
    N = len(partitions)
    paths = []
    num_windows = (T_total - window_size) // stride + 1

    for w in range(num_windows):
        start = w * stride
        end = start + window_size
        window_data = fmri_data[start:end]  # (window_size, num_ROIs)
        corr_matrix = np.corrcoef(window_data.T)  # (num_ROIs, num_ROIs)

        # Compute inter-partition connectivity
        path_values = []
        for i in range(N):
            for j in range(i + 1, N):
                roi_i = partitions[i]
                roi_j = partitions[j]
                sub_corr = corr_matrix[np.ix_(roi_i, roi_j)]
                avg_conn = np.nanmean(sub_corr)  # mean across ROI pairs
                path_values.append(avg_conn)
        paths.append(path_values)

    return np.array(paths).T  # shape: (num_paths, num_windows)
class PathwayEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(1, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

    def forward(self, path):  # (B, T, 1)
        x = self.linear(path)
        x = self.pos_encoding(x)
        return x  # (B, T, d_model)
class PathwayEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(1, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

    def forward(self, path):  # (B, T, 1)
        x = self.linear(path)
        x = self.pos_encoding(x)
        return x  # (B, T, d_model)
class SinglePathEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):  # (B, T, d_model)
        x = x.permute(1, 0, 2)  # (T, B, d)
        encoded = self.transformer(x)  # (T, B, d)
        encoded = encoded.permute(1, 2, 0)  # (B, d, T)
        pooled = self.temporal_pool(encoded).squeeze(-1)  # (B, d)
        return pooled
class CrossPathAttentionFusion(nn.Module):
    def __init__(self, d_model, n_heads, n_layers):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, path_reps):  # (B, L, d)
        x = path_reps.permute(1, 0, 2)  # (L, B, d)
        out = self.transformer(x)  # (L, B, d)
        return out.permute(1, 0, 2)  # (B, L, d)
class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.context = nn.Parameter(torch.randn(d_model))

    def forward(self, H):  # (B, L, d)
        score = torch.tanh(self.proj(H))  # (B, L, d)
        alpha = torch.matmul(score, self.context)  # (B, L)
        alpha = F.softmax(alpha, dim=1).unsqueeze(-1)
        pooled = (alpha * H).sum(dim=1)  # (B, d)
        return pooled
class FullBrainNetworkModel(nn.Module):
    def __init__(self, num_paths, time_steps, d_model=64, n_heads=4, n_layers=2, num_classes=2):
        super().__init__()
        self.embedding = PathwayEmbedding(d_model)
        self.path_encoder = SinglePathEncoder(d_model, n_heads, n_layers)
        self.cross_path_encoder = CrossPathAttentionFusion(d_model, n_heads, n_layers)
        self.attn_pooling = AttentionPooling(d_model)
        self.classifier = BrainClassifier(d_model, num_classes)

        self.num_paths = num_paths
        self.time_steps = time_steps

    def forward(self, x):  # x: (B, num_paths, T)
        B, L, T = x.shape
        x = x.unsqueeze(-1)  # (B, L, T, 1)
        x = x.view(B * L, T, 1)  # Flatten batch and path
        x = self.embedding(x)  # (B*L, T, d)
        path_reps = self.path_encoder(x)  # (B*L, d)
        path_reps = path_reps.view(B, L, -1)  # (B, L, d)
        fused = self.cross_path_encoder(path_reps)  # (B, L, d)
        pooled = self.attn_pooling(fused)  # (B, d)
        logits = self.classifier(pooled)  # (B, C)
        return logits
