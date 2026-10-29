"""Convolutional front end that downsamples, Transformer encoder over time, pooled linear head."""
import torch
from torch import nn


class CNNTransformer(nn.Module):
    def __init__(self, num_classes: int = 5, d_model: int = 128, num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.front = nn.Sequential(
            nn.Conv1d(24, 64, 15, stride=2, padding=7, bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, d_model, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(d_model), nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, 5, stride=2, padding=2, bias=False), nn.BatchNorm1d(d_model), nn.ReLU(inplace=True))
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, num_heads, d_model * 2, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_classes)

    def encode(self, signal, lead_mask):
        x = torch.cat((signal * lead_mask.unsqueeze(-1), lead_mask.unsqueeze(-1).expand_as(signal)), dim=1)
        z = self.front(x).transpose(1, 2)  # [B, T', D]
        z = self.norm(self.encoder(z + self.pos[:, : z.shape[1]]))
        return z.transpose(1, 2)  # [B, D, T']

    def forward(self, signal, lead_mask):
        features = self.encode(signal, lead_mask)
        return self.head(self.dropout(features.mean(dim=-1))), features
