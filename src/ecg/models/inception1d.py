"""InceptionTime-style 1D CNN sharing the (signal, lead_mask) -> (logits, features) interface."""
import torch
from torch import nn


class InceptionModule(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 32, kernels=(9, 19, 39), bottleneck: int = 32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_ch, bottleneck, 1, bias=False) if in_ch > 1 else nn.Identity()
        b = bottleneck if in_ch > 1 else in_ch
        self.convs = nn.ModuleList(nn.Conv1d(b, out_ch, k, padding=k // 2, bias=False) for k in kernels)
        self.pool_conv = nn.Sequential(nn.MaxPool1d(3, 1, 1), nn.Conv1d(in_ch, out_ch, 1, bias=False))
        self.bn = nn.BatchNorm1d(out_ch * (len(kernels) + 1))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        z = self.bottleneck(x)
        out = torch.cat([c(z) for c in self.convs] + [self.pool_conv(x)], dim=1)
        return self.act(self.bn(out))


class InceptionBlock(nn.Module):
    def __init__(self, in_ch: int, width: int, depth: int = 3):
        super().__init__()
        mods, c = [], in_ch
        for _ in range(depth):
            mods.append(InceptionModule(c, width)); c = width * 4
        self.body = nn.Sequential(*mods)
        self.skip = nn.Sequential(nn.Conv1d(in_ch, c, 1, bias=False), nn.BatchNorm1d(c))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))


class InceptionTime1D(nn.Module):
    def __init__(self, num_classes: int = 5, width: int = 32, blocks: int = 2, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Sequential(nn.AvgPool1d(2))  # halves T to bound compute
        layers, c = [], 24
        for _ in range(blocks):
            layers.append(InceptionBlock(c, width)); c = width * 4
        self.blocks = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(c, num_classes)

    def encode(self, signal, lead_mask):
        x = torch.cat((signal * lead_mask.unsqueeze(-1), lead_mask.unsqueeze(-1).expand_as(signal)), dim=1)
        return self.blocks(self.down(x))

    def forward(self, signal, lead_mask):
        features = self.encode(signal, lead_mask)
        return self.head(self.dropout(features.mean(dim=-1))), features
