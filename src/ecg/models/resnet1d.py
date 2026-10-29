"""Lead-mask aware ResNet1D baseline."""
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 7, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm1d(out_channels))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x))))) + self.skip(x))


class ResNet1D(nn.Module):
    def __init__(self, num_classes: int = 5, channels: tuple[int, ...] = (64, 128, 256, 512),
                 blocks_per_stage: int = 2, dropout: float = 0.0):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(24, channels[0], 15, stride=2, padding=7, bias=False),
                                  nn.BatchNorm1d(channels[0]), nn.ReLU(inplace=True), nn.MaxPool1d(3, 2, 1))
        stages, current = [], channels[0]
        for i, width in enumerate(channels):
            blocks = [ResidualBlock(current, width, 1 if i == 0 else 2)]
            blocks += [ResidualBlock(width, width) for _ in range(blocks_per_stage - 1)]
            stages.append(nn.Sequential(*blocks)); current = width
        self.stages = nn.Sequential(*stages)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(channels[-1], num_classes)

    def encode(self, signal: torch.Tensor, lead_mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat((signal * lead_mask.unsqueeze(-1), lead_mask.unsqueeze(-1).expand_as(signal)), dim=1)
        return self.stages(self.stem(x))

    def forward(self, signal: torch.Tensor, lead_mask: torch.Tensor):
        features = self.encode(signal, lead_mask)
        logits = self.head(self.dropout(features.mean(dim=-1)))
        return logits, features
