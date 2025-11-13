"""Real BullFlag CNN multi-head model with proper 1D CNN backbone."""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BullFlagCNNMultiHead(nn.Module):
    """Proper 1D CNN backbone with 4 output heads for BullFlag pattern detection."""

    def __init__(self, in_channels: int, num_heads: int = 4, window_len: int = 200) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.window_len = window_len

        # Backbone: Conv1d layers with increasing channels
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Flatten and dense layers
        # After two pools of 2, length is reduced by factor of 4
        reduced_len = window_len // 4
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * reduced_len, 128)
        self.fc2 = nn.Linear(128, num_heads)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning probabilities for each head."""
        # x shape: [B, C, L]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)  # [B, num_heads]

        probs = torch.sigmoid(logits)
        # Return dict for compatibility with existing GradCAM and inference
        return {
            "flag": probs[:, 0],
            "breakout": probs[:, 1],
            "retest": probs[:, 2],
            "continuation": probs[:, 3],
        }

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits for training."""
        # x shape: [B, C, L]
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)  # [B, num_heads]
        return logits


def build_model_from_config(window_len: int, in_channels: int, num_heads: int = 4) -> BullFlagCNNMultiHead:
    """Factory function to build model with given config."""
    return BullFlagCNNMultiHead(
        in_channels=in_channels,
        num_heads=num_heads,
        window_len=window_len
    )
