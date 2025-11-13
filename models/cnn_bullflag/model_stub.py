"""Lightweight BullFlag CNN stub used for smoke tests and tooling demos.

NOTE: This is legacy/stub code. The real model is in model.py.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class BullFlagCNNMultiHeadStub(nn.Module):
    """A minimal multi-head CNN with a named conv4 block for Grad-CAM demos."""

    def __init__(self, in_channels: int = 12, base_channels: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(),
        )
        # Explicitly name conv4 so Grad-CAM can target it
        self.conv4 = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.heads = nn.ModuleDict(
            {
                "flag": nn.Linear(base_channels * 2, 1),
                "breakout": nn.Linear(base_channels * 2, 1),
                "retest": nn.Linear(base_channels * 2, 1),
                "continuation": nn.Linear(base_channels * 2, 1),
            }
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning sigmoid probabilities for each head."""
        out = self.conv1(x)
        out = F.max_pool1d(out, kernel_size=2)
        out = self.conv2(out)
        out = F.max_pool1d(out, kernel_size=2)
        out = self.conv3(out)
        out = self.conv4(out)
        pooled = self.global_pool(out).view(out.size(0), -1)
        logits = {name: head(pooled) for name, head in self.heads.items()}
        probs = {name: torch.sigmoid(logit) for name, logit in logits.items()}
        return probs
