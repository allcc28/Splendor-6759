"""Policy-value network for AlphaZero-style Splendor training."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Small residual block for the compact 3x4 board tensor."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + residual)
        return out


class SplendorPolicyValueNet(nn.Module):
    """Dual-head network producing policy logits and scalar value."""

    def __init__(
        self,
        input_channels: int,
        policy_size: int = 200,
        trunk_channels: int = 128,
        num_res_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.policy_size = policy_size

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, trunk_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(trunk_channels),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.Sequential(*[ResidualBlock(trunk_channels) for _ in range(num_res_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(trunk_channels, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 3 * 4, policy_size),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(trunk_channels, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(16 * 3 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.res_blocks(self.stem(x))
        policy_logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return policy_logits, value
