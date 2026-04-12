"""MuZero neural networks: representation, dynamics, and prediction.

Three-network architecture following the canonical MuZero design.
Reference: werner-duvaud/muzero-general models.py AbstractNetwork + MuZeroResidualNetwork.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MuZeroNetworkOutput:
    """Output container for network inference."""

    latent_state: torch.Tensor
    policy_logits: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor


class ResidualBlock(nn.Module):
    """Standard residual block with BatchNorm."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class RepresentationNetwork(nn.Module):
    """Maps raw observation tensor to latent state.

    Input: (batch, 80, 3, 4) observation tensor.
    Output: (batch, latent_channels, 3, 4) latent state.
    """

    def __init__(
        self,
        input_channels: int = 80,
        latent_channels: int = 64,
        num_res_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.conv_stem = nn.Conv2d(input_channels, latent_channels, 3, padding=1, bias=False)
        self.bn_stem = nn.BatchNorm2d(latent_channels)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(latent_channels) for _ in range(num_res_blocks)]
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn_stem(self.conv_stem(observation)))
        return self.res_blocks(x)


class DynamicsNetwork(nn.Module):
    """Predicts next latent state and immediate reward from current state + action.

    Input: latent state (batch, latent_channels, 3, 4) + action index (batch,).
    Output: next latent state (batch, latent_channels, 3, 4), reward (batch,).
    """

    def __init__(
        self,
        latent_channels: int = 64,
        action_space_size: int = 2048,
        action_embed_dim: int = 16,
        num_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(action_space_size, action_embed_dim)
        # Project action embedding to spatial dimensions and concatenate with state.
        self.action_proj = nn.Linear(action_embed_dim, 3 * 4)

        input_channels = latent_channels + 1  # state channels + 1 action plane
        self.conv_stem = nn.Conv2d(input_channels, latent_channels, 3, padding=1, bias=False)
        self.bn_stem = nn.BatchNorm2d(latent_channels)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(latent_channels) for _ in range(num_res_blocks)]
        )

        # Reward prediction head.
        self.reward_conv = nn.Conv2d(latent_channels, 1, 1)
        self.reward_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self, latent_state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = latent_state.size(0)

        # Create action plane: embed action → project to spatial → reshape.
        action_embed = self.action_embedding(action)  # (batch, embed_dim)
        action_plane = self.action_proj(action_embed)  # (batch, 12)
        action_plane = action_plane.view(batch_size, 1, 3, 4)  # (batch, 1, 3, 4)

        # Concatenate state and action plane.
        x = torch.cat([latent_state, action_plane], dim=1)
        x = F.relu(self.bn_stem(self.conv_stem(x)))
        next_state = self.res_blocks(x)

        # Reward prediction.
        reward = self.reward_fc(self.reward_conv(next_state))
        reward = reward.squeeze(-1)

        return next_state, reward


class PredictionNetwork(nn.Module):
    """Predicts policy logits and value from latent state.

    Input: latent state (batch, latent_channels, 3, 4).
    Output: policy logits (batch, policy_size), value (batch,).
    """

    def __init__(
        self,
        latent_channels: int = 64,
        policy_size: int = 2048,
    ) -> None:
        super().__init__()
        # Policy head.
        self.policy_conv = nn.Conv2d(latent_channels, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 3 * 4, policy_size),
        )

        # Value head.
        self.value_conv = nn.Conv2d(latent_channels, 16, 1)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 3 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, latent_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Policy.
        p = F.relu(self.policy_bn(self.policy_conv(latent_state)))
        policy_logits = self.policy_fc(p)

        # Value.
        v = F.relu(self.value_bn(self.value_conv(latent_state)))
        value = self.value_fc(v).squeeze(-1)

        return policy_logits, value


class MuZeroNetwork(nn.Module):
    """Combined MuZero network with canonical initial/recurrent inference API.

    Reference: werner-duvaud/muzero-general models.py AbstractNetwork.
    """

    def __init__(
        self,
        input_channels: int = 80,
        latent_channels: int = 64,
        policy_size: int = 2048,
        action_space_size: int = 2048,
        action_embed_dim: int = 16,
        repr_res_blocks: int = 3,
        dyn_res_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.representation = RepresentationNetwork(
            input_channels=input_channels,
            latent_channels=latent_channels,
            num_res_blocks=repr_res_blocks,
        )
        self.dynamics = DynamicsNetwork(
            latent_channels=latent_channels,
            action_space_size=action_space_size,
            action_embed_dim=action_embed_dim,
            num_res_blocks=dyn_res_blocks,
        )
        self.prediction = PredictionNetwork(
            latent_channels=latent_channels,
            policy_size=policy_size,
        )

    def initial_inference(
        self, observation: torch.Tensor
    ) -> MuZeroNetworkOutput:
        """Encode observation → latent state, predict policy + value."""
        latent_state = self.representation(observation)
        policy_logits, value = self.prediction(latent_state)
        # No reward at the root.
        reward = torch.zeros_like(value)
        return MuZeroNetworkOutput(
            latent_state=latent_state,
            policy_logits=policy_logits,
            value=value,
            reward=reward,
        )

    def recurrent_inference(
        self, latent_state: torch.Tensor, action: torch.Tensor
    ) -> MuZeroNetworkOutput:
        """Step dynamics → next state + reward, predict policy + value."""
        next_state, reward = self.dynamics(latent_state, action)
        policy_logits, value = self.prediction(next_state)
        return MuZeroNetworkOutput(
            latent_state=next_state,
            policy_logits=policy_logits,
            value=value,
            reward=reward,
        )
