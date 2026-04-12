"""Tests for MuZero network output shapes and inference consistency."""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from muzero.network import (
    MuZeroNetwork,
    RepresentationNetwork,
    DynamicsNetwork,
    PredictionNetwork,
)


@pytest.fixture
def network():
    return MuZeroNetwork(
        input_channels=80,
        latent_channels=32,
        policy_size=256,
        action_space_size=256,
        action_embed_dim=8,
        repr_res_blocks=1,
        dyn_res_blocks=1,
    )


class TestRepresentationNetwork:
    def test_output_shape(self):
        net = RepresentationNetwork(input_channels=80, latent_channels=32, num_res_blocks=1)
        obs = torch.randn(2, 80, 3, 4)
        out = net(obs)
        assert out.shape == (2, 32, 3, 4)


class TestDynamicsNetwork:
    def test_output_shapes(self):
        net = DynamicsNetwork(latent_channels=32, action_space_size=256, action_embed_dim=8, num_res_blocks=1)
        state = torch.randn(2, 32, 3, 4)
        action = torch.tensor([0, 1], dtype=torch.long)
        next_state, reward = net(state, action)
        assert next_state.shape == (2, 32, 3, 4)
        assert reward.shape == (2,)


class TestPredictionNetwork:
    def test_output_shapes(self):
        net = PredictionNetwork(latent_channels=32, policy_size=256)
        state = torch.randn(2, 32, 3, 4)
        policy, value = net(state)
        assert policy.shape == (2, 256)
        assert value.shape == (2,)

    def test_value_range(self):
        net = PredictionNetwork(latent_channels=32, policy_size=256)
        state = torch.randn(4, 32, 3, 4)
        _, value = net(state)
        # Value head uses Tanh, so should be in [-1, 1].
        assert torch.all(value >= -1.0)
        assert torch.all(value <= 1.0)


class TestMuZeroNetwork:
    def test_initial_inference(self, network):
        obs = torch.randn(2, 80, 3, 4)
        output = network.initial_inference(obs)
        assert output.latent_state.shape == (2, 32, 3, 4)
        assert output.policy_logits.shape == (2, 256)
        assert output.value.shape == (2,)
        assert output.reward.shape == (2,)

    def test_recurrent_inference(self, network):
        obs = torch.randn(2, 80, 3, 4)
        initial = network.initial_inference(obs)

        action = torch.tensor([5, 10], dtype=torch.long)
        output = network.recurrent_inference(initial.latent_state, action)
        assert output.latent_state.shape == (2, 32, 3, 4)
        assert output.policy_logits.shape == (2, 256)
        assert output.value.shape == (2,)
        assert output.reward.shape == (2,)

    def test_recurrent_shape_consistency(self, network):
        """Multiple recurrent steps should maintain shape."""
        obs = torch.randn(1, 80, 3, 4)
        output = network.initial_inference(obs)
        state = output.latent_state

        for _ in range(5):
            action = torch.tensor([0], dtype=torch.long)
            output = network.recurrent_inference(state, action)
            state = output.latent_state
            assert state.shape == (1, 32, 3, 4)
