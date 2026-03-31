"""Deterministic action indexing for AlphaZero policy heads."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from gym_splendor_code.envs.mechanics.action import Action


class StableActionIndexer:
    """Map action semantics to fixed policy-head indices.

    The mapping is collision-free inside a run by assigning each unique action key
    to a unique index in [0, policy_size).
    """

    def __init__(self, policy_size: int) -> None:
        self.policy_size = int(policy_size)
        if self.policy_size <= 0:
            raise ValueError("policy_size must be positive")
        self._key_to_index: dict[str, int] = {}
        self._next_index = 0

    def action_key(self, action: Action) -> str:
        payload: dict[str, Any] = {"action_type": action.action_type}

        # Trade action: full gem flow is the identity.
        if hasattr(action, "gems_from_board_to_player"):
            payload["gems_flow"] = tuple(int(x) for x in action.gems_from_board_to_player.to_dict())

        # Buy action: include card id and explicit gold-usage fields.
        if hasattr(action, "card") and getattr(action, "card") is not None:
            payload["card_id"] = int(action.card.id)
        if hasattr(action, "n_gold_gems_to_use"):
            payload["n_gold"] = int(action.n_gold_gems_to_use)
        if hasattr(action, "use_gold_as") and getattr(action, "use_gold_as") is not None:
            payload["use_gold_as"] = tuple(int(x) for x in action.use_gold_as.to_dict())

        # Reserve action specific fields.
        if hasattr(action, "take_golden_gem"):
            payload["take_gold"] = bool(action.take_golden_gem)
        if hasattr(action, "return_gem_color"):
            payload["return_gem_color"] = (
                int(action.return_gem_color.value)
                if action.return_gem_color is not None
                else None
            )

        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def action_index(self, action: Action) -> int:
        key = self.action_key(action)
        existing = self._key_to_index.get(key)
        if existing is not None:
            return existing

        if self._next_index >= self.policy_size:
            raise RuntimeError(
                f"Action vocabulary exhausted: policy_size={self.policy_size}, "
                f"seen_unique_actions={self._next_index}."
            )

        assigned = self._next_index
        self._key_to_index[key] = assigned
        self._next_index += 1
        return assigned

    def legal_indices(self, legal_actions: list[Action]) -> np.ndarray:
        return np.array([self.action_index(action) for action in legal_actions], dtype=np.int64)

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy_size": self.policy_size,
            "key_to_index": dict(self._key_to_index),
            "next_index": self._next_index,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if int(payload.get("policy_size", self.policy_size)) != self.policy_size:
            raise ValueError("Incompatible policy_size when loading action indexer state.")
        self._key_to_index = {str(k): int(v) for k, v in payload.get("key_to_index", {}).items()}
        self._next_index = int(payload.get("next_index", len(self._key_to_index)))
