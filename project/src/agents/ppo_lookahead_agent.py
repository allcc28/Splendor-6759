"""PPO-guided lookahead agent for Splendor.

Combines a trained PPO policy with forward model search and event-based
evaluation. The PPO provides action probabilities (strategy intuition),
the forward model provides 1-2 step lookahead, and event-value functions
provide domain-knowledge scoring.

Inspired by Rinascimento (Bravi & Lucas, IEEE CoG 2020): shallow forward
search + event-value functions outperform deep MCTS in Splendor.
"""

from __future__ import annotations

from math import inf
from typing import List, Optional

import numpy as np
import torch

from sb3_contrib import MaskablePPO

from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.action_space_generator import (
    generate_all_legal_buys,
    generate_all_legal_reservations,
    generate_all_legal_trades,
)
from gym_splendor_code.envs.mechanics.state import State
from planning.adapter import SplendorPlanningAdapter
from reward.event_based_reward import DEFAULT_EVENT_WEIGHTS, compute_event_reward
from utils.event_detector import capture_state_snapshot, detect_events
from utils.state_vectorizer import SplendorStateVectorizer


class PPOLookaheadAgent:
    """PPO policy + forward model lookahead + event-based evaluation.

    Decision flow per turn:
    1. PPO gives action probabilities for all legal actions
    2. Top-K candidates selected by PPO probability
    3. Each candidate evaluated via forward simulation:
       - Event reward from executing the action
       - PPO value estimate of the resulting state
    4. Final score = α*ppo_prob + β*event_reward + γ*future_value
    """

    def __init__(
        self,
        ppo_model_path: str,
        top_k: int = 15,
        search_depth: int = 1,
        alpha: float = 0.3,
        beta: float = 0.5,
        gamma: float = 0.2,
        event_weights: Optional[np.ndarray] = None,
        device: str = "cpu",
    ) -> None:
        self.model = MaskablePPO.load(ppo_model_path, device=device)
        self.model.policy.set_training_mode(False)
        self.vectorizer = SplendorStateVectorizer()
        self.top_k = top_k
        self.search_depth = search_depth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.event_weights = event_weights if event_weights is not None else DEFAULT_EVENT_WEIGHTS.copy()
        self.device = device
        self.name = "PPOLookahead"

        # Observation dim PPO expects (may be 204 with event features).
        self._obs_dim = self.model.observation_space.shape[0]
        # Track whether we need gem gaps (60-dim) and last event (9-dim).
        self._include_gem_gaps = self._obs_dim > 135
        self._include_last_event = self._obs_dim > 195
        self._last_event = np.zeros(9, dtype=np.float32)

    def choose_action(self, observation, previous_actions) -> Optional[Action]:
        """Arena-compatible interface: pick best action from observation."""
        state = observation.recreate_state()
        return self.choose_action_from_state(state)

    def choose_action_from_state(self, state: State) -> Optional[Action]:
        """Pick best action given a raw State object."""
        legal_actions = self._get_legal_actions(state)
        if not legal_actions:
            return None

        # Depth 0: pure PPO (no search).
        if self.search_depth == 0:
            return self._ppo_greedy(state, legal_actions)

        player_id = state.active_player_id

        # Get PPO probabilities over legal actions.
        ppo_probs = self._get_ppo_probs(state, legal_actions, player_id)

        # Select top-K candidates.
        k = min(self.top_k, len(legal_actions))
        top_indices = np.argsort(ppo_probs)[-k:]

        # Evaluate each candidate with forward search.
        best_score = -inf
        best_action = legal_actions[0]
        best_events = None

        adapter = SplendorPlanningAdapter(state=state)

        for idx in top_indices:
            action = legal_actions[idx]
            score, events = self._evaluate_action(
                adapter, action, ppo_probs[idx], player_id
            )
            if score > best_score:
                best_score = score
                best_action = action
                best_events = events

        # Update last_event for next observation (keeps PPO obs consistent).
        if best_events is not None:
            self._last_event = best_events.astype(np.float32)

        return best_action

    def finish_game(self) -> None:
        """Arena compatibility."""
        return None

    # ------------------------------------------------------------------
    # PPO interface
    # ------------------------------------------------------------------

    def _get_ppo_probs(
        self, state: State, legal_actions: List[Action], player_id: int
    ) -> np.ndarray:
        """Get PPO's probability distribution over legal actions."""
        obs = self._vectorize(state, player_id)
        obs_t = torch.as_tensor(obs).unsqueeze(0).float()

        policy = self.model.policy
        with torch.no_grad():
            features = policy.extract_features(obs_t, policy.pi_features_extractor)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            dist = policy.action_dist.proba_distribution(
                policy.action_net(latent_pi)
            )
            all_probs = dist.distribution.probs[0].cpu().numpy()

        # Map probabilities to legal actions (indices 0..len-1).
        n = len(legal_actions)
        probs = all_probs[:n].copy()
        probs = np.maximum(probs, 0.0)
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(n, dtype=np.float32) / n
        return probs

    def _get_ppo_value(self, state: State, player_id: int) -> float:
        """Get PPO's value estimate for a state."""
        obs = self._vectorize(state, player_id)
        obs_t = torch.as_tensor(obs).unsqueeze(0).float()

        policy = self.model.policy
        with torch.no_grad():
            features = policy.extract_features(obs_t, policy.vf_features_extractor)
            latent_vf = policy.mlp_extractor.forward_critic(features)
            value = policy.value_net(latent_vf)
        return float(value.item())

    def _ppo_greedy(self, state: State, legal_actions: List[Action]) -> Action:
        """Pure PPO greedy action (no search)."""
        probs = self._get_ppo_probs(state, legal_actions, state.active_player_id)
        return legal_actions[int(np.argmax(probs))]

    # ------------------------------------------------------------------
    # Forward search
    # ------------------------------------------------------------------

    def _evaluate_action(
        self,
        adapter: SplendorPlanningAdapter,
        action: Action,
        ppo_prob: float,
        player_id: int,
    ) -> tuple[float, np.ndarray]:
        """Evaluate one candidate action via forward simulation.

        Returns (score, event_vector).
        """
        # Capture pre-state snapshot for event detection.
        prev_snap = capture_state_snapshot(adapter.state, player_id)

        # Clone and execute.
        cloned = adapter.clone()
        cloned.step(action)

        # Event detection.
        next_snap = capture_state_snapshot(cloned.state, player_id)
        events = detect_events(prev_snap, action, next_snap)

        # Immediate terminal: winning move gets huge bonus.
        if cloned.is_terminal():
            return 100.0 * cloned.terminal_value(player_id), events

        # Event reward from this action.
        event_reward = compute_event_reward(events, self.event_weights)

        # PPO value of resulting state.
        future_value = self._get_ppo_value(cloned.state, player_id)

        # Depth-2: consider opponent's best response.
        if self.search_depth >= 2:
            future_value = self._opponent_adjusted_value(cloned, player_id)

        # Normalize future_value to roughly [-1, 1] range.
        # PPO values can be large; scale down.
        future_value_norm = np.tanh(future_value / 50.0)

        score = (
            self.alpha * ppo_prob
            + self.beta * event_reward
            + self.gamma * future_value_norm
        )
        return score, events

    def _opponent_adjusted_value(
        self, adapter: SplendorPlanningAdapter, our_player_id: int
    ) -> float:
        """Evaluate state after opponent's best response (minimax 1-ply).

        Assumes opponent plays the PPO-greedy action.
        """
        if adapter.is_terminal():
            return adapter.terminal_value(our_player_id) * 50.0

        opp_id = adapter.current_player
        if opp_id == our_player_id:
            # It's still our turn (shouldn't happen in 2-player), return value.
            return self._get_ppo_value(adapter.state, our_player_id)

        # Get opponent's legal actions and pick PPO-greedy for them.
        opp_legal = adapter.legal_actions
        if not opp_legal:
            return self._get_ppo_value(adapter.state, our_player_id)

        opp_probs = self._get_ppo_probs(adapter.state, opp_legal, opp_id)
        opp_action = opp_legal[int(np.argmax(opp_probs))]

        # Execute opponent's move.
        after_opp = adapter.clone()
        after_opp.step(opp_action)

        return self._get_ppo_value(after_opp.state, our_player_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _vectorize(self, state: State, player_id: int = 0) -> np.ndarray:
        """Vectorize state for PPO, matching the event-wrapped observation format.

        PPO was trained with 204-dim obs: 135 (base) + 60 (gem gaps) + 9 (last event).
        """
        parts = [self.vectorizer.vectorize(state, player_id=player_id)]
        if self._include_gem_gaps:
            gem_gaps = self.vectorizer.get_gem_gap_features(state, player_id)
            parts.append(gem_gaps)
        if self._include_last_event:
            parts.append(self._last_event.astype(np.float32))
        vec = np.concatenate(parts)
        # Safety: pad or truncate to exact expected dim.
        if len(vec) < self._obs_dim:
            vec = np.concatenate([vec, np.zeros(self._obs_dim - len(vec), dtype=np.float32)])
        return vec[:self._obs_dim]

    @staticmethod
    def _get_legal_actions(state: State) -> List[Action]:
        trades = generate_all_legal_trades(state)
        buys = generate_all_legal_buys(state)
        reserves = generate_all_legal_reservations(state)
        return trades + buys + reserves
