"""
State Vectorizer for Splendor Score-based Agent

Converts Splendor game state into a fixed-size vector (159 dimensions) 
suitable for PPO policy network input.

Author: AI Agent
Date: 2026-02-24
Version: 1.0
"""

import numpy as np
from typing import Optional
import sys
sys.path.insert(0, "modules")

from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.enums import GemColor, Row
from gym_splendor_code.envs.mechanics.card import Card
from gym_splendor_code.envs.mechanics.noble import Noble


class SplendorStateVectorizer:
    """
    Converts Splendor game state to fixed-size vector representation.
    
    Vector Structure (135 dimensions):
        - Active player hand: 35 dims
        - Opponent hand: 14 dims
        - Board gems: 6 dims
        - Board cards: 72 dims (12 cards × 6 features)
        - Board nobles: 6 dims (3 nobles × 2 features)
        - Game progress: 2 dims
    """
    
    # Constants for normalization
    MAX_GEMS_ON_HAND = 10
    MAX_CARD_COUNT = 10
    MAX_VICTORY_POINTS = 20
    MAX_CARD_PRICE = 7
    MAX_CARD_VP = 5
    MAX_RESERVED = 3
    MAX_NOBLES = 3
    MAX_TURNS = 120
    MAX_CARDS_ON_BOARD = 12
    MAX_GEMS_ON_BOARD = 7  # Initial gems per color (excluding gold=5)
    
    # Gem colors (excluding GOLD for some features)
    GEM_COLORS = [GemColor.GOLD, GemColor.RED, GemColor.GREEN, 
                  GemColor.BLUE, GemColor.WHITE, GemColor.BLACK]
    DISCOUNT_COLORS = [GemColor.RED, GemColor.GREEN, GemColor.BLUE, 
                       GemColor.WHITE, GemColor.BLACK]
    
    # Card rows
    CARD_ROWS = [Row.CHEAP, Row.MEDIUM, Row.EXPENSIVE]
    
    def __init__(self):
        """Initialize the vectorizer."""
        # Updated vector size: 35 + 14 + 6 + 72 + 6 + 2 = 135
        self.vector_size = 135
        
    def vectorize(self, state: State, player_id: int, turn_count: int = 0) -> np.ndarray:
        """
        Convert game state to fixed-size vector.
        
        Args:
            state: Splendor State object
            player_id: ID of the player for whom to generate observation (0 or 1)
            turn_count: Current turn number (default 0)
            
        Returns:
            np.ndarray: Shape (135,) dtype float32, values in [0, 1]
        """
        vector_parts = []
        
        # Get active and opponent hands
        active_hand = state.list_of_players_hands[player_id]
        opponent_id = 1 - player_id
        opponent_hand = state.list_of_players_hands[opponent_id]
        
        # 1. Active player hand (35 dims)
        vector_parts.append(self._vectorize_active_player(active_hand, state))
        
        # 2. Opponent hand (14 dims)
        vector_parts.append(self._vectorize_opponent(opponent_hand))
        
        # 3. Board gems (6 dims)
        vector_parts.append(self._vectorize_gems(state.board.gems_on_board, 
                                                  self.MAX_GEMS_ON_BOARD))
        
        # 4. Board cards (96 dims)
        vector_parts.append(self._vectorize_board_cards(state.board.cards_on_board, 
                                                         active_hand))
        
        # 5. Board nobles (6 dims)
        vector_parts.append(self._vectorize_board_nobles(state.board.nobles_on_board, 
                                                          active_hand))
        
        # 6. Game progress (2 dims)
        vector_parts.append(self._vectorize_game_progress(player_id, turn_count))
        
        # Concatenate all parts
        full_vector = np.concatenate(vector_parts)
        
        # Validate and return
        assert full_vector.shape == (self.vector_size,), \
            f"Expected shape ({self.vector_size},), got {full_vector.shape}"
        assert not np.any(np.isnan(full_vector)), "Vector contains NaN values"
        assert not np.any(np.isinf(full_vector)), "Vector contains Inf values"
        
        # Clip to [0, 1] and convert to float32
        full_vector = np.clip(full_vector, 0.0, 1.0).astype(np.float32)
        
        return full_vector
    
    def _vectorize_active_player(self, hand, state) -> np.ndarray:
        """
        Vectorize active player's hand (35 dimensions).
        
        Components:
            - Gems possessed: 6 dims
            - Card discounts: 5 dims
            - Reserved cards count: 1 dim
            - Victory points: 1 dim
            - Nobles possessed: 1 dim
            - Can afford count: 1 dim
            - Reserved cards features: 20 dims (3 cards × ~7 features, padded)
        """
        parts = []
        
        # Gems possessed (6 dims)
        parts.append(self._vectorize_gems(hand.gems_possessed, self.MAX_GEMS_ON_HAND))
        
        # Card discounts (5 dims) - count of each discount color
        discount = hand.discount()
        discount_vec = np.array([
            discount.value(color) / self.MAX_CARD_COUNT 
            for color in self.DISCOUNT_COLORS
        ])
        parts.append(discount_vec)
        
        # Reserved cards count (1 dim)
        parts.append(np.array([len(hand.cards_reserved) / self.MAX_RESERVED]))
        
        # Victory points (1 dim)
        parts.append(np.array([hand.number_of_my_points() / self.MAX_VICTORY_POINTS]))
        
        # Nobles possessed (1 dim)
        parts.append(np.array([len(hand.nobles_possessed) / self.MAX_NOBLES]))
        
        # Can afford count (1 dim) - how many cards on board can be afforded
        can_afford = sum(1 for card in state.board.cards_on_board 
                        if hand.can_afford_card(card))
        parts.append(np.array([can_afford / self.MAX_CARDS_ON_BOARD]))
        
        # Reserved cards features (20 dims = 3 cards × ~7 features, padded)
        reserved_vec = self._vectorize_reserved_cards(hand.cards_reserved)
        parts.append(reserved_vec)
        
        result = np.concatenate(parts)
        assert result.shape == (35,), f"Expected 35 dims, got {result.shape}"
        return result
    
    def _vectorize_opponent(self, hand) -> np.ndarray:
        """
        Vectorize opponent's hand (14 dimensions).
        
        Components:
            - Gems possessed: 6 dims
            - Card discounts: 5 dims
            - Victory points: 1 dim
            - Nobles possessed: 1 dim
            - Reserved cards count: 1 dim
        """
        parts = []
        
        # Gems possessed (6 dims)
        parts.append(self._vectorize_gems(hand.gems_possessed, self.MAX_GEMS_ON_HAND))
        
        # Card discounts (5 dims)
        discount = hand.discount()
        discount_vec = np.array([
            discount.value(color) / self.MAX_CARD_COUNT 
            for color in self.DISCOUNT_COLORS
        ])
        parts.append(discount_vec)
        
        # Victory points (1 dim)
        parts.append(np.array([hand.number_of_my_points() / self.MAX_VICTORY_POINTS]))
        
        # Nobles possessed (1 dim)
        parts.append(np.array([len(hand.nobles_possessed) / self.MAX_NOBLES]))
        
        # Reserved cards count (1 dim)
        parts.append(np.array([len(hand.cards_reserved) / self.MAX_RESERVED]))
        
        result = np.concatenate(parts)
        assert result.shape == (14,), f"Expected 14 dims, got {result.shape}"
        return result
    
    def _vectorize_gems(self, gems_collection, max_value: int) -> np.ndarray:
        """
        Vectorize gems collection (6 dimensions).
        
        Args:
            gems_collection: GemsCollection object
            max_value: Maximum value for normalization
            
        Returns:
            np.ndarray: Shape (6,) with normalized gem counts
        """
        return np.array([
            gems_collection.value(color) / max_value 
            for color in self.GEM_COLORS
        ])
    
    def _vectorize_board_cards(self, cards_on_board, active_hand) -> np.ndarray:
        """
        Vectorize cards on board (72 dimensions = 12 cards × 6 features).
        
        For each card (simplified encoding):
            - Row: 1 dim (0=cheap, 0.5=medium, 1=expensive)
            - Discount profit: 1 dim (encoded as 0.2/0.4/0.6/0.8/1.0 for 5 colors)
            - Victory points: 1 dim (normalized by 5)
            - Price total: 1 dim (sum of all gem costs / 20)
            - Can afford: 1 dim (binary)
            - Padding: 1 dim (reserved for future use)
        
        Cards ordered by row: [CHEAP×4, MEDIUM×4, EXPENSIVE×4]
        Empty positions filled with zeros.
        """
        # Sort cards by row
        cards_by_row = {row: [] for row in self.CARD_ROWS}
        for card in cards_on_board:
            cards_by_row[card.row].append(card)
        
        # Sort within each row by ID for determinism
        for row in self.CARD_ROWS:
            cards_by_row[row] = sorted(cards_by_row[row], key=lambda c: c.id)
        
        # Create card vectors (exactly 12 positions: 4 per row)
        all_card_vectors = []
        for row in self.CARD_ROWS:
            row_cards = cards_by_row[row][:4]  # Take up to 4 cards
            for i in range(4):
                if i < len(row_cards):
                    card_vec = self._vectorize_single_card_simplified(row_cards[i], active_hand)
                    all_card_vectors.append(card_vec)
                else:
                    all_card_vectors.append(np.zeros(6))  # Empty position
        
        # Concatenate all card vectors
        result = np.concatenate(all_card_vectors)
        assert result.shape == (72,), \
            f"Expected 72 dims, got {result.shape}. Card vectors: {len(all_card_vectors)}"
        return result
    
    def _vectorize_single_card_simplified(self, card: Card, active_hand) -> np.ndarray:
        """
        Vectorize a single card with simplified encoding (6 dimensions).
        
        Returns:
            - Row: 1 dim (0.33=cheap, 0.67=medium, 1.0=expensive)
            - Discount profit: 1 dim (0.2/0.4/0.6/0.8/1.0 for 5 colors)
            - Victory points: 1 dim (normalized by 5)
            - Price total: 1 dim (sum of all costs / 20)
            - Can afford: 1 dim (binary)
            - Padding: 1 dim (0 for now)
        """
        # Row encoding (0.33, 0.67, 1.0)
        row_val = (self.CARD_ROWS.index(card.row) + 1) / 3.0
        
        # Discount profit encoding (0.2, 0.4, 0.6, 0.8, 1.0)
        if card.discount_profit in self.DISCOUNT_COLORS:
            discount_val = (self.DISCOUNT_COLORS.index(card.discount_profit) + 1) / 5.0
        else:
            discount_val = 0.0
        
        # Victory points
        vp_val = card.victory_points / self.MAX_CARD_VP
        
        # Total price
        price_sum = sum(card.price.value(c) for c in self.DISCOUNT_COLORS)
        price_val = price_sum / 20.0  # Max total cost is around 20
        
        # Can afford
        can_afford_val = 1.0 if active_hand.can_afford_card(card) else 0.0
        
        # Padding
        padding_val = 0.0
        
        return np.array([row_val, discount_val, vp_val, price_val, can_afford_val, padding_val])
    
    def _vectorize_single_card(self, card: Card) -> np.ndarray:
        """
        Vectorize a single card (14 dimensions) - DEPRECATED, use _vectorize_single_card_simplified.
        
        Returns:
            - Row indicator: 3 dims (one-hot)
            - Discount profit: 5 dims (one-hot)
            - Victory points: 1 dim
            - Price: 5 dims (excluding gold)
        """
        # Row one-hot (3 dims)
        row_vec = np.zeros(3)
        row_index = self.CARD_ROWS.index(card.row)
        row_vec[row_index] = 1.0
        
        # Discount profit one-hot (5 dims)
        discount_vec = np.zeros(5)
        if card.discount_profit in self.DISCOUNT_COLORS:
            discount_index = self.DISCOUNT_COLORS.index(card.discount_profit)
            discount_vec[discount_index] = 1.0
        
        # Victory points (1 dim)
        vp_vec = np.array([card.victory_points / self.MAX_CARD_VP])
        
        # Price (5 dims, excluding gold)
        price_vec = np.array([
            card.price.value(color) / self.MAX_CARD_PRICE 
            for color in self.DISCOUNT_COLORS
        ])
        
        return np.concatenate([row_vec, discount_vec, vp_vec, price_vec])
    
    def _vectorize_reserved_cards(self, reserved_cards) -> np.ndarray:
        """
        Vectorize reserved cards (20 dimensions = placeholder for 3 cards).
        
        Simplified encoding: For each of 3 reserved card slots:
            - Row: 1 dim (0=none, 0.33=cheap, 0.67=medium, 1.0=expensive)
            - Discount: 1 dim (0=none, 0.2/0.4/0.6/0.8/1.0 for 5 colors)
            - VP: 1 dim
            - Price sum: 1 dim (total price normalized)
            - Can afford: 1 dim
            - Padding: 2 dims (for future use)
        
        Total: 3 cards × 7 features = 21 dims → reduce to 20 by removing last padding
        """
        reserved_list = sorted(list(reserved_cards), key=lambda c: c.id)[:3]
        
        vectors = []
        for i in range(3):
            if i < len(reserved_list):
                card = reserved_list[i]
                # Row encoding (0.33, 0.67, 1.0)
                row_val = (self.CARD_ROWS.index(card.row) + 1) / 3.0
                
                # Discount encoding (0.2, 0.4, 0.6, 0.8, 1.0)
                if card.discount_profit in self.DISCOUNT_COLORS:
                    discount_val = (self.DISCOUNT_COLORS.index(card.discount_profit) + 1) / 5.0
                else:
                    discount_val = 0.0
                
                # VP
                vp_val = card.victory_points / self.MAX_CARD_VP
                
                # Total price
                price_sum = sum(card.price.value(c) for c in self.DISCOUNT_COLORS)
                price_val = price_sum / (self.MAX_CARD_PRICE * 5)  # Max total ~35
                
                # Can afford (placeholder - would need hand reference)
                can_afford_val = 0.0  # Simplified for now
                
                # Padding
                card_vec = np.array([row_val, discount_val, vp_val, price_val, 
                                    can_afford_val, 0.0, 0.0])
            else:
                card_vec = np.zeros(7)
            
            vectors.append(card_vec)
        
        result = np.concatenate(vectors)[:20]  # Take first 20 dims
        return result
    
    def _vectorize_board_nobles(self, nobles_on_board, active_hand) -> np.ndarray:
        """
        Vectorize nobles on board (6 dimensions = 3 nobles × 2 features).
        
        For each noble:
            - Requirement gems count: 1 dim (total cards needed)
            - Can obtain now: 1 dim (binary)
        
        Empty positions filled with zeros.
        """
        nobles_list = sorted(list(nobles_on_board), key=lambda n: n.id)[:3]
        
        vectors = []
        for i in range(3):
            if i < len(nobles_list):
                noble = nobles_list[i]
                
                # Total requirement (sum of all discount requirements)
                # Note: Noble uses 'price' attribute for minimum_possessions
                total_req = sum(noble.price.value(c) for c in self.DISCOUNT_COLORS)
                req_normalized = total_req / 10.0  # Max requirement ~10
                
                # Can obtain (check if player has enough discounts)
                player_discount = active_hand.discount()
                can_obtain = 1.0 if all(
                    player_discount.value(color) >= noble.price.value(color)
                    for color in self.DISCOUNT_COLORS
                ) else 0.0
                
                noble_vec = np.array([req_normalized, can_obtain])
            else:
                noble_vec = np.zeros(2)
            
            vectors.append(noble_vec)
        
        result = np.concatenate(vectors)
        assert result.shape == (6,), f"Expected 6 dims, got {result.shape}"
        return result
    
    def _vectorize_game_progress(self, player_id: int, turn_count: int) -> np.ndarray:
        """
        Vectorize game progress (2 dimensions).
        
        Returns:
            - Turn progress: 1 dim (normalized)
            - Is active player: 1 dim (always 1 for this player's perspective)
        """
        turn_progress = min(turn_count / self.MAX_TURNS, 1.0)
        is_active = 1.0  # Always from active player's perspective
        
        return np.array([turn_progress, is_active])


# Utility function for easy import
def vectorize_state(state: State, player_id: int, turn_count: int = 0) -> np.ndarray:
    """
    Convenience function to vectorize a state.
    
    Args:
        state: Splendor State object
        player_id: ID of the player (0 or 1)
        turn_count: Current turn number
        
    Returns:
        np.ndarray: Shape (159,) dtype float32
    """
    vectorizer = SplendorStateVectorizer()
    return vectorizer.vectorize(state, player_id, turn_count)
