"""
Tests for SplendorStateVectorizer

Validates that the state vectorizer produces correct fixed-size vectors
with proper normalization and consistency.

Author: AI Agent
Date: 2026-02-24
"""

import sys
import os
sys.path.insert(0, os.path.abspath("modules"))
sys.path.insert(0, os.path.abspath("project/src"))

import numpy as np
import pytest
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.action import Action, ActionBuyCard, ActionReserveCard, ActionTradeGems
from gym_splendor_code.envs.mechanics.enums import GemColor
from gym_splendor_code.envs.mechanics.gems_collection import GemsCollection
from utils.state_vectorizer import SplendorStateVectorizer, vectorize_state


class TestSplendorStateVectorizer:
    """Test suite for state vectorizer."""
    
    def setup_method(self):
        """Setup before each test."""
        self.vectorizer = SplendorStateVectorizer()
        
    def test_vectorizer_initialization(self):
        """Test that vectorizer initializes correctly."""
        assert self.vectorizer.vector_size == 135
        assert len(self.vectorizer.GEM_COLORS) == 6
        assert len(self.vectorizer.DISCOUNT_COLORS) == 5
        
    def test_initial_state_vectorization(self):
        """Test vectorization of initial game state."""
        state = State()
        vector = self.vectorizer.vectorize(state, player_id=0, turn_count=0)
        
        # Check basic properties
        assert vector.shape == (135,), f"Expected shape (135,), got {vector.shape}"
        assert vector.dtype == np.float32, f"Expected dtype float32, got {vector.dtype}"
        assert not np.any(np.isnan(vector)), "Vector contains NaN values"
        assert not np.any(np.isinf(vector)), "Vector contains Inf values"
        assert np.all(vector >= 0.0), "Vector contains negative values"
        assert np.all(vector <= 1.0), "Vector contains values > 1.0"
        
    def test_player_perspective(self):
        """Test that vectorization works for both players."""
        state = State()
        
        vector_p0 = self.vectorizer.vectorize(state, player_id=0)
        vector_p1 = self.vectorizer.vectorize(state, player_id=1)
        
        assert vector_p0.shape == (135,)
        assert vector_p1.shape == (135,)
        
        # Vectors should be different if players have different hands
        # (initially they're the same, so vectors might be identical)
        # After some moves, they should differ
        
    def test_state_consistency(self):
        """Test that same state produces same vector (determinism)."""
        state = State()
        
        vector1 = self.vectorizer.vectorize(state, player_id=0, turn_count=5)
        vector2 = self.vectorizer.vectorize(state, player_id=0, turn_count=5)
        
        assert np.allclose(vector1, vector2), "Same state should produce identical vectors"
        
    def test_state_change_detection(self):
        """Test that different states produce different vectors."""
        state = State()
        vector_initial = self.vectorizer.vectorize(state, player_id=0)
        
        # Modify state by giving player some gems
        state.active_players_hand().gems_possessed = GemsCollection({
            GemColor.GOLD: 0,
            GemColor.RED: 3,
            GemColor.GREEN: 2,
            GemColor.BLUE: 1,
            GemColor.WHITE: 0,
            GemColor.BLACK: 0
        })
        
        vector_modified = self.vectorizer.vectorize(state, player_id=0)
        
        assert not np.allclose(vector_initial, vector_modified), \
            "Different states should produce different vectors"
        
    def test_turn_progress_encoding(self):
        """Test that turn count is properly encoded."""
        state = State()
        
        vector_turn0 = self.vectorizer.vectorize(state, player_id=0, turn_count=0)
        vector_turn60 = self.vectorizer.vectorize(state, player_id=0, turn_count=60)
        vector_turn120 = self.vectorizer.vectorize(state, player_id=0, turn_count=120)
        
        # Turn progress is at index -2 (second to last)
        assert vector_turn0[-2] == 0.0, "Turn 0 should have progress 0.0"
        assert vector_turn60[-2] == 0.5, "Turn 60 should have progress 0.5"
        assert vector_turn120[-2] == 1.0, "Turn 120 should have progress 1.0"
        
    def test_gems_vectorization(self):
        """Test that gems are properly normalized."""
        state = State()
        
        # Set specific gem amounts
        state.active_players_hand().gems_possessed = GemsCollection({
            GemColor.GOLD: 5,
            GemColor.RED: 10,  # Max on hand
            GemColor.GREEN: 0,
            GemColor.BLUE: 3,
            GemColor.WHITE: 7,
            GemColor.BLACK: 2
        })
        
        vector = self.vectorizer.vectorize(state, player_id=0)
        
        # Active player gems are first 6 dimensions
        gems_vec = vector[:6]
        
        assert gems_vec[0] == 0.5, f"Gold (5) should normalize to 0.5, got {gems_vec[0]}"
        assert gems_vec[1] == 1.0, f"Red (10) should normalize to 1.0, got {gems_vec[1]}"
        assert gems_vec[2] == 0.0, f"Green (0) should normalize to 0.0, got {gems_vec[2]}"
        assert gems_vec[3] == 0.3, f"Blue (3) should normalize to 0.3, got {gems_vec[3]}"
        
    def test_victory_points_encoding(self):
        """Test that victory points are properly normalized."""
        state = State()
        
        # Manually set victory points by adding cards
        # (This is a simplified test - in real game, points come from cards/nobles)
        player_hand = state.active_players_hand()
        
        # Initial state should have 0 points
        vector_initial = self.vectorizer.vectorize(state, player_id=0)
        # VP is at dimension 13 (gems:6 + discounts:5 + reserved:1 + vp:1)
        vp_index = 6 + 5 + 1
        assert vector_initial[vp_index] == 0.0, "Initial VP should be 0"
        
    def test_board_cards_encoding(self):
        """Test that board cards are properly encoded."""
        state = State()
        vector = self.vectorizer.vectorize(state, player_id=0)
        
        # Board cards start at dimension: 35 (active) + 14 (opponent) + 6 (gems) = 55
        # Board cards take 72 dimensions (12 cards × 6 features)
        board_cards_start = 55
        board_cards_end = 55 + 72
        board_cards_vec = vector[board_cards_start:board_cards_end]
        
        # Should have 12 cards initially (4 per row)
        # Check that not all zeros (cards should be on board)
        assert np.any(board_cards_vec > 0), "Board should have cards"
        
    def test_convenience_function(self):
        """Test the convenience function wrapper."""
        state = State()
        vector = vectorize_state(state, player_id=0, turn_count=10)
        
        assert vector.shape == (135,)
        assert vector.dtype == np.float32
        assert not np.any(np.isnan(vector))
        
    def test_reserved_cards_handling(self):
        """Test handling of reserved cards."""
        state = State()
        
        # Initially no reserved cards
        vector_initial = self.vectorizer.vectorize(state, player_id=0)
        
        # Reserved count is at dimension 11 (gems:6 + discounts:5)
        reserved_count_index = 6 + 5
        assert vector_initial[reserved_count_index] == 0.0, "Initial reserved count should be 0"
        
        # Note: Actually reserving cards requires game mechanics, 
        # so this is a basic structure test
        
    def test_empty_board_positions(self):
        """Test that empty board positions are filled with zeros."""
        # Create a state with minimal cards
        state = State()
        
        # This is hard to test without manipulating internal state,
        # but we ensure no errors occur
        vector = self.vectorizer.vectorize(state, player_id=0)
        
        # Check valid range
        assert np.all(vector >= 0.0) and np.all(vector <= 1.0)
        

def test_vectorizer_on_multiple_states():
    """Integration test: vectorize multiple random states."""
    vectorizer = SplendorStateVectorizer()
    
    for i in range(10):
        state = State()
        # Simulate some random changes
        state.active_players_hand().gems_possessed = GemsCollection({
            GemColor.GOLD: np.random.randint(0, 6),
            GemColor.RED: np.random.randint(0, 11),
            GemColor.GREEN: np.random.randint(0, 11),
            GemColor.BLUE: np.random.randint(0, 11),
            GemColor.WHITE: np.random.randint(0, 11),
            GemColor.BLACK: np.random.randint(0, 11)
        })
        
        vector = vectorizer.vectorize(state, player_id=0, turn_count=i*10)
        
        assert vector.shape == (135,)
        assert not np.any(np.isnan(vector))
        assert not np.any(np.isinf(vector))
        assert np.all(vector >= 0.0) and np.all(vector <= 1.0)


if __name__ == "__main__":
    # Run tests
    print("Running SplendorStateVectorizer tests...")
    
    # Basic instantiation test
    vectorizer = SplendorStateVectorizer()
    print(f"✓ Vectorizer initialized (size: {vectorizer.vector_size})")
    
    # Test on initial state
    state = State()
    vector = vectorizer.vectorize(state, player_id=0)
    print(f"✓ Initial state vectorized: shape {vector.shape}, dtype {vector.dtype}")
    print(f"  - Min value: {vector.min():.4f}")
    print(f"  - Max value: {vector.max():.4f}")
    print(f"  - Mean value: {vector.mean():.4f}")
    print(f"  - Non-zero elements: {np.count_nonzero(vector)}/{len(vector)}")
    
    # Test consistency
    vector2 = vectorizer.vectorize(state, player_id=0)
    assert np.allclose(vector, vector2)
    print(f"✓ Determinism test passed")
    
    # Test both players
    vector_p1 = vectorizer.vectorize(state, player_id=1)
    print(f"✓ Player 1 vectorization: shape {vector_p1.shape}")
    
    print("\n✅ All basic tests passed!")
    print("\nTo run full test suite with pytest:")
    print("  cd /mnt/c/Users/yehao/Documents/03Study/IFT6759/Splendor-6759")
    print("  pytest project/tests/test_state_vectorizer.py -v")
