import pytest
from Skyjo.src.card import Card

@pytest.fixture
def player_state():
    from Skyjo.src.player_state import PlayerState
    return PlayerState(player_id=1)

def test_initial_score(player_state):
    assert player_state.score == 0

def test_get_grid(player_state):
    grid = [
        [Card(1), Card(2)],
        [Card(3), Card(4)]
    ]
    player_state.grid = grid
    
    assert player_state.get_grid() == grid

def test_calculate_score(player_state):
    # Set up a grid with hidden cards
    player_state.grid = [
        [Card(5), Card(10), Card(3)],
        [Card(2), Card(7), Card(4)],
        [Card(1), Card(6), Card(8)]
    ]
    # reveal some cards
    player_state.grid[0][0].reveal()
    player_state.grid[0][1].reveal()
    player_state.grid[1][0].reveal()
    player_state.grid[2][2].reveal()
    
    expected_score = 5 + 10 + 2 + 8  # Only revealed cards count
    calculated_score = player_state.calculate_score()
    
    assert calculated_score == expected_score
    assert player_state.score == expected_score

def test_calculate_score_empty_grid(player_state):
    player_state.grid = []
    assert player_state.calculate_score() == 0

def test_reset(player_state):
    player_state.score = 50
    player_state.grid = [[Card(5)]]
    player_state.reset()
    
    assert player_state.score == 0
    assert player_state.grid == []

