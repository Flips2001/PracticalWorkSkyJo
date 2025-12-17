import pytest
from Skyjo.src.card import Card
from typing import List


@pytest.fixture
def player_state():
    from Skyjo.src.player_state import PlayerState

    return PlayerState(player_id=1)


def test_initial_game_score(player_state):
    assert player_state.get_game_score() == 0


def test_get_round_score_initial(player_state):
    assert player_state.get_round_score() == 0


def test_get_grid(player_state):
    initial_grid = player_state.get_grid()
    assert isinstance(initial_grid, List)
    assert len(initial_grid) == 0  # Default grid should be empty list


def test_calculate_current_score(player_state):
    # Set up a grid with some revealed and hidden cards

    player_state.grid = [
        [Card(5), Card(10), Card(3)],
        [Card(2), Card(7), Card(4)],
        [Card(1), Card(6), Card(8)],
    ]

    player_state.grid[0][0].reveal()
    player_state.grid[0][1].reveal()
    player_state.grid[1][0].reveal()
    player_state.grid[2][2].reveal()

    expected_score = 5 + 10 + 2 + 8  # Only revealed cards count
    calculated_score = player_state.get_round_score()

    assert calculated_score == expected_score
    assert player_state.get_round_score() == expected_score


def test_get_set_game_score(player_state):
    player_state.set_game_score(25)
    assert player_state.get_game_score() == 25


def test_reset(player_state):
    player_state.set_game_score(-65)
    player_state.grid = [[Card(5)]]

    player_state.reset()

    assert player_state.get_game_score() == 0
    assert player_state.grid == []
