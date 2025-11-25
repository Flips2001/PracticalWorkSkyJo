import pytest
from players.random_player import RandomPlayer
from card import Card

@pytest.fixture
def random_player():
    return RandomPlayer(player_id=0, player_name="TestPlayer")

@pytest.fixture
def card():
    return Card(3)

@pytest.fixture
def mock_observation(card):
    class MockObservation:
        def __init__(self):
            self.own_grid = [[card, card], [card, card]]
            self.discard_top = card
    return MockObservation()

def test_random_player_initialization(random_player):
    assert random_player.player_id == 0
    assert random_player.player_name == "TestPlayer"

def test_random_player_select_action(random_player, mock_observation):
    legal_actions = ["Action1", "Action2", "Action3"]

    selected_action = random_player.select_action(mock_observation, legal_actions)

    assert selected_action in legal_actions

def test_random_player_select_action_no_legal_actions(random_player, mock_observation):

    legal_actions = []

    with pytest.raises(IndexError):
        random_player.select_action(mock_observation, legal_actions)