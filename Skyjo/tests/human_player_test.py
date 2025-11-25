import pytest
from players.human_player import HumanPlayer
from card import Card

@pytest.fixture
def human_player():
    return HumanPlayer(player_id=0, player_name="TestPlayer")

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

def test_human_player_initialization(human_player):
    assert human_player.player_id == 0
    assert human_player.player_name == "TestPlayer"

@pytest.mark.parametrize("user_input, expected_action", [
    ('0', "Action1"),
    ('2', "Action3"),
])
def test_human_player_select_action_various_inputs(
        monkeypatch, human_player, mock_observation, user_input, expected_action
):
    legal_actions = ["Action1", "Action2", "Action3"]

    monkeypatch.setattr('builtins.input', lambda _: user_input)

    selected_action = human_player.select_action(mock_observation, legal_actions)

    assert selected_action == expected_action

def test_human_player_select_action_invalid_input(monkeypatch, human_player, mock_observation):

    legal_actions = ["Action1", "Action2"]

    inputs = iter(['invalid', '5', '-1', '1'])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))

    selected_action = human_player.select_action(mock_observation, legal_actions)

    assert selected_action == "Action2"

def test_human_player_select_action_no_legal_actions(human_player, mock_observation):

    legal_actions = []

    with pytest.raises(ValueError):
        human_player.select_action(mock_observation, legal_actions)