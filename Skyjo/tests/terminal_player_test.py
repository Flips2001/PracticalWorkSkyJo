import pytest
from unittest.mock import MagicMock, patch
import curses

from Skyjo.src.players.terminal_player import TerminalPlayer
from Skyjo.src.card import Card
from Skyjo.src.action import Action
from Skyjo.src.action_type import ActionType
from Skyjo.src.turn_phase import TurnPhase


@pytest.fixture
def mock_stdscr():
    """Create a mock curses screen."""
    stdscr = MagicMock()
    stdscr.getmaxyx.return_value = (40, 100)
    stdscr.keypad = MagicMock()
    return stdscr


@pytest.fixture
def terminal_player(mock_stdscr):
    """Create a TerminalPlayer with mocked curses."""
    with patch("Skyjo.src.ui.terminal_ui.init_colors"):
        with patch("curses.curs_set"):
            player = TerminalPlayer(
                player_id=0,
                player_name="TestPlayer",
                stdscr=mock_stdscr,
                opponent_name="Opponent",
            )
    return player


@pytest.fixture
def mock_observation():
    class MockObservation:
        def __init__(self):
            card = Card(3, face_up=True)
            hidden_card = Card(5, face_up=False)
            self.card_grid = [
                [card, card, hidden_card, card],
                [hidden_card, card, card, hidden_card],
                [card, hidden_card, card, card],
            ]
            self.opponent_cards = [
                [
                    [card, card, card, card],
                    [card, card, card, card],
                    [card, card, card, card],
                ]
            ]
            self.discard_top = Card(7, face_up=True)
            self.hand_card = None
            self.scores = [10, 15]
            self.player_id = 0
            self.draw_pile_size = 80
            self.turn_phase = TurnPhase.CHOOSE_DRAW
            self.final_turn_phase = False
            self.first_finisher_id = None

    return MockObservation()


def test_terminal_player_initialization(terminal_player):
    assert terminal_player.player_id == 0
    assert terminal_player.player_name == "TestPlayer"
    assert terminal_player.opponent_name == "Opponent"


def test_terminal_player_select_action_enter(
    terminal_player, mock_stdscr, mock_observation
):
    """Test selecting an action by pressing Enter on first item."""
    legal_actions = [
        Action(ActionType.DRAW_HIDDEN_CARD),
        Action(ActionType.DRAW_OPEN_CARD),
    ]
    # Simulate pressing Enter immediately (select first action)
    mock_stdscr.getch.return_value = 10  # Enter key

    with patch.object(terminal_player.renderer, "render_game"):
        result = terminal_player.select_action(mock_observation, legal_actions)

    assert result == legal_actions[0]


def test_terminal_player_select_action_arrow_down_then_enter(
    terminal_player, mock_stdscr, mock_observation
):
    """Test navigating down then selecting."""
    legal_actions = [
        Action(ActionType.DRAW_HIDDEN_CARD),
        Action(ActionType.DRAW_OPEN_CARD),
    ]
    # Simulate pressing Down then Enter
    mock_stdscr.getch.side_effect = [curses.KEY_DOWN, 10]

    with patch.object(terminal_player.renderer, "render_game"):
        result = terminal_player.select_action(mock_observation, legal_actions)

    assert result == legal_actions[1]


def test_terminal_player_select_action_quit(
    terminal_player, mock_stdscr, mock_observation
):
    """Test quitting with 'q' key."""
    legal_actions = [Action(ActionType.DRAW_HIDDEN_CARD)]
    mock_stdscr.getch.return_value = ord("q")

    with patch.object(terminal_player.renderer, "render_game"):
        with pytest.raises(KeyboardInterrupt):
            terminal_player.select_action(mock_observation, legal_actions)


def test_terminal_player_no_legal_actions(terminal_player, mock_observation):
    """Test that ValueError is raised with no legal actions."""
    with pytest.raises(ValueError):
        terminal_player.select_action(mock_observation, [])


def test_terminal_player_wraps_selection(
    terminal_player, mock_stdscr, mock_observation
):
    """Test that arrow key wraps around the action list."""
    legal_actions = [
        Action(ActionType.DRAW_HIDDEN_CARD),
        Action(ActionType.DRAW_OPEN_CARD),
    ]
    # Press Up (wraps to last) then Enter
    mock_stdscr.getch.side_effect = [curses.KEY_UP, 10]

    with patch.object(terminal_player.renderer, "render_game"):
        result = terminal_player.select_action(mock_observation, legal_actions)

    assert result == legal_actions[1]
