import pytest

from Skyjo.src.skyjo_game import SkyjoGame
from Skyjo.src.players.player import Player
from Skyjo.src.card import Card
from Skyjo.src.action import Action
from Skyjo.src.action_type import ActionType
from Skyjo.src.turn_phase import TurnPhase


class TestPlayer(Player):
    """Deterministic test player with a scripted action plan."""

    def __init__(self, player_id: int, name: str, plan: list[Action]):
        super().__init__(player_id, name)
        self._plan = list(plan)

    def select_action(self, observation, legal_actions):
        assert self._plan, "Action plan exhausted"
        target = self._plan.pop(0)

        for action in legal_actions:
            if action.type == target.type and (
                target.pos is None or target.pos == action.pos
            ):
                return action

        raise AssertionError(
            f"Planned action {target} not in legal actions {legal_actions}"
        )


def grid_from_values(values):
    return [[Card(v) for v in row] for row in values]


@pytest.fixture
def game():
    return SkyjoGame()


@pytest.fixture
def two_players(game):
    p0 = TestPlayer(0, "P0", plan=[])
    p1 = TestPlayer(1, "P1", plan=[])
    game.add_player(p0)
    game.add_player(p1)
    return game, p0, p1


@pytest.fixture
def empty_grid():
    return [[Card(0) for _ in range(4)] for _ in range(3)]


def test_get_observation_basic_fields_and_deepcopy(two_players):
    game, p0, p1 = two_players

    p0.player_state.grid = grid_from_values(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    )
    p1.player_state.grid = grid_from_values(
        [[-1, 0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10]]
    )

    game.game_state.discard_pile = [Card(42)]
    game.game_state.draw_pile = [Card(-2), Card(-1)]

    obs = game.get_observation(p0)

    assert obs.player_id == 0
    assert obs.scores == [0, 0]
    assert obs.discard_top.value == 42
    assert obs.draw_pile_size == 2
    assert obs.opponent_cards == [None, p1.player_state.grid]

    # deepcopy check
    obs.card_grid[0][0].reveal()
    assert p0.player_state.grid[0][0].is_hidden()


def test_get_legal_actions_by_phase(two_players):
    game, p0, _ = two_players

    p0.player_state.grid = grid_from_values(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    )
    p0.player_state.grid[0][0].reveal()

    game.game_state.draw_pile = [Card(99)]

    game.game_state.phase = TurnPhase.CHOOSE_DRAW
    game.game_state.discard_pile = []
    legal = game.get_legal_actions(p0)
    assert {a.type for a in legal} == {ActionType.DRAW_HIDDEN_CARD}

    game.game_state.discard_pile = [Card(7)]
    legal = game.get_legal_actions(p0)
    assert {a.type for a in legal} == {
        ActionType.DRAW_HIDDEN_CARD,
        ActionType.DRAW_OPEN_CARD,
    }

    game.game_state.phase = TurnPhase.HAVE_DRAWN
    legal = game.get_legal_actions(p0)
    assert sum(a.type == ActionType.SWAP_CARD for a in legal) == 12
    assert any(a.type == ActionType.DISCARD_CARD for a in legal)

    game.game_state.phase = TurnPhase.HAVE_TO_FLIP_AFTER_DISCARD
    legal = game.get_legal_actions(p0)
    assert sum(a.type == ActionType.FLIP_CARD for a in legal) == 11

    game.game_state.phase = TurnPhase.END_TURN
    assert game.get_legal_actions(p0) == []


def test_execute_action_draw_hidden_then_swap(two_players, empty_grid):
    game, p0, _ = two_players
    p0.player_state.grid = empty_grid

    draw_card = Card(55)
    game.game_state.draw_pile = [draw_card]
    game.game_state.phase = TurnPhase.CHOOSE_DRAW

    game.execute_action(p0, Action(ActionType.DRAW_HIDDEN_CARD))
    assert game.game_state.hand_card is draw_card
    assert game.game_state.phase == TurnPhase.HAVE_DRAWN

    game.execute_action(p0, Action(ActionType.SWAP_CARD, pos=(1, 2)))
    assert game.game_state.hand_card is None
    assert p0.player_state.grid[1][2] is draw_card
    assert draw_card.face_up
    assert game.game_state.discard_pile[-1].face_up
    assert game.game_state.phase == TurnPhase.END_TURN


def test_execute_action_draw_open_discard_then_flip(two_players, empty_grid):
    game, p0, _ = two_players
    p0.player_state.grid = empty_grid

    discard = Card(9)
    discard.reveal()
    game.game_state.discard_pile = [discard]
    game.game_state.phase = TurnPhase.CHOOSE_DRAW

    game.execute_action(p0, Action(ActionType.DRAW_OPEN_CARD))
    assert game.game_state.hand_card is discard

    game.execute_action(p0, Action(ActionType.DISCARD_CARD))
    assert game.game_state.phase == TurnPhase.HAVE_TO_FLIP_AFTER_DISCARD

    game.execute_action(p0, Action(ActionType.FLIP_CARD, pos=(0, 0)))
    assert p0.player_state.grid[0][0].face_up
    assert game.game_state.phase == TurnPhase.END_TURN


def test_turn_executes_full_plan_and_resets_phase(game):
    plan = [
        Action(ActionType.DRAW_HIDDEN_CARD),
        Action(ActionType.SWAP_CARD, pos=(0, 0)),
    ]

    p0 = TestPlayer(0, "P0", plan=plan)
    p1 = TestPlayer(1, "P1", plan=[])

    game.add_player(p0)
    game.add_player(p1)

    p0.player_state.grid = [[Card(0) for _ in range(4)] for _ in range(3)]
    p1.player_state.grid = [[Card(0) for _ in range(4)] for _ in range(3)]

    game.game_state.draw_pile = [Card(7)]
    game.game_state.phase = TurnPhase.CHOOSE_DRAW

    game.turn(p0)

    assert game.game_state.phase == TurnPhase.CHOOSE_DRAW
    assert any(card.face_up for row in p0.player_state.grid for card in row)
