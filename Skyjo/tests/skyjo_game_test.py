from dataclasses import FrozenInstanceError

import pytest

from Skyjo.src.skyjo_game import SkyjoGame
from Skyjo.src.game_state import ColumnClearStats
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

    def select_starting_flips(self, hidden_positions, count=2):
        # For testing, just pick the first `count` hidden positions
        return hidden_positions[:count]

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


def test_get_observation_basic_fields_and_immutability(two_players):
    game, p0, p1 = two_players
    p0_state = game.get_player_state(p0)
    p1_state = game.get_player_state(p1)

    p0_state.grid = grid_from_values([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    p1_state.grid = grid_from_values([[-1, 0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10]])

    game.game_state.discard_pile = [Card(42, face_up=True)]
    game.game_state.draw_pile = [Card(-2), Card(-1)]

    obs = game.get_observation(p0)

    assert obs.player_id == 0
    assert obs.scores == (0, 0)
    assert obs.discard_top.get_value() == 42
    assert obs.draw_pile_size == 2
    assert obs.opponent_cards[0] is None
    assert all(card.is_hidden() for row in obs.opponent_cards[1] for card in row)

    hidden_card = obs.card_grid[0][0]
    assert hidden_card.value is None
    assert not hasattr(hidden_card, "reveal")
    with pytest.raises(ValueError, match="face down"):
        hidden_card.get_value()
    with pytest.raises(FrozenInstanceError):
        hidden_card.face_up = True
    with pytest.raises(FrozenInstanceError):
        obs.draw_pile_size = 0
    with pytest.raises(TypeError):
        obs.card_grid[0][0] = obs.card_grid[0][1]

    p0_state.grid[0][0].reveal()
    assert obs.card_grid[0][0].is_hidden()
    assert not p0_state.grid[0][0].is_hidden()


def test_player_state_is_owned_by_game(two_players):
    game, p0, _ = two_players

    assert not hasattr(p0, "player_state")
    assert game.get_player_state(p0).player_id == p0.player_id


def test_first_round_starter_is_determined_from_revealed_cards(two_players):
    game, p0, p1 = two_players
    game.get_player_state(p0).grid = grid_from_values(
        [[1, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    )
    game.get_player_state(p1).grid = grid_from_values(
        [[5, 6, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    )
    game.get_player_state(p0).grid[0][0].reveal()
    game.get_player_state(p0).grid[0][1].reveal()
    game.get_player_state(p1).grid[0][0].reveal()
    game.get_player_state(p1).grid[0][1].reveal()

    assert game._determine_starting_player() == 1


def test_later_round_starts_with_previous_round_finisher(two_players):
    game, _, _ = two_players
    game.game_state.round_number = 2
    game.game_state.previous_round_finisher_id = 0

    assert game._determine_starting_player() == 0


def test_get_legal_actions_after_drawing_hidden(two_players):
    game, p0, _ = two_players

    game.game_state.phase = TurnPhase.HAVE_DRAWN_HIDDEN
    game.game_state.hand_card = Card(5)
    legal_actions = game.get_legal_actions(p0)

    expected_actions = {
        Action(ActionType.SWAP_CARD, pos=(r, c)) for r in range(3) for c in range(4)
    }.union({Action(ActionType.DISCARD_CARD)})

    assert set(legal_actions) == expected_actions


def test_execute_action_draw_hidden_then_swap(two_players, empty_grid):
    game, p0, _ = two_players
    p0_state = game.get_player_state(p0)
    p0_state.grid = empty_grid

    draw_card = Card(55)
    game.game_state.draw_pile = [draw_card]
    game.game_state.phase = TurnPhase.CHOOSE_DRAW

    game.execute_action(p0, Action(ActionType.DRAW_HIDDEN_CARD))
    assert game.game_state.hand_card is draw_card
    assert (
        game.game_state.phase == TurnPhase.HAVE_DRAWN_HIDDEN
        or TurnPhase.HAVE_DRAWN_OPEN
    )

    game.execute_action(p0, Action(ActionType.SWAP_CARD, pos=(1, 2)))
    assert game.game_state.hand_card is None
    assert p0_state.grid[1][2] is draw_card
    assert draw_card.face_up
    assert game.game_state.discard_pile[-1].face_up
    assert game.game_state.phase == TurnPhase.END_TURN


def test_execute_action_draw_hidden_discard_then_flip(two_players, empty_grid):
    game, p0, _ = two_players
    p0_state = game.get_player_state(p0)
    p0_state.grid = empty_grid

    drawn_card = Card(9)
    game.game_state.draw_pile = [drawn_card]
    game.game_state.phase = TurnPhase.CHOOSE_DRAW

    game.execute_action(p0, Action(ActionType.DRAW_HIDDEN_CARD))
    assert game.game_state.hand_card is drawn_card

    game.execute_action(p0, Action(ActionType.DISCARD_CARD))
    assert game.game_state.phase == TurnPhase.HAVE_TO_FLIP_AFTER_DISCARD

    game.execute_action(p0, Action(ActionType.FLIP_CARD, pos=(0, 0)))
    assert p0_state.grid[0][0].face_up
    assert game.game_state.phase == TurnPhase.END_TURN


def test_execute_action_rejects_action_not_in_legal_actions(two_players):
    game, p0, _ = two_players
    game.game_state.phase = TurnPhase.CHOOSE_DRAW
    game.game_state.draw_pile = [Card(5)]

    illegal_action = Action(ActionType.SWAP_CARD, pos=(0, 0))

    with pytest.raises(ValueError, match="Illegal action"):
        game.execute_action(p0, illegal_action)

    assert game.game_state.phase == TurnPhase.CHOOSE_DRAW
    assert game.game_state.hand_card is None
    assert len(game.game_state.draw_pile) == 1


def test_turn_executes_full_plan_and_resets_phase(game):
    plan = [
        Action(ActionType.DRAW_HIDDEN_CARD),
        Action(ActionType.SWAP_CARD, pos=(0, 0)),
    ]

    p0 = TestPlayer(0, "P0", plan=plan)
    p1 = TestPlayer(1, "P1", plan=[])

    game.add_player(p0)
    game.add_player(p1)

    p0_state = game.get_player_state(p0)
    p1_state = game.get_player_state(p1)
    p0_state.grid = [[Card(0) for _ in range(4)] for _ in range(3)]
    p1_state.grid = [[Card(0) for _ in range(4)] for _ in range(3)]

    game.game_state.draw_pile = [Card(7)]
    game.game_state.phase = TurnPhase.CHOOSE_DRAW

    game.turn(p0)

    assert game.game_state.phase == TurnPhase.CHOOSE_DRAW
    assert any(card.face_up for row in p0_state.grid for card in row)


def test_turn_tracks_total_columns_cleared(game):
    plan = [
        Action(ActionType.DRAW_OPEN_CARD),
        Action(ActionType.SWAP_CARD, pos=(0, 0)),
    ]

    p0 = TestPlayer(0, "P0", plan=plan)
    p1 = TestPlayer(1, "P1", plan=[])

    game.add_player(p0)
    game.add_player(p1)

    p0_state = game.get_player_state(p0)
    p1_state = game.get_player_state(p1)
    p0_state.grid = [
        [
            Card(11, face_up=False),
            Card(4, face_up=True),
            Card(5, face_up=True),
            Card(6, face_up=True),
        ],
        [
            Card(12, face_up=True),
            Card(8, face_up=True),
            Card(9, face_up=True),
            Card(10, face_up=True),
        ],
        [
            Card(12, face_up=True),
            Card(0, face_up=True),
            Card(1, face_up=True),
            Card(2, face_up=True),
        ],
    ]
    p1_state.grid = [[Card(0) for _ in range(4)] for _ in range(3)]
    game.game_state.discard_pile = [Card(12, face_up=True)]
    game.game_state.phase = TurnPhase.CHOOSE_DRAW

    game.turn(p0)

    assert game.total_columns_cleared[0] == 1
    assert game.total_column_clear_value_sum[0] == 36


def test_final_reveal_removes_uniform_columns_before_scoring(two_players):
    game, p0, p1 = two_players
    p0_state = game.get_player_state(p0)
    p1_state = game.get_player_state(p1)

    # Player 0's first column becomes uniform only when the final hidden cards
    # are revealed. The other columns and player 1's grid are non-uniform.
    p0_state.grid = grid_from_values([[5, 1, 2, 3], [5, 4, 5, 6], [5, 7, 8, 9]])
    p1_state.grid = grid_from_values([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    # Keep the complete 150-card deck invariant required by round reset.
    game.game_state.draw_pile = [Card(0) for _ in range(126)]
    game.game_state.discard_pile = []

    game.reset()

    assert game.game_state.all_player_final_scores == [45, 78]
    assert game.last_column_clear_stats[0] == ColumnClearStats(
        columns_removed=1, removed_card_value_sum=15
    )
    assert game.total_columns_cleared[0] == 1
    assert game.total_column_clear_value_sum[0] == 15


def test_observer_snapshots_are_frozen_at_decision_time(two_players):
    game, p0, p1 = two_players
    p0_state = game.get_player_state(p0)
    p0_state.grid = grid_from_values([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

    snapshot = game._observer_snapshots(p0)[p1.player_id]

    # The acting player's grid mutates after the snapshot was taken; the
    # snapshot's view of it must not change with it.
    p0_state.grid[0][0].reveal()

    assert snapshot.opponent_cards[p0.player_id][0][0].face_up is False
