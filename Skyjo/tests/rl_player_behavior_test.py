"""Behavioural regression: does the trained model clear a column when it should?

Each board is built one swap away from completing a *positive-value* column,
with the completing card sitting face-up on the discard pile — so clearing is
unambiguously the best move. A model that learned the tactic draws that card and
swaps it into the gap, removing the column. A model that did not would clear only
~4% of these boards (random draw-open x random target slot), so requiring a clear
majority is a strong, low-noise regression signal.

The test skips when the checkpoint is absent or cannot be deserialised in the
current environment (e.g. a numpy major-version mismatch with the trained
artifact) — those are environment issues, not tactic regressions.
"""

from pathlib import Path

import numpy as np
import pytest

from Skyjo.src.action_type import ActionType
from Skyjo.src.card import Card
from Skyjo.src.players.player import Player
from Skyjo.src.rl.encoding import CARD_VALUES, INITIAL_CARD_COUNTS
from Skyjo.src.skyjo_game import SkyjoGame
from Skyjo.src.turn_phase import TurnPhase

CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "rl"
    / "checkpoints"
    / "skyjo_ppo_best.zip"
)

POSITIVE_CARD_VALUES = [value for value in CARD_VALUES if value > 0]

RUNS = 20
# Well above the ~4% a random policy reaches; tighten after a local calibration
# run if your checkpoint clears more reliably.
MIN_CLEAR_RATE = 0.6


class _PassivePlayer(Player):
    def select_action(self, observation, legal_actions):
        return legal_actions[0]


def _pick_value(rng, budget, exclude=frozenset()):
    candidates = [v for v, count in budget.items() if count > 0 and v not in exclude]
    return int(rng.choice(candidates))


def _one_swap_from_clear_board(rng):
    """A fully-revealed board whose single gap, once filled, clears a >0 column.

    Non-target columns are kept non-uniform, so the only column that can ever be
    removed is the target one.
    """
    budget = dict(INITIAL_CARD_COUNTS)
    target_col = int(rng.integers(0, 4))
    missing_row = int(rng.integers(0, 3))
    target_value = int(rng.choice(POSITIVE_CARD_VALUES))

    # Two target cards already in the column + the completing card on the discard.
    budget[target_value] -= 3
    hidden_value = _pick_value(rng, budget, exclude=frozenset({target_value}))
    budget[hidden_value] -= 1

    grid: list[list[Card]] = [[None] * 4 for _ in range(3)]
    for col in range(4):
        for row in range(3):
            if col == target_col:
                if row == missing_row:
                    grid[row][col] = Card(hidden_value, face_up=False)
                else:
                    grid[row][col] = Card(target_value, face_up=True)
                continue
            exclude = {target_value}
            above = [grid[r][col].value for r in range(row)]
            if len(above) == 2 and above[0] == above[1]:
                exclude.add(above[0])  # don't let the third cell uniform the column
            value = _pick_value(rng, budget, exclude=frozenset(exclude))
            budget[value] -= 1
            grid[row][col] = Card(value, face_up=True)

    opponent_grid = []
    for _ in range(3):
        row_cards = []
        for _ in range(4):
            value = _pick_value(rng, budget)
            budget[value] -= 1
            row_cards.append(Card(value, face_up=bool(rng.random() < 0.5)))
        opponent_grid.append(row_cards)

    discard_pile = [Card(target_value, face_up=True)]

    remaining = dict(INITIAL_CARD_COUNTS)
    for board in (grid, opponent_grid):
        for board_row in board:
            for card in board_row:
                remaining[card.value] -= 1
    remaining[target_value] -= 1  # discard top
    draw_pile = [Card(v) for v, count in remaining.items() for _ in range(count)]

    return grid, opponent_grid, discard_pile, draw_pile


def _model_clears_column(model, rng) -> bool:
    from Skyjo.src.players.rl_player import RLPlayer

    grid, opponent_grid, discard_pile, draw_pile = _one_swap_from_clear_board(rng)

    game = SkyjoGame()
    rl_player = RLPlayer(0, "RL", model=model)
    opponent = _PassivePlayer(1, "Opponent")
    game.add_player(rl_player)
    game.add_player(opponent)

    rl_player.player_state.grid = grid
    opponent.player_state.grid = opponent_grid
    game.game_state.discard_pile = discard_pile
    game.game_state.draw_pile = draw_pile
    game.game_state.hand_card = None
    game.game_state.phase = TurnPhase.CHOOSE_DRAW

    draw_action = rl_player.select_action(
        game.get_observation(rl_player), game.get_legal_actions(rl_player)
    )
    if draw_action.type != ActionType.DRAW_OPEN_CARD:
        return False

    game.execute_action(rl_player, draw_action)
    game.game_state.remove_uniform_columns_to_discard_pile(rl_player.player_state)

    swap_action = rl_player.select_action(
        game.get_observation(rl_player), game.get_legal_actions(rl_player)
    )
    game.execute_action(rl_player, swap_action)
    clear_stats = game.game_state.remove_uniform_columns_to_discard_pile(
        rl_player.player_state
    )
    return clear_stats.columns_removed > 0


def test_model_clears_column_when_it_is_the_best_move():
    pytest.importorskip("sb3_contrib")
    from sb3_contrib import MaskablePPO

    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"RL checkpoint not found: {CHECKPOINT_PATH}")
    try:
        model = MaskablePPO.load(str(CHECKPOINT_PATH), device="cpu")
    except (ModuleNotFoundError, ImportError, ValueError, RuntimeError) as exc:
        pytest.skip(f"RL checkpoint cannot be loaded in this environment: {exc}")

    cleared = sum(
        _model_clears_column(model, np.random.default_rng(seed)) for seed in range(RUNS)
    )

    assert cleared >= RUNS * MIN_CLEAR_RATE, (
        f"model cleared only {cleared}/{RUNS} winnable columns "
        f"(expected >= {int(RUNS * MIN_CLEAR_RATE)}); the column-clear tactic "
        "appears to have regressed."
    )
