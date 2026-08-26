"""Behavioural regression: does the trained model clear a column when it should?

Each board is built one swap away from completing a column, with the completing
card sitting face-up on the discard pile. For a *positive* target, clearing is
unambiguously the best move: a model that learned the tactic draws that card and
swaps it into the gap, removing the column. A model that did not would clear only
~4% of these boards (random draw-open x random target slot), so requiring a clear
majority is a strong, low-noise regression signal.

For a *negative* target the same shape is a trap: the card is still worth taking
(negatives lower the score wherever they land), but completing the column would
hand the minus points back. The mirror test therefore requires the model to take
the card AND leave the column uncompleted. Negative boards keep the gap face
down — a revealed high junk card there can genuinely make completing the best
move, and the guard must only use boards where the rule is unambiguous.

The boards deliberately vary everything *around* the tactic — how deep the
discard pile is, how much of the board is face up, and whether the slot to fill
shows a junk card or is face down. That breadth is what makes this a test of the
tactic rather than of the drill: a generator narrowed to the drill's own context
would pass on a model that has merely learned to recognise that context, and
such a model clears almost nothing in a real mid-round position. Keep it broad.

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
NEGATIVE_CARD_VALUES = [value for value in CARD_VALUES if value < 0]

RUNS = 40
# Well above the ~4% a random policy reaches; tighten after a local calibration
# run if your checkpoint clears more reliably.
MIN_CLEAR_RATE = 0.6
# Negative-trap guard: the model must take the offered negative card on a clear
# majority of boards and must almost never complete the negative column.
MIN_NEGATIVE_TAKE_RATE = 0.6
MAX_NEGATIVE_CLEAR_RATE = 0.1
# Deepest discard pile a board may be dealt, so the tactic gets exercised in
# early-, mid- and late-round pile contexts.
MAX_DISCARD_JUNK = 40


class _PassivePlayer(Player):
    def select_action(self, observation, legal_actions):
        return legal_actions[0]


def _pick_value(rng, budget, exclude=frozenset()):
    candidates = [v for v, count in budget.items() if count > 0 and v not in exclude]
    return int(rng.choice(candidates))


def _one_swap_from_clear_board(rng, values=POSITIVE_CARD_VALUES):
    """A board whose single gap, once filled from the discard, clears a column.

    Non-target columns are kept non-uniform, so the only column that can ever be
    removed is the target one. The reveal fraction, the discard-pile depth and
    whether the gap shows a junk card are all sampled per board — see the module
    docstring for why that breadth is load-bearing (and why negative targets
    keep the gap face down).
    """
    budget = dict(INITIAL_CARD_COUNTS)
    target_col = int(rng.integers(0, 4))
    missing_row = int(rng.integers(0, 3))
    target_value = int(rng.choice(values))
    gap_face_up = target_value > 0 and bool(rng.random() < 0.5)
    reveal_prob = float(rng.uniform(0.3, 1.0))

    # Two target cards already in the column + the completing card on the discard.
    budget[target_value] -= 3
    gap_value = _pick_value(rng, budget, exclude=frozenset({target_value}))
    budget[gap_value] -= 1

    grid: list[list[Card]] = [[None] * 4 for _ in range(3)]
    for col in range(4):
        for row in range(3):
            if col == target_col:
                if row == missing_row:
                    grid[row][col] = Card(gap_value, face_up=gap_face_up)
                else:
                    grid[row][col] = Card(target_value, face_up=True)
                continue
            exclude = {target_value}
            # Compare by engine value: cells above may already be face down.
            above = [grid[r][col]._get_value_for_engine() for r in range(row)]
            if len(above) == 2 and above[0] == above[1]:
                exclude.add(above[0])  # don't let the third cell uniform the column
            value = _pick_value(rng, budget, exclude=frozenset(exclude))
            budget[value] -= 1
            grid[row][col] = Card(value, face_up=bool(rng.random() < reveal_prob))

    opponent_grid = []
    for _ in range(3):
        row_cards = []
        for _ in range(4):
            value = _pick_value(rng, budget)
            budget[value] -= 1
            row_cards.append(Card(value, face_up=bool(rng.random() < 0.5)))
        opponent_grid.append(row_cards)

    # Junk under the completing card, so the pile looks like a real mid-round one.
    discard_pile = []
    for _ in range(int(rng.integers(0, MAX_DISCARD_JUNK + 1))):
        value = _pick_value(rng, budget)
        budget[value] -= 1
        discard_pile.append(Card(value, face_up=True))
    discard_pile.append(Card(target_value, face_up=True))

    remaining = dict(INITIAL_CARD_COUNTS)
    for board in (grid, opponent_grid):
        for board_row in board:
            for card in board_row:
                remaining[card._get_value_for_engine()] -= 1
    for card in discard_pile:
        remaining[card.get_value()] -= 1
    assert all(count >= 0 for count in remaining.values()), "board over-used the deck"
    draw_pile = [Card(v) for v, count in remaining.items() for _ in range(count)]

    return grid, opponent_grid, discard_pile, draw_pile


def _draw_and_swap(model, rng, values=POSITIVE_CARD_VALUES) -> tuple[bool, bool]:
    """Play one draw+swap on a one-swap-from-clear board.

    Returns (drew the open card, cleared the target column).
    """
    from Skyjo.src.players.rl_player import RLPlayer

    grid, opponent_grid, discard_pile, draw_pile = _one_swap_from_clear_board(
        rng, values
    )

    game = SkyjoGame()
    rl_player = RLPlayer(0, "RL", model=model)
    opponent = _PassivePlayer(1, "Opponent")
    game.add_player(rl_player)
    game.add_player(opponent)

    rl_state = game.get_player_state(rl_player)
    opponent_state = game.get_player_state(opponent)
    rl_state.grid = grid
    opponent_state.grid = opponent_grid
    game.game_state.discard_pile = discard_pile
    game.game_state.draw_pile = draw_pile
    game.game_state.hand_card = None
    game.game_state.phase = TurnPhase.CHOOSE_DRAW

    draw_action = rl_player.select_action(
        game.get_observation(rl_player), game.get_legal_actions(rl_player)
    )
    if draw_action.type != ActionType.DRAW_OPEN_CARD:
        return False, False

    game.execute_action(rl_player, draw_action)
    game.game_state.remove_uniform_columns_to_discard_pile(rl_state)

    swap_action = rl_player.select_action(
        game.get_observation(rl_player), game.get_legal_actions(rl_player)
    )
    game.execute_action(rl_player, swap_action)
    clear_stats = game.game_state.remove_uniform_columns_to_discard_pile(rl_state)
    return True, clear_stats.columns_removed > 0


def _load_model():
    pytest.importorskip("sb3_contrib")
    from sb3_contrib import MaskablePPO

    if not CHECKPOINT_PATH.exists():
        pytest.skip(f"RL checkpoint not found: {CHECKPOINT_PATH}")
    try:
        return MaskablePPO.load(str(CHECKPOINT_PATH), device="cpu")
    except (ModuleNotFoundError, ImportError, ValueError, RuntimeError) as exc:
        pytest.skip(f"RL checkpoint cannot be loaded in this environment: {exc}")


def test_model_clears_column_when_it_is_the_best_move():
    model = _load_model()

    cleared = sum(
        _draw_and_swap(model, np.random.default_rng(seed))[1] for seed in range(RUNS)
    )

    assert cleared >= RUNS * MIN_CLEAR_RATE, (
        f"model cleared only {cleared}/{RUNS} winnable columns "
        f"(expected >= {int(RUNS * MIN_CLEAR_RATE)}); the column-clear tactic "
        "appears to have regressed."
    )


def test_model_takes_the_negative_card_without_completing_the_column():
    model = _load_model()

    results = [
        _draw_and_swap(
            model, np.random.default_rng(10_000 + seed), NEGATIVE_CARD_VALUES
        )
        for seed in range(RUNS)
    ]
    took = sum(drew for drew, _ in results)
    completed = sum(cleared for _, cleared in results)

    assert took >= RUNS * MIN_NEGATIVE_TAKE_RATE, (
        f"model took the offered negative card on only {took}/{RUNS} boards "
        f"(expected >= {int(RUNS * MIN_NEGATIVE_TAKE_RATE)}); it appears to have "
        "learned to refuse negative discard cards."
    )
    assert completed <= RUNS * MAX_NEGATIVE_CLEAR_RATE, (
        f"model completed the negative column on {completed}/{RUNS} boards "
        f"(expected <= {int(RUNS * MAX_NEGATIVE_CLEAR_RATE)}); completing hands "
        "the minus points back."
    )
