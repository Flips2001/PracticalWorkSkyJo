"""Short curriculum environment for learning the column-clear tactic.

Rewards are deliberately kept on the same small scale as the real-game shaping
(|reward| <= ~0.1) so that mixing this drill into the self-play vector env does
not dominate PPO's advantage normalisation and drown out the win signal. The
board is freshly randomised each reset (values, target column/row, target value,
opponent grid) so the agent learns the *pattern* — "draw the open card that
completes a column, then swap it into the gap" — rather than one fixed layout.
"""

from collections import Counter

import numpy as np
from gymnasium import Env, spaces
from sb3_contrib.common.wrappers import ActionMasker

from Skyjo.src.action import Action
from Skyjo.src.action_type import ActionType
from Skyjo.src.card import Card
from Skyjo.src.observation import Observation
from Skyjo.src.rl.action_mapping import NUM_ACTIONS, action_to_int, int_to_action
from Skyjo.src.rl.encoding import (
    CARD_VALUES,
    INITIAL_CARD_COUNTS,
    OBS_SIZE,
    encode_observation,
)
from Skyjo.src.turn_phase import TurnPhase

# Small rewards, comparable to the real-game round/clear shaping (|r| <= ~0.1).
DRILL_DRAW_REWARD = 0.05
DRILL_SWAP_REWARD = 0.1
DRILL_BAD_CLEAR_REWARD = 0.05

# Fraction of non-target player cells that stay revealed (the rest are hidden).
_PLAYER_REVEAL_PROB = 0.7


class ColumnClearDrillEnv(Env):
    """Two-step drill: take useful discard, then swap it into the clearing slot."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-0.5, high=1.5, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self._rng = np.random.default_rng()
        self._target_pos = (0, 0)
        self._grid = []
        self._opponent_grid = []
        self._draw_pile = []
        self._phase = TurnPhase.CHOOSE_DRAW
        self._target_value = 12
        self._current_mask = np.zeros(NUM_ACTIONS, dtype=np.int8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        target_col = int(self._rng.integers(0, 4))
        missing_row = int(self._rng.integers(0, 3))
        # 0-value clears are neutral in-game, so never use them as a target:
        # the drill should only ever teach "good clear" (>0) or "avoid bad
        # clear" (<0) decisions.
        target_value = int(self._rng.choice(CARD_VALUES))
        while target_value == 0:
            target_value = int(self._rng.choice(CARD_VALUES))

        self._target_value = target_value
        self._target_pos = (missing_row, target_col)
        self._grid, self._opponent_grid, discard_pile, self._draw_pile = (
            _make_drill_state(self._rng, target_col, missing_row, target_value)
        )
        self._phase = TurnPhase.CHOOSE_DRAW
        self._current_mask = self._draw_mask()
        return encode_observation(self._observation(discard_top=discard_pile[-1])), {}

    def step(self, action_int):
        action = int_to_action(int(action_int))

        if self._phase == TurnPhase.CHOOSE_DRAW:
            if self._target_value < 0:
                # Completing this column would *raise* our score: the right move
                # is to leave it and draw a hidden card instead.
                reward = (
                    DRILL_BAD_CLEAR_REWARD
                    if action.type == ActionType.DRAW_HIDDEN_CARD
                    else -DRILL_BAD_CLEAR_REWARD
                )
                return np.zeros(OBS_SIZE, dtype=np.float32), reward, True, False, {}

            if action.type == ActionType.DRAW_OPEN_CARD:
                self._phase = TurnPhase.HAVE_DRAWN_OPEN
                self._current_mask = self._swap_mask()
                return (
                    encode_observation(
                        self._observation(hand_card=Card(self._target_value, True))
                    ),
                    DRILL_DRAW_REWARD,
                    False,
                    False,
                    {},
                )

            return (
                np.zeros(OBS_SIZE, dtype=np.float32),
                -DRILL_DRAW_REWARD,
                True,
                False,
                {},
            )

        correct_action = Action(ActionType.SWAP_CARD, pos=self._target_pos)
        reward = DRILL_SWAP_REWARD if action == correct_action else -DRILL_SWAP_REWARD
        return np.zeros(OBS_SIZE, dtype=np.float32), reward, True, False, {}

    def action_masks(self) -> np.ndarray:
        return self._current_mask

    def _draw_mask(self) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
        mask[action_to_int(Action(ActionType.DRAW_HIDDEN_CARD))] = 1
        mask[action_to_int(Action(ActionType.DRAW_OPEN_CARD))] = 1
        return mask

    def _swap_mask(self) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
        for row in range(3):
            for col in range(4):
                mask[action_to_int(Action(ActionType.SWAP_CARD, pos=(row, col)))] = 1
        return mask

    def _observation(
        self, discard_top: Card | None = None, hand_card: Card | None = None
    ) -> Observation:
        return Observation(
            player_id=0,
            card_grid=self._grid,
            hand_card=hand_card,
            opponent_cards=[None, self._opponent_grid],
            scores=[
                self._round_score(self._grid),
                self._round_score(self._opponent_grid),
            ],
            discard_top=discard_top,
            draw_pile_size=len(self._draw_pile),
            turn_phase=self._phase,
            draw_pile_value_counts=self._draw_pile_value_counts(),
        )

    def _draw_pile_value_counts(self) -> list[int]:
        value_counts = Counter(card.value for card in self._draw_pile)
        return [value_counts.get(value, 0) for value in CARD_VALUES]

    def _round_score(self, grid: list[list[Card]]) -> int:
        return sum(card.value for card in _iter_grid_cards(grid) if card.face_up)


def mask_fn(env: Env) -> np.ndarray:
    return getattr(env, "action_masks")()


def make_column_clear_drill_env():
    def _init():
        return ActionMasker(ColumnClearDrillEnv(), mask_fn)

    return _init


def _make_drill_state(rng, target_col: int, missing_row: int, target_value: int):
    """Randomised board that is one swap away from clearing the target column.

    The target value never appears in revealed non-target cells, so the target
    column is the unique column the drawn card can complete; other columns may
    still show coincidental matches (useful distractors).
    """
    budget = dict(INITIAL_CARD_COUNTS)

    # Two visible target cards in the column + one on the discard top.
    budget[target_value] -= 3
    hidden_value = _pick_value(rng, budget, exclude=frozenset({target_value}))
    budget[hidden_value] -= 1

    grid: list[list[Card]] = []
    for row in range(3):
        cards = []
        for col in range(4):
            if col == target_col:
                if row == missing_row:
                    cards.append(Card(hidden_value, face_up=False))
                else:
                    cards.append(Card(target_value, face_up=True))
            else:
                value = _pick_value(rng, budget, exclude=frozenset({target_value}))
                budget[value] -= 1
                face_up = bool(rng.random() < _PLAYER_REVEAL_PROB)
                cards.append(Card(value, face_up=face_up))
        grid.append(cards)

    opponent_grid: list[list[Card]] = []
    for _ in range(3):
        cards = []
        for _ in range(4):
            value = _pick_value(rng, budget)
            budget[value] -= 1
            cards.append(Card(value, face_up=bool(rng.random() < 0.5)))
        opponent_grid.append(cards)

    discard_pile = [Card(target_value, face_up=True)]
    draw_pile = _remaining_draw_pile([grid, opponent_grid], discard_pile)
    return grid, opponent_grid, discard_pile, draw_pile


def _iter_grid_cards(grid: list[list[Card]]):
    for row in grid:
        yield from row


def _pick_value(rng, budget: dict[int, int], exclude: frozenset = frozenset()) -> int:
    candidates = [
        value for value, count in budget.items() if count > 0 and value not in exclude
    ]
    if not candidates:
        raise ValueError("No card values left in the deck budget.")
    return int(rng.choice(candidates))


def _remaining_draw_pile(
    grids: list[list[list[Card]]], discard_pile: list[Card]
) -> list[Card]:
    """Every card not already dealt into the grids or discard pile."""
    remaining_counts = dict(INITIAL_CARD_COUNTS)
    for grid in grids:
        for card in _iter_grid_cards(grid):
            remaining_counts[card.value] -= 1
    for card in discard_pile:
        remaining_counts[card.value] -= 1

    if any(count < 0 for count in remaining_counts.values()):
        raise ValueError("Drill state uses more cards than exist in the deck.")

    return [
        Card(value) for value, count in remaining_counts.items() for _ in range(count)
    ]
