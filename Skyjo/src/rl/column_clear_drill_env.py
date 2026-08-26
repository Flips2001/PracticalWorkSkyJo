"""Short curriculum environment for learning the column-clear tactic.

Each reset samples one of two scenarios (see `DrillMode`):

* ``CLEAR_COLUMN`` — two matching cards are already face up in a column and the
  third lies on the discard top. For a positive target: take it and swap it into
  the gap, removing the column. For a negative target the card is still worth
  taking — negatives lower the score wherever they land — but completing the
  column would hand the minus points back, so the drill rewards placing it on
  any slot outside that column that it improves: a hidden card or a revealed
  higher one.
* ``BUILD_PAIR`` — a mostly-covered board shows a single card and its match lies
  on the discard top: take it and swap it into the *same* column, lining the pair
  up so a later third card can clear it.

Rewards are deliberately kept on the same small scale as the real-game shaping
(|reward| <= ~0.1) so that mixing this drill into the self-play vector env does
not dominate PPO's advantage normalisation and drown out the win signal.

Everything the scenario does not pin down is randomised per reset — values,
target column/row, target value, opponent grid, how much of the board is face
up, whether the slot to fill is face down or shows a junk card, and how deep the
discard pile is. The tactic has to be recognisable from the board pattern alone,
so no other part of the observation may correlate with it: a constant would let
the policy key on that instead and the behaviour would not survive outside the
drill.
"""

from enum import Enum, auto

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
# Pairing only *sets up* a clear, so it pays half of what finishing one pays.
DRILL_BUILD_DRAW_REWARD = 0.025
DRILL_BUILD_SWAP_REWARD = 0.05

# Share of resets that drill pair building instead of finishing a column.
DEFAULT_BUILD_PAIR_PROB = 0.5

# Everything below varies per episode; keep it that way. A value held constant
# is a tell the policy can key the tactic on instead of the board pattern.
_MAX_DISCARD_JUNK = 40
# Fraction of non-target player cells that stay revealed (the rest are hidden).
_REVEAL_PROB_RANGE = (0.2, 0.9)
# Pair building starts from a mostly-covered board, so outside the pair column
# only a few distractors are face up.
_BUILD_REVEAL_PROB_RANGE = (0.1, 0.5)
# Share of positive-target CLEAR_COLUMN resets whose completing slot already
# shows a junk card rather than being face down; both shapes occur in real
# positions. Negative targets always deal the gap face down: a revealed high
# junk card there can make completing the negative column the best move
# (dumping it outweighs the returned minus points), and the drill must never
# produce a board where its own "never complete" lesson is wrong.
_FACE_UP_GAP_PROB = 0.5


class DrillMode(Enum):
    CLEAR_COLUMN = auto()
    BUILD_PAIR = auto()


class ColumnClearDrillEnv(Env):
    """Two-step drill: take the useful discard, then swap it into the right slot."""

    metadata = {"render_modes": []}

    def __init__(self, build_pair_prob: float = DEFAULT_BUILD_PAIR_PROB):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-0.5, high=1.5, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self._build_pair_prob = build_pair_prob
        self._rng = np.random.default_rng()
        self._mode = DrillMode.CLEAR_COLUMN
        self._reward_positions: frozenset[tuple[int, int]] = frozenset()
        self._grid = []
        self._opponent_grid = []
        self._discard_pile = []
        self._draw_pile = []
        self._phase = TurnPhase.CHOOSE_DRAW
        self._target_value = 12
        self._current_mask = np.zeros(NUM_ACTIONS, dtype=np.int8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._mode = (
            DrillMode.BUILD_PAIR
            if self._rng.random() < self._build_pair_prob
            else DrillMode.CLEAR_COLUMN
        )
        target_col = int(self._rng.integers(0, 4))
        # The gap to fill (CLEAR_COLUMN) resp. the seed card's row (BUILD_PAIR).
        target_row = int(self._rng.integers(0, 3))
        self._target_value = _pick_target_value(self._rng)

        make_state = (
            _make_build_pair_state
            if self._mode is DrillMode.BUILD_PAIR
            else _make_drill_state
        )
        (
            self._grid,
            self._opponent_grid,
            self._discard_pile,
            self._draw_pile,
        ) = make_state(self._rng, target_col, target_row, self._target_value)
        self._reward_positions = self._rewarded_swaps(target_col, target_row)
        self._phase = TurnPhase.CHOOSE_DRAW
        self._current_mask = self._draw_mask()
        return encode_observation(self._observation()), {}

    def step(self, action_int):
        action = int_to_action(int(action_int))
        draw_reward, swap_reward = self._reward_scale()

        if self._phase == TurnPhase.CHOOSE_DRAW:
            # The offered card is worth taking in every scenario — a low one
            # lowers our score wherever it lands, a high one buys the pair — so
            # the draw step always drills DRAW_OPEN. The negative-column trap
            # lives at the swap step: `_rewarded_swaps` excludes that column
            # and any revealed card the drawn one cannot beat.
            if action.type == ActionType.DRAW_OPEN_CARD:
                self._phase = TurnPhase.HAVE_DRAWN_OPEN
                self._current_mask = self._swap_mask()
                # Taking the top card uncovers whatever lies under it, exactly as
                # the real engine's DRAW_OPEN_CARD does.
                hand_card = self._discard_pile.pop()
                return (
                    encode_observation(self._observation(hand_card=hand_card)),
                    draw_reward,
                    False,
                    False,
                    {},
                )

            return (
                np.zeros(OBS_SIZE, dtype=np.float32),
                -draw_reward,
                True,
                False,
                {},
            )

        correct = (
            action.type == ActionType.SWAP_CARD and action.pos in self._reward_positions
        )
        reward = swap_reward if correct else -swap_reward
        return np.zeros(OBS_SIZE, dtype=np.float32), reward, True, False, {}

    def action_masks(self) -> np.ndarray:
        return self._current_mask

    def _rewarded_swaps(
        self, target_col: int, target_row: int
    ) -> frozenset[tuple[int, int]]:
        """Swap slots that earn the positive reward for this episode."""
        if self._target_value < 0:
            # A column of negative cards is a trap: the engine removes uniform
            # columns automatically, so completing (CLEAR_COLUMN) or lining up
            # (BUILD_PAIR) one would give the minus points back. The drawn
            # negative card still belongs on the board, but only where it
            # improves it — a hidden card or a revealed higher one. Covering a
            # revealed card it cannot beat (a -2 under a drawn -1) would raise
            # the score and hand the better card to the opponent.
            return frozenset(
                (row, col)
                for row in range(3)
                for col in range(4)
                if col != target_col
                and (
                    not self._grid[row][col].face_up
                    or self._grid[row][col].get_value() > self._target_value
                )
            )

        if self._mode is DrillMode.CLEAR_COLUMN:
            return frozenset({(target_row, target_col)})

        # Either remaining slot of the column lines the pair up.
        return frozenset((row, target_col) for row in range(3) if row != target_row)

    def _reward_scale(self) -> tuple[float, float]:
        """(draw, swap) reward magnitudes for the current scenario."""
        if self._mode is DrillMode.BUILD_PAIR:
            return DRILL_BUILD_DRAW_REWARD, DRILL_BUILD_SWAP_REWARD
        return DRILL_DRAW_REWARD, DRILL_SWAP_REWARD

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

    def _observation(self, hand_card: Card | None = None) -> Observation:
        """The drill board as the real engine would report it.

        Discard top and the per-value pile counts are read off the actual pile,
        so the observation is indistinguishable from a real mid-round one.
        """
        return Observation(
            player_id=0,
            card_grid=self._grid,
            hand_card=hand_card,
            opponent_cards=[None, self._opponent_grid],
            scores=[
                self._round_score(self._grid),
                self._round_score(self._opponent_grid),
            ],
            discard_top=self._discard_pile[-1] if self._discard_pile else None,
            draw_pile_size=len(self._draw_pile),
            turn_phase=self._phase,
            discard_pile_value_counts=[
                sum(1 for card in self._discard_pile if card.get_value() == value)
                for value in CARD_VALUES
            ],
        )

    def _round_score(self, grid: list[list[Card]]) -> int:
        return sum(card.get_value() for card in _iter_grid_cards(grid) if card.face_up)


def mask_fn(env: Env) -> np.ndarray:
    return getattr(env, "action_masks")()


def make_column_clear_drill_env(build_pair_prob: float = DEFAULT_BUILD_PAIR_PROB):
    def _init():
        return ActionMasker(ColumnClearDrillEnv(build_pair_prob), mask_fn)

    return _init


def _pick_target_value(rng) -> int:
    """A non-zero card value.

    0-value columns are score-neutral in-game, so they teach neither "complete
    the column" (>0) nor "keep it uncompleted" (<0) — the drill never uses them
    as a target.
    """
    value = 0
    while value == 0:
        value = int(rng.choice(CARD_VALUES))
    return value


def _make_drill_state(rng, target_col: int, missing_row: int, target_value: int):
    """Randomised board that is one swap away from clearing the target column.

    The target value never appears in revealed non-target cells, so the target
    column is the unique column the drawn card can complete; other columns may
    still show coincidental matches (useful distractors). The slot to fill is
    face down or shows a junk card, since both shapes occur in real play.
    """
    budget = dict(INITIAL_CARD_COUNTS)

    # Two visible target cards in the column + one on the discard top.
    budget[target_value] -= 3
    gap_value = _pick_value(rng, budget, exclude=frozenset({target_value}))
    budget[gap_value] -= 1
    # Negative targets keep the gap face down — see _FACE_UP_GAP_PROB.
    gap_face_up = target_value > 0 and bool(rng.random() < _FACE_UP_GAP_PROB)
    reveal_prob = rng.uniform(*_REVEAL_PROB_RANGE)

    grid: list[list[Card]] = []
    for row in range(3):
        cards = []
        for col in range(4):
            if col == target_col:
                if row == missing_row:
                    cards.append(Card(gap_value, face_up=gap_face_up))
                else:
                    cards.append(Card(target_value, face_up=True))
            else:
                value = _pick_value(rng, budget, exclude=frozenset({target_value}))
                budget[value] -= 1
                face_up = bool(rng.random() < reveal_prob)
                cards.append(Card(value, face_up=face_up))
        grid.append(cards)
    _hide_accidental_clears(rng, grid)

    opponent_grid = _make_opponent_grid(rng, budget)
    discard_pile = _make_discard_pile(rng, budget, target_value)
    draw_pile = _remaining_draw_pile([grid, opponent_grid], discard_pile)
    return grid, opponent_grid, discard_pile, draw_pile


def _make_build_pair_state(rng, pair_col: int, seed_row: int, pair_value: int):
    """Mostly-covered board with one revealed card whose match is on the discard.

    The two other cells of the pair column stay face down, so swapping the drawn
    card into either of them lines the pair up. `pair_value` occurs nowhere else
    on the board, which makes the pair column the only pairing site the agent can
    see and keeps a third copy available in the draw pile.
    """
    budget = dict(INITIAL_CARD_COUNTS)

    # The revealed seed card + its match on the discard top.
    budget[pair_value] -= 2
    reveal_prob = rng.uniform(*_BUILD_REVEAL_PROB_RANGE)

    grid: list[list[Card]] = []
    for row in range(3):
        cards = []
        for col in range(4):
            if col == pair_col and row == seed_row:
                cards.append(Card(pair_value, face_up=True))
                continue
            value = _pick_value(rng, budget, exclude=frozenset({pair_value}))
            budget[value] -= 1
            # Both free pair-column slots stay hidden so neither is favoured by
            # its face value.
            face_up = col != pair_col and bool(rng.random() < reveal_prob)
            cards.append(Card(value, face_up=face_up))
        grid.append(cards)
    _hide_accidental_clears(rng, grid)

    opponent_grid = _make_opponent_grid(rng, budget)
    discard_pile = _make_discard_pile(rng, budget, pair_value)
    draw_pile = _remaining_draw_pile([grid, opponent_grid], discard_pile)
    return grid, opponent_grid, discard_pile, draw_pile


def _make_discard_pile(rng, budget: dict[int, int], top_value: int) -> list[Card]:
    """Discard pile of random depth with ``top_value`` face up on top.

    The cards underneath shrink the draw pile and populate the per-value count
    features, which is what stops the pile from identifying a drill board. Their
    values are unconstrained: pile contents never change which move is correct.
    """
    pile = []
    for _ in range(int(rng.integers(0, _MAX_DISCARD_JUNK + 1))):
        value = _pick_value(rng, budget)
        budget[value] -= 1
        pile.append(Card(value, face_up=True))
    pile.append(Card(top_value, face_up=True))
    return pile


def _make_opponent_grid(rng, budget: dict[int, int]) -> list[list[Card]]:
    grid: list[list[Card]] = []
    for _ in range(3):
        cards = []
        for _ in range(4):
            value = _pick_value(rng, budget)
            budget[value] -= 1
            cards.append(Card(value, face_up=bool(rng.random() < 0.5)))
        grid.append(cards)
    _hide_accidental_clears(rng, grid)
    return grid


def _hide_accidental_clears(rng, grid: list[list[Card]]):
    """Hide one card of any fully revealed uniform column.

    Cell values are drawn independently, so a column can come out uniform by
    chance — a state the engine would already have cleared away, and one the
    agent must therefore never be trained on.
    """
    for col in range(4):
        cards = [row[col] for row in grid]
        if all(card.face_up for card in cards) and (
            len({card.get_value() for card in cards}) == 1
        ):
            cards[int(rng.integers(0, 3))].face_up = False


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
            remaining_counts[card._get_value_for_engine()] -= 1
    for card in discard_pile:
        remaining_counts[card.get_value()] -= 1

    if any(count < 0 for count in remaining_counts.values()):
        raise ValueError("Drill state uses more cards than exist in the deck.")

    return [
        Card(value) for value, count in remaining_counts.items() for _ in range(count)
    ]
