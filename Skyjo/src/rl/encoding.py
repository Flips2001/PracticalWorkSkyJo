"""
Encodes an Observation into a flat 75-dim numpy vector for the RL agent.

Card values are normalized via (value + 2) / 14 → [0, 1].
Each grid slot is encoded as a (value, is_revealed) pair:
  - Face-up card:   (normalized_value, 1.0)
  - Face-down card: (0.0, 0.0) — value left at default zero (sentinel, not
    the normalization formula applied to any real card value). Note that
    normalized(-2) also equals 0.0, but the is_revealed flag (0 vs 1)
    disambiguates face-down slots from revealed -2 cards.
  - Removed column: (normalized(0), 1.0) — see _encode_grid() for details.
"""

import numpy as np
from gymnasium import spaces
from typing import Optional, List

from Skyjo.src.observation import Observation
from Skyjo.src.card import Card
from Skyjo.src.turn_phase import TurnPhase

OBS_SIZE = 75
GRID_ROWS = 3
GRID_COLS = 4
CARD_VALUES = list(range(-2, 13))

INITIAL_CARD_COUNTS = {
    -2: 5,
    -1: 10,
    0: 15,
    1: 10,
    2: 10,
    3: 10,
    4: 10,
    5: 10,
    6: 10,
    7: 10,
    8: 10,
    9: 10,
    10: 10,
    11: 10,
    12: 10,
}

# Phase ordering for one-hot encoding (only phases where agent acts)
_PHASE_ORDER = [
    TurnPhase.STARTING_FLIPS,
    TurnPhase.CHOOSE_DRAW,
    TurnPhase.HAVE_DRAWN_HIDDEN,
    TurnPhase.HAVE_DRAWN_OPEN,
    TurnPhase.HAVE_TO_FLIP_AFTER_DISCARD,
]


def normalize_card_value(value: float) -> float:
    return (value + 2) / 14.0


def expected_card_value(draw_pile_value_counts: Optional[List[int]]) -> float:
    if draw_pile_value_counts is not None:
        counts = np.asarray(draw_pile_value_counts, dtype=np.float32)
        if counts.shape == (len(CARD_VALUES),) and float(counts.sum()) > 0:
            values = np.asarray(CARD_VALUES, dtype=np.float32)
            return float(np.dot(values, counts) / counts.sum())

    total_cards = sum(INITIAL_CARD_COUNTS.values())
    return (
        sum(value * count for value, count in INITIAL_CARD_COUNTS.items()) / total_cards
    )


def _encode_grid(grid: Optional[List[List[Card]]], obs_vec: np.ndarray, offset: int):
    """Encode a 3×4 grid into obs_vec starting at offset.
    Per slot: (normalized_value, is_revealed).

    Three possible states per slot:
      - Face-down card:    (0.0, 0.0)  — value unknown, not revealed
      - Face-up card:      (normalized_value, 1.0) — real card value, revealed
      - Removed column:    (normalized(0), 1.0) — slot no longer in play;
            encoded as revealed with value 0 (neutral) so the agent sees it
            as a resolved/safe slot rather than an unknown one.
    """
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            idx = offset + (r * GRID_COLS + c) * 2
            if grid is not None and r < len(grid) and c < len(grid[r]):
                card = grid[r][c]
                if card is not None and card.face_up:
                    obs_vec[idx] = normalize_card_value(card.value)
                    obs_vec[idx + 1] = 1.0
            else:
                # Removed column → encoded as revealed with neutral value 0
                obs_vec[idx] = normalize_card_value(0)
                obs_vec[idx + 1] = 1.0


def _encode_draw_pile_value_counts(obs: Observation, obs_vec: np.ndarray, offset: int):
    """Encode normalized remaining count per value for draw pile cards (-2..12)."""
    if not obs.draw_pile_value_counts:
        return

    for i, value in enumerate(CARD_VALUES):
        if i >= len(obs.draw_pile_value_counts):
            break
        initial_count = INITIAL_CARD_COUNTS[value]
        obs_vec[offset + i] = obs.draw_pile_value_counts[i] / float(initial_count)


def encode_observation(obs: Observation) -> np.ndarray:
    """Encode an Observation into a flat float32 numpy array.

    Layout (75 dims):
      0-23:  Own grid (12 slots × 2: value, revealed)
      24-47: Opponent grid (12 slots × 2)
             Grid slot states: face-down → (0, 0); face-up → (norm_val, 1);
             removed column → (norm(0), 1) — treated as revealed neutral value.
      48-49: Discard top (value, has_card)
      50-51: Hand card (value, has_card)
      52-56: Turn phase one-hot (5 phases)
      57:    Draw pile size / 150
      58:    Final turn flag
      59:    Is first finisher flag
      60-74: Remaining draw-pile card counts per value -2..12,
             normalized by initial deck counts for each value
    """
    vec = np.zeros(OBS_SIZE, dtype=np.float32)

    # Own grid (0-23)
    _encode_grid(obs.card_grid, vec, 0)

    # Opponent grid (24-47)
    opponent_grid = next((g for g in obs.opponent_cards if g is not None), None)
    _encode_grid(opponent_grid, vec, 24)

    # Discard top (48-49)
    if obs.discard_top is not None:
        vec[48] = normalize_card_value(obs.discard_top.value)
        vec[49] = 1.0

    # Hand card (50-51)
    if obs.hand_card is not None:
        vec[50] = normalize_card_value(obs.hand_card.value)
        vec[51] = 1.0

    # Turn phase one-hot (52-56)
    for i, phase in enumerate(_PHASE_ORDER):
        if obs.turn_phase == phase:
            vec[52 + i] = 1.0
            break

    # Draw pile size (57)
    vec[57] = obs.draw_pile_size / 150.0

    # Final turn flag (58)
    vec[58] = 1.0 if obs.final_turn_phase else 0.0

    # Is first finisher (59)
    vec[59] = (
        1.0
        if (
            obs.first_finisher_id is not None and obs.first_finisher_id == obs.player_id
        )
        else 0.0
    )

    # Draw-pile card value distribution (60-74)
    _encode_draw_pile_value_counts(obs, vec, 60)

    return vec


def get_observation_space() -> spaces.Box:
    return spaces.Box(low=-0.5, high=1.5, shape=(OBS_SIZE,), dtype=np.float32)
