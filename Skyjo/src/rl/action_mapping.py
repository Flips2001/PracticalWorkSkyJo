"""
Maps between the game's Action objects and a flat discrete action space.

Action space (27 actions total):
  0: DRAW_HIDDEN_CARD
  1: DRAW_OPEN_CARD
  2: DISCARD_CARD
  3-14: FLIP_CARD at positions (row, col) for 3x4 grid (row-major)
  15-26: SWAP_CARD at positions (row, col) for 3x4 grid (row-major)
"""

import numpy as np
from typing import List, Tuple

from Skyjo.src.action import Action
from Skyjo.src.action_type import ActionType

NUM_ACTIONS = 27
GRID_ROWS = 3
GRID_COLS = 4


def _pos_to_index(row: int, col: int) -> int:
    return row * GRID_COLS + col


def _index_to_pos(index: int) -> Tuple[int, int]:
    return index // GRID_COLS, index % GRID_COLS


def action_to_int(action: Action) -> int:
    match action.type:
        case ActionType.DRAW_HIDDEN_CARD:
            return 0
        case ActionType.DRAW_OPEN_CARD:
            return 1
        case ActionType.DISCARD_CARD:
            return 2
        case ActionType.FLIP_CARD:
            return 3 + _pos_to_index(*action.pos)
        case ActionType.SWAP_CARD:
            return 15 + _pos_to_index(*action.pos)
        case _:
            raise ValueError(f"Unknown action type: {action.type}")


def int_to_action(action_int: int) -> Action:
    if action_int == 0:
        return Action(ActionType.DRAW_HIDDEN_CARD)
    elif action_int == 1:
        return Action(ActionType.DRAW_OPEN_CARD)
    elif action_int == 2:
        return Action(ActionType.DISCARD_CARD)
    elif 3 <= action_int <= 14:
        pos = _index_to_pos(action_int - 3)
        return Action(ActionType.FLIP_CARD, pos=pos)
    elif 15 <= action_int <= 26:
        pos = _index_to_pos(action_int - 15)
        return Action(ActionType.SWAP_CARD, pos=pos)
    raise ValueError(f"Invalid action int: {action_int}")


def legal_actions_mask(legal_actions: List[Action]) -> np.ndarray:
    """Returns a binary mask of shape (NUM_ACTIONS,) indicating legal actions."""
    mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
    for action in legal_actions:
        mask[action_to_int(action)] = 1
    return mask
