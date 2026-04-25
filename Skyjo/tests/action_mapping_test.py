import numpy as np
import pytest

from Skyjo.src.action import Action
from Skyjo.src.action_type import ActionType
from Skyjo.src.rl.action_mapping import (
    NUM_ACTIONS,
    action_to_int,
    int_to_action,
    legal_actions_mask,
)


class TestActionToInt:
    def test_draw_hidden(self):
        assert action_to_int(Action(ActionType.DRAW_HIDDEN_CARD)) == 0

    def test_draw_open(self):
        assert action_to_int(Action(ActionType.DRAW_OPEN_CARD)) == 1

    def test_discard(self):
        assert action_to_int(Action(ActionType.DISCARD_CARD)) == 2

    def test_flip_card_first_position(self):
        assert action_to_int(Action(ActionType.FLIP_CARD, pos=(0, 0))) == 3

    def test_flip_card_last_position(self):
        assert action_to_int(Action(ActionType.FLIP_CARD, pos=(2, 3))) == 14

    def test_swap_card_first_position(self):
        assert action_to_int(Action(ActionType.SWAP_CARD, pos=(0, 0))) == 15

    def test_swap_card_last_position(self):
        assert action_to_int(Action(ActionType.SWAP_CARD, pos=(2, 3))) == 26


class TestIntToAction:
    def test_draw_hidden(self):
        action = int_to_action(0)
        assert action.type == ActionType.DRAW_HIDDEN_CARD
        assert action.pos is None

    def test_draw_open(self):
        action = int_to_action(1)
        assert action.type == ActionType.DRAW_OPEN_CARD

    def test_discard(self):
        action = int_to_action(2)
        assert action.type == ActionType.DISCARD_CARD

    def test_flip_positions(self):
        action = int_to_action(3)
        assert action.type == ActionType.FLIP_CARD
        assert action.pos == (0, 0)

        action = int_to_action(7)
        assert action.type == ActionType.FLIP_CARD
        assert action.pos == (1, 0)

    def test_swap_positions(self):
        action = int_to_action(15)
        assert action.type == ActionType.SWAP_CARD
        assert action.pos == (0, 0)

        action = int_to_action(19)
        assert action.type == ActionType.SWAP_CARD
        assert action.pos == (1, 0)

    def test_invalid_action_int(self):
        with pytest.raises(ValueError):
            int_to_action(27)

        with pytest.raises(ValueError):
            int_to_action(-1)


class TestRoundtrip:
    """action_to_int and int_to_action should be inverses of each other."""

    def test_roundtrip_all_actions(self):
        for i in range(NUM_ACTIONS):
            action = int_to_action(i)
            assert action_to_int(action) == i

    def test_roundtrip_specific_actions(self):
        actions = [
            Action(ActionType.DRAW_HIDDEN_CARD),
            Action(ActionType.DRAW_OPEN_CARD),
            Action(ActionType.DISCARD_CARD),
            Action(ActionType.FLIP_CARD, pos=(1, 2)),
            Action(ActionType.SWAP_CARD, pos=(2, 1)),
        ]
        for action in actions:
            assert int_to_action(action_to_int(action)) == action


class TestLegalActionsMask:
    def test_empty_legal_actions(self):
        mask = legal_actions_mask([])
        assert mask.shape == (NUM_ACTIONS,)
        assert mask.sum() == 0

    def test_single_legal_action(self):
        mask = legal_actions_mask([Action(ActionType.DRAW_OPEN_CARD)])
        assert mask[1] == 1
        assert mask.sum() == 1

    def test_multiple_legal_actions(self):
        actions = [
            Action(ActionType.DRAW_HIDDEN_CARD),
            Action(ActionType.DRAW_OPEN_CARD),
        ]
        mask = legal_actions_mask(actions)
        assert mask[0] == 1
        assert mask[1] == 1
        assert mask.sum() == 2

    def test_mask_dtype(self):
        mask = legal_actions_mask([Action(ActionType.DISCARD_CARD)])
        assert mask.dtype == np.int8

    def test_flip_and_swap_positions_in_mask(self):
        actions = [
            Action(ActionType.FLIP_CARD, pos=(0, 0)),
            Action(ActionType.SWAP_CARD, pos=(2, 3)),
        ]
        mask = legal_actions_mask(actions)
        assert mask[3] == 1  # flip (0,0)
        assert mask[26] == 1  # swap (2,3)
        assert mask.sum() == 2
