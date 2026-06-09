import numpy as np
import pytest

from Skyjo.src.action import Action
from Skyjo.src.action_type import ActionType
from Skyjo.src.rl.action_mapping import action_to_int
from Skyjo.src.rl.column_clear_drill_env import (
    DRILL_BAD_CLEAR_REWARD,
    DRILL_DRAW_REWARD,
    DRILL_SWAP_REWARD,
    ColumnClearDrillEnv,
)
from Skyjo.src.rl.encoding import OBS_SIZE


def _legal_actions(env: ColumnClearDrillEnv):
    return set(np.where(env.action_masks() == 1)[0])


def test_reset_returns_draw_choice_with_draw_mask():
    env = ColumnClearDrillEnv()

    obs, info = env.reset(seed=1)

    assert info == {}
    assert obs.shape == (OBS_SIZE,)
    assert _legal_actions(env) == {
        action_to_int(Action(ActionType.DRAW_HIDDEN_CARD)),
        action_to_int(Action(ActionType.DRAW_OPEN_CARD)),
    }


def test_reset_samples_different_target_values():
    values = set()
    for seed in range(8):
        env = ColumnClearDrillEnv()
        env.reset(seed=seed)
        values.add(env._target_value)

    assert len(values) > 1


def test_draw_open_advances_to_swap_phase_with_positive_reward():
    env = ColumnClearDrillEnv()
    env.reset(seed=1)

    obs, reward, terminated, truncated, info = env.step(
        action_to_int(Action(ActionType.DRAW_OPEN_CARD))
    )

    assert obs.shape == (OBS_SIZE,)
    assert env._target_value > 0
    assert reward == pytest.approx(DRILL_DRAW_REWARD)
    assert terminated is False
    assert truncated is False
    assert info == {}
    assert all(
        action_to_int(Action(ActionType.SWAP_CARD, pos=(row, col)))
        in _legal_actions(env)
        for row in range(3)
        for col in range(4)
    )


def test_draw_hidden_terminates_with_negative_reward():
    env = ColumnClearDrillEnv()
    env.reset(seed=1)

    _, reward, terminated, truncated, _ = env.step(
        action_to_int(Action(ActionType.DRAW_HIDDEN_CARD))
    )

    assert env._target_value > 0
    assert reward == pytest.approx(-DRILL_DRAW_REWARD)
    assert terminated is True
    assert truncated is False


def test_target_swap_terminates_with_positive_reward():
    env = ColumnClearDrillEnv()
    env.reset(seed=1)
    env.step(action_to_int(Action(ActionType.DRAW_OPEN_CARD)))

    _, reward, terminated, truncated, _ = env.step(
        action_to_int(Action(ActionType.SWAP_CARD, pos=env._target_pos))
    )

    assert env._target_value > 0
    assert reward == pytest.approx(DRILL_SWAP_REWARD)
    assert terminated is True
    assert truncated is False


def test_wrong_swap_terminates_with_negative_reward():
    env = ColumnClearDrillEnv()
    env.reset(seed=1)
    env.step(action_to_int(Action(ActionType.DRAW_OPEN_CARD)))
    wrong_pos = next(
        (row, col)
        for row in range(3)
        for col in range(4)
        if (row, col) != env._target_pos
    )

    _, reward, terminated, truncated, _ = env.step(
        action_to_int(Action(ActionType.SWAP_CARD, pos=wrong_pos))
    )

    assert env._target_value > 0
    assert reward == pytest.approx(-DRILL_SWAP_REWARD)
    assert terminated is True
    assert truncated is False


def test_bad_clear_draw_hidden_terminates_with_positive_reward():
    env = ColumnClearDrillEnv()
    env.reset(seed=5)

    _, reward, terminated, truncated, _ = env.step(
        action_to_int(Action(ActionType.DRAW_HIDDEN_CARD))
    )

    assert env._target_value == -2
    assert reward == pytest.approx(DRILL_BAD_CLEAR_REWARD)
    assert terminated is True
    assert truncated is False


def test_bad_clear_draw_open_terminates_with_negative_reward():
    env = ColumnClearDrillEnv()
    env.reset(seed=5)

    _, reward, terminated, truncated, _ = env.step(
        action_to_int(Action(ActionType.DRAW_OPEN_CARD))
    )

    assert env._target_value == -2
    assert reward == pytest.approx(-DRILL_BAD_CLEAR_REWARD)
    assert terminated is True
    assert truncated is False
