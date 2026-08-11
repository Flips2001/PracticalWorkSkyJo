import numpy as np
import pytest

from Skyjo.src.action import Action
from Skyjo.src.action_type import ActionType
from Skyjo.src.card import Card
from Skyjo.src.rl.action_mapping import action_to_int
from Skyjo.src.rl.column_clear_drill_env import (
    DRILL_BAD_CLEAR_REWARD,
    DRILL_BUILD_DRAW_REWARD,
    DRILL_BUILD_SWAP_REWARD,
    DRILL_DRAW_REWARD,
    DRILL_SWAP_REWARD,
    ColumnClearDrillEnv,
    DrillMode,
    _hide_accidental_clears,
)
from Skyjo.src.rl.encoding import OBS_SIZE, normalize_card_value

ALL_POSITIONS = [(row, col) for row in range(3) for col in range(4)]


def _legal_actions(env: ColumnClearDrillEnv):
    return set(np.where(env.action_masks() == 1)[0])


def _env_and_obs(build_pair_prob: float, positive: bool):
    """Env reset onto the wanted scenario and target-value sign, plus its obs.

    Reseeding until the sign matches keeps the tests independent of how many
    random draws the generator happens to consume. Seeds are scanned in a fixed
    order, so repeated calls with the same arguments yield the same board.
    """
    env = ColumnClearDrillEnv(build_pair_prob=build_pair_prob)
    for seed in range(50):
        obs, _ = env.reset(seed=seed)
        if (env._target_value > 0) == positive:
            return env, obs
    raise AssertionError("no seed produced a target value with the wanted sign")


def _env(build_pair_prob: float, positive: bool) -> ColumnClearDrillEnv:
    return _env_and_obs(build_pair_prob, positive)[0]


def _pair_col(env: ColumnClearDrillEnv) -> int:
    """Column holding the revealed seed card of a BUILD_PAIR board."""
    if env._target_value > 0:
        cols = {col for _, col in env._reward_positions}
        assert len(cols) == 1
        return cols.pop()
    # Negative pairs are drilled the other way round: everything *but* the seed
    # column is rewarded.
    return next(col for _, col in set(ALL_POSITIONS) - env._reward_positions)


def _swap(env: ColumnClearDrillEnv, pos: tuple[int, int]):
    return env.step(action_to_int(Action(ActionType.SWAP_CARD, pos=pos)))


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


def test_reset_samples_both_scenarios():
    env = ColumnClearDrillEnv()
    modes = set()
    for seed in range(20):
        env.reset(seed=seed)
        modes.add(env._mode)

    assert modes == {DrillMode.CLEAR_COLUMN, DrillMode.BUILD_PAIR}


def test_build_pair_prob_selects_the_scenario():
    clear_env = ColumnClearDrillEnv(build_pair_prob=0.0)
    build_env = ColumnClearDrillEnv(build_pair_prob=1.0)

    for seed in range(5):
        clear_env.reset(seed=seed)
        build_env.reset(seed=seed)
        assert clear_env._mode is DrillMode.CLEAR_COLUMN
        assert build_env._mode is DrillMode.BUILD_PAIR


def test_draw_open_advances_to_swap_phase_with_positive_reward():
    env = _env(0.0, positive=True)

    obs, reward, terminated, truncated, info = env.step(
        action_to_int(Action(ActionType.DRAW_OPEN_CARD))
    )

    assert obs.shape == (OBS_SIZE,)
    assert reward == pytest.approx(DRILL_DRAW_REWARD)
    assert terminated is False
    assert truncated is False
    assert info == {}
    assert all(
        action_to_int(Action(ActionType.SWAP_CARD, pos=pos)) in _legal_actions(env)
        for pos in ALL_POSITIONS
    )


def test_draw_hidden_terminates_with_negative_reward():
    env = _env(0.0, positive=True)

    _, reward, terminated, truncated, _ = env.step(
        action_to_int(Action(ActionType.DRAW_HIDDEN_CARD))
    )

    assert reward == pytest.approx(-DRILL_DRAW_REWARD)
    assert terminated is True
    assert truncated is False


def test_target_swap_terminates_with_positive_reward():
    env = _env(0.0, positive=True)
    env.step(action_to_int(Action(ActionType.DRAW_OPEN_CARD)))

    _, reward, terminated, truncated, _ = _swap(env, next(iter(env._reward_positions)))

    assert reward == pytest.approx(DRILL_SWAP_REWARD)
    assert terminated is True
    assert truncated is False


def test_wrong_swap_terminates_with_negative_reward():
    env = _env(0.0, positive=True)
    env.step(action_to_int(Action(ActionType.DRAW_OPEN_CARD)))
    wrong_pos = next(pos for pos in ALL_POSITIONS if pos not in env._reward_positions)

    _, reward, terminated, truncated, _ = _swap(env, wrong_pos)

    assert reward == pytest.approx(-DRILL_SWAP_REWARD)
    assert terminated is True
    assert truncated is False


def test_bad_clear_draw_hidden_terminates_with_positive_reward():
    env = _env(0.0, positive=False)

    _, reward, terminated, truncated, _ = env.step(
        action_to_int(Action(ActionType.DRAW_HIDDEN_CARD))
    )

    assert env._target_value < 0
    assert reward == pytest.approx(DRILL_BAD_CLEAR_REWARD)
    assert terminated is True
    assert truncated is False


def test_bad_clear_draw_open_terminates_with_negative_reward():
    env = _env(0.0, positive=False)

    _, reward, terminated, truncated, _ = env.step(
        action_to_int(Action(ActionType.DRAW_OPEN_CARD))
    )

    assert env._target_value < 0
    assert reward == pytest.approx(-DRILL_BAD_CLEAR_REWARD)
    assert terminated is True
    assert truncated is False


def test_hide_accidental_clears_breaks_up_a_revealed_uniform_column():
    grid = [[Card(4, face_up=True) for _ in range(4)] for _ in range(3)]

    _hide_accidental_clears(np.random.default_rng(0), grid)

    assert all(any(not grid[row][col].face_up for row in range(3)) for col in range(4))


def test_boards_never_contain_an_already_clearable_column():
    """Such a column would have been removed by the engine before our turn."""
    env = ColumnClearDrillEnv()
    for seed in range(200):
        env.reset(seed=seed)
        for grid in (env._grid, env._opponent_grid):
            for col in range(4):
                cards = [row[col] for row in grid]
                revealed_values = {c.get_value() for c in cards if c.face_up}
                assert not (len(revealed_values) == 1 and all(c.face_up for c in cards))


def test_build_pair_board_shows_one_matching_card_on_a_mostly_covered_board():
    env = _env(1.0, positive=True)
    pair_col = _pair_col(env)

    revealed_matches = [
        (row, col)
        for row, col in ALL_POSITIONS
        if env._grid[row][col].face_up
        and env._grid[row][col].get_value() == env._target_value
    ]
    hidden_in_pair_col = {
        (row, pair_col) for row in range(3) if not env._grid[row][pair_col].face_up
    }

    assert len(revealed_matches) == 1
    assert revealed_matches[0][1] == pair_col
    assert hidden_in_pair_col == env._reward_positions


def test_build_pair_offers_the_matching_card_on_the_discard():
    env, obs = _env_and_obs(1.0, positive=True)

    assert obs[48] == pytest.approx(normalize_card_value(env._target_value))
    assert obs[49] == pytest.approx(1.0)


def test_build_pair_draw_open_hands_over_the_matching_card():
    env = _env(1.0, positive=True)

    obs, reward, terminated, _, _ = env.step(
        action_to_int(Action(ActionType.DRAW_OPEN_CARD))
    )

    assert obs[50] == pytest.approx(normalize_card_value(env._target_value))
    assert obs[51] == pytest.approx(1.0)
    assert reward == pytest.approx(DRILL_BUILD_DRAW_REWARD)
    assert terminated is False


def test_build_pair_draw_hidden_terminates_with_negative_reward():
    env = _env(1.0, positive=True)

    _, reward, terminated, _, _ = env.step(
        action_to_int(Action(ActionType.DRAW_HIDDEN_CARD))
    )

    assert reward == pytest.approx(-DRILL_BUILD_DRAW_REWARD)
    assert terminated is True


def test_build_pair_rewards_either_free_slot_of_the_pair_column():
    pair_col = _pair_col(_env(1.0, positive=True))
    slots = sorted(_env(1.0, positive=True)._reward_positions)

    assert slots == [(row, pair_col) for row, _ in slots]
    assert len(slots) == 2

    for pos in slots:
        # A swap ends the episode, so each slot needs its own (identical) board.
        env = _env(1.0, positive=True)
        env.step(action_to_int(Action(ActionType.DRAW_OPEN_CARD)))

        _, reward, terminated, _, _ = _swap(env, pos)

        assert reward == pytest.approx(DRILL_BUILD_SWAP_REWARD)
        assert terminated is True


def test_build_pair_swap_into_another_column_is_penalised():
    env = _env(1.0, positive=True)
    env.step(action_to_int(Action(ActionType.DRAW_OPEN_CARD)))
    other_pos = next(pos for pos in ALL_POSITIONS if pos not in env._reward_positions)

    _, reward, terminated, _, _ = _swap(env, other_pos)

    assert reward == pytest.approx(-DRILL_BUILD_SWAP_REWARD)
    assert terminated is True


def test_negative_build_pair_rewards_keeping_the_low_cards_apart():
    env = _env(1.0, positive=False)
    pair_col = _pair_col(env)
    env.step(action_to_int(Action(ActionType.DRAW_OPEN_CARD)))

    away_pos = next(pos for pos in ALL_POSITIONS if pos[1] != pair_col)
    _, reward, terminated, _, _ = _swap(env, away_pos)

    assert reward == pytest.approx(DRILL_BUILD_SWAP_REWARD)
    assert terminated is True


def test_negative_build_pair_penalises_completing_the_negative_column():
    env = _env(1.0, positive=False)
    pair_col = _pair_col(env)
    env.step(action_to_int(Action(ActionType.DRAW_OPEN_CARD)))

    _, reward, terminated, _, _ = _swap(env, (0, pair_col))

    assert reward == pytest.approx(-DRILL_BUILD_SWAP_REWARD)
    assert terminated is True


def test_negative_build_pair_still_rewards_taking_the_open_card():
    env = _env(1.0, positive=False)

    _, reward, terminated, _, _ = env.step(
        action_to_int(Action(ActionType.DRAW_OPEN_CARD))
    )

    assert reward == pytest.approx(DRILL_BUILD_DRAW_REWARD)
    assert terminated is False
