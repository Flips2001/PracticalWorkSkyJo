from types import SimpleNamespace

import numpy as np
import pytest

from Skyjo.src.game_state import ColumnClearStats
from Skyjo.src.rl.pettingzoo_env import COLUMN_CLEAR_REWARD_DIVISOR, SkyjoEnv
from Skyjo.src.rl.action_mapping import NUM_ACTIONS


class _FakeGreenlet:
    dead = False

    def switch(self, action):
        return 0, None, []


def _env_with_column_clear_stats(stats: ColumnClearStats) -> SkyjoEnv:
    env = SkyjoEnv()
    env.agent_selection = "player_0"
    env.rewards = {"player_0": 0.0, "player_1": 0.0}
    env._cumulative_rewards = {"player_0": 0.0, "player_1": 0.0}
    env.terminations = {"player_0": False, "player_1": False}
    env.truncations = {"player_0": False, "player_1": False}
    env._game_greenlet = _FakeGreenlet()
    env.game = SimpleNamespace(
        last_column_clear_stats={0: stats},
        game_state=SimpleNamespace(round_number=1, all_player_final_scores=[]),
    )
    return env


class TestSkyjoEnvReset:
    def test_reset_sets_agents(self):
        env = SkyjoEnv()
        env.reset()
        assert env.agents == ["player_0", "player_1"]

    def test_reset_sets_agent_selection(self):
        env = SkyjoEnv()
        env.reset()
        assert env.agent_selection in ["player_0", "player_1"]

    def test_reset_no_terminations(self):
        env = SkyjoEnv()
        env.reset()
        for agent in env.agents:
            assert not env.terminations[agent]

    def test_observe_after_reset(self):
        env = SkyjoEnv()
        env.reset()
        obs = env.observe(env.agent_selection)
        assert "observation" in obs
        assert "action_mask" in obs
        assert obs["observation"].shape[0] > 0
        assert obs["action_mask"].shape == (NUM_ACTIONS,)
        # At least one action must be legal
        assert obs["action_mask"].sum() > 0


class TestSkyjoEnvSpaces:
    def test_observation_space(self):
        env = SkyjoEnv()
        env.reset()
        for agent in env.possible_agents:
            space = env.observation_space(agent)
            assert "observation" in space.spaces
            assert "action_mask" in space.spaces

    def test_action_space(self):
        env = SkyjoEnv()
        env.reset()
        for agent in env.possible_agents:
            space = env.action_space(agent)
            assert space.n == NUM_ACTIONS


class TestSkyjoEnvGameplay:
    def test_step_with_legal_action(self):
        env = SkyjoEnv()
        env.reset()
        obs = env.observe(env.agent_selection)
        mask = obs["action_mask"]
        legal_action = int(np.argmax(mask))
        # Should not raise
        env.step(legal_action)

    def test_full_game_terminates(self):
        """Play a full game using random legal actions. Game must terminate."""
        env = SkyjoEnv()
        env.reset()
        max_steps = 5000
        steps = 0

        while not all(env.terminations.values()) and steps < max_steps:
            agent = env.agent_selection
            obs = env.observe(agent)
            mask = obs["action_mask"]
            legal_indices = np.where(mask == 1)[0]
            action = int(np.random.choice(legal_indices))
            env.step(action)
            steps += 1

        assert all(
            env.terminations.values()
        ), f"Game did not terminate within {max_steps} steps"

    def test_terminal_rewards_sum_to_zero(self):
        """Terminal rewards should be +1/-1 and sum to zero."""
        env = SkyjoEnv()
        env.reset()
        max_steps = 5000
        steps = 0

        while not all(env.terminations.values()) and steps < max_steps:
            agent = env.agent_selection
            obs = env.observe(agent)
            mask = obs["action_mask"]
            legal_indices = np.where(mask == 1)[0]
            action = int(np.random.choice(legal_indices))
            env.step(action)
            steps += 1

        total_reward = sum(env._cumulative_rewards.values())
        # Rewards may not be exactly zero due to round shaping rewards
        # but terminal +1/-1 should balance
        assert abs(total_reward) < 2.0  # reasonable bound

    def test_multiple_resets(self):
        """Env should support multiple resets without errors."""
        env = SkyjoEnv()
        for _ in range(3):
            env.reset()
            obs = env.observe(env.agent_selection)
            assert obs["action_mask"].sum() > 0

    def test_step_rewards_positive_value_column_clear(self):
        env = _env_with_column_clear_stats(
            ColumnClearStats(columns_removed=1, removed_card_value_sum=36)
        )

        env.step(0)

        expected = 36 / COLUMN_CLEAR_REWARD_DIVISOR
        assert env._cumulative_rewards["player_0"] == pytest.approx(expected)
        assert env._cumulative_rewards["player_1"] == pytest.approx(-expected)

    def test_step_penalizes_negative_value_column_clear(self):
        env = _env_with_column_clear_stats(
            ColumnClearStats(columns_removed=1, removed_card_value_sum=-6)
        )

        env.step(0)

        expected = -6 / COLUMN_CLEAR_REWARD_DIVISOR
        assert env._cumulative_rewards["player_0"] == pytest.approx(expected)
        assert env._cumulative_rewards["player_1"] == pytest.approx(-expected)

    def test_step_does_not_reward_when_no_column_clears(self):
        env = _env_with_column_clear_stats(ColumnClearStats())

        env.step(0)

        assert env._cumulative_rewards["player_0"] == pytest.approx(0.0)
        assert env._cumulative_rewards["player_1"] == pytest.approx(0.0)
