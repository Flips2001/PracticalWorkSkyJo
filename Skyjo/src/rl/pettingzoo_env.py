"""PettingZoo AEC Environment for Skyjo (2-player self-play via greenlet coroutines)."""

import functools
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from greenlet import greenlet
from typing import List, Optional

from Skyjo.src.skyjo_game import SkyjoGame
from Skyjo.src.players.player import Player
from Skyjo.src.action import Action
from Skyjo.src.observation import Observation
from Skyjo.src.rl.action_mapping import (
    NUM_ACTIONS,
    int_to_action,
    legal_actions_mask,
)
from Skyjo.src.rl.encoding import encode_observation, get_observation_space


class ProxyPlayer(Player):
    """A player that yields control back to the env greenlet when an action is needed."""

    def __init__(self, player_id: int, player_name: str):
        super().__init__(player_id, player_name)
        self.env_greenlet: Optional[greenlet] = None

    def select_action(
        self, observation: Observation, legal_actions: List[Action]
    ) -> Action:
        # Switch back to env greenlet, passing (player_id, obs, legal_actions)
        action = self.env_greenlet.switch((self.player_id, observation, legal_actions))
        return action


class SkyjoEnv(AECEnv):
    metadata = {"render_modes": [], "name": "skyjo_v0"}

    def __init__(self):
        super().__init__()
        self.possible_agents = ["player_0", "player_1"]
        self.agents = self.possible_agents[:]

        self._action_spaces = {
            agent: spaces.Discrete(NUM_ACTIONS) for agent in self.possible_agents
        }
        self._observation_spaces = {
            agent: spaces.Dict(
                {
                    "observation": get_observation_space(),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(NUM_ACTIONS,), dtype=np.int8
                    ),
                }
            )
            for agent in self.possible_agents
        }

        self.game: Optional[SkyjoGame] = None
        self._cumulative_rewards = {agent: 0.0 for agent in self.possible_agents}
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}

        self.agent_selection = None
        self._proxy_players: List[ProxyPlayer] = []
        self._game_greenlet: Optional[greenlet] = None
        self._current_obs = None
        self._current_legal_actions = None
        self._last_round_number = 1
        self._last_scores = [0, 0]

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def _agent_name(self, agent_id: int) -> str:
        return f"player_{agent_id}"

    def _run_game(self):
        """Game loop running inside its own greenlet."""
        self.game.play_game()

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._last_round_number = 1
        self._last_scores = [0, 0]

        # Create proxy players
        self._proxy_players = [
            ProxyPlayer(player_id=0, player_name="rl_0"),
            ProxyPlayer(player_id=1, player_name="rl_1"),
        ]

        # Set env greenlet reference so proxy players can switch back
        env_gr = greenlet.getcurrent()
        for p in self._proxy_players:
            p.env_greenlet = env_gr

        # Create game using existing framework
        self.game = SkyjoGame()
        for p in self._proxy_players:
            self.game.add_player(p)

        # Create game greenlet and start it
        self._game_greenlet = greenlet(self._run_game)
        result = self._game_greenlet.switch()

        if result is None:
            self._handle_game_over()
        else:
            player_id, obs, legal_actions = result
            self._current_obs = obs
            self._current_legal_actions = legal_actions
            self.agent_selection = self._agent_name(player_id)

    def _handle_game_over(self):
        """Assign terminal +1/-1 rewards based on who won."""
        scores = self.game.game_state.all_player_final_scores
        if scores and len(scores) >= 2:
            winner_id = scores.index(min(scores))
            loser_id = 1 - winner_id
            self.rewards[self._agent_name(winner_id)] += 1.0
            self.rewards[self._agent_name(loser_id)] -= 1.0
        self.terminations = {agent: True for agent in self.agents}

    def _check_round_reward(self):
        """Give shaped reward when a round completes based on round score difference."""
        current_round = self.game.game_state.round_number
        if current_round > self._last_round_number:
            scores = self.game.game_state.all_player_final_scores
            if scores and len(scores) >= 2:
                delta_0 = (scores[1] - self._last_scores[1]) - (
                    scores[0] - self._last_scores[0]
                )
                self.rewards["player_0"] += delta_0 / 50.0
                self.rewards["player_1"] -= delta_0 / 50.0
                self._last_scores = list(scores)
            self._last_round_number = current_round

    def observe(self, agent):
        if self._current_obs is not None and agent == self.agent_selection:
            encoded = encode_observation(self._current_obs)
            mask = legal_actions_mask(self._current_legal_actions)
        else:
            encoded = np.zeros(get_observation_space().shape, dtype=np.float32)
            mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
        return {"observation": encoded, "action_mask": mask}

    def step(self, action_int: int):
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action_int)
            return

        # Accumulate rewards from the previous step, then clear for this step
        self._accumulate_rewards()

        action = int_to_action(action_int)

        # Resume the game greenlet with the chosen action
        result = self._game_greenlet.switch(action)

        if result is None or self._game_greenlet.dead:
            self._handle_game_over()
            self._accumulate_rewards()
        else:
            player_id, obs, legal_actions = result
            self._current_obs = obs
            self._current_legal_actions = legal_actions
            self.agent_selection = self._agent_name(player_id)
            self._check_round_reward()
            self._accumulate_rewards()

    def _accumulate_rewards(self):
        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards[agent]
        self.rewards = {agent: 0.0 for agent in self.agents}

    def render(self):
        pass

    def close(self):
        pass
