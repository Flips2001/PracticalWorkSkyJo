"""Single-agent self-play wrapper for the 2-player Skyjo PettingZoo environment."""

import os
from typing import Optional

import numpy as np
import torch
from gymnasium import Env, spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from Skyjo.src.rl.action_mapping import NUM_ACTIONS
from Skyjo.src.rl.encoding import OBS_SIZE
from Skyjo.src.rl.pettingzoo_env import SkyjoEnv

_opponent_model: Optional[MaskablePPO] = None
_reset_counter = 0
_best_model_path: Optional[str] = None
_device = "cpu"


def _configure_opponent_loader(best_model_path: str, device: str) -> None:
    """Configure subprocess-local state for loading the current best opponent model."""
    global _best_model_path, _device
    _best_model_path = best_model_path
    _device = device


def _load_opponent_if_needed() -> None:
    """Load best opponent model from disk in subprocess (checked every 20 resets)."""
    global _opponent_model, _reset_counter
    _reset_counter += 1
    if _reset_counter % 20 != 0:
        return

    if _best_model_path is None:
        return

    try:
        if os.path.exists(_best_model_path + ".zip"):
            _opponent_model = MaskablePPO.load(_best_model_path, device=_device)
    except Exception:
        # Keep training robust if a checkpoint is being written concurrently.
        pass


def _get_opponent_action(env: SkyjoEnv) -> int:
    """Get action from opponent: 90% self-play model, 10% random."""
    obs_dict = env.observe(env.agent_selection)
    mask = obs_dict["action_mask"]

    # 10% random for exploration.
    if _reset_counter % 10 == 0:
        legal = np.where(mask == 1)[0]
        return int(np.random.choice(legal))

    # 90% self-play model (falls back to random if no model loaded yet).
    if _opponent_model is not None:
        obs_t = torch.as_tensor(obs_dict["observation"], dtype=torch.float32).unsqueeze(
            0
        )
        mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
        with torch.no_grad():
            dist = _opponent_model.policy.get_distribution(obs_t)
            dist.apply_masking(mask_t)
            action = dist.sample()
        return int(action.item())

    legal = np.where(mask == 1)[0]
    return int(np.random.choice(legal))


def _play_opponent_turns(env: SkyjoEnv) -> None:
    """Play all opponent (player_1) turns until player_0's turn or game over."""
    while env.agent_selection == "player_1" and not all(env.terminations.values()):
        env.step(_get_opponent_action(env))


class SkyjoSelfPlayWrapper(Env):
    """Wrap the 2-player AEC env into a single-agent Gymnasium env for player_0."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.env = SkyjoEnv()
        self.observation_space = spaces.Box(
            low=-0.5, high=1.5, shape=(OBS_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self._current_mask = np.ones(NUM_ACTIONS, dtype=np.int8)

    def reset(self, seed=None, options=None):
        _load_opponent_if_needed()
        self.env.reset(seed=seed)
        _play_opponent_turns(self.env)

        if all(self.env.terminations.values()):
            return np.zeros(OBS_SIZE, dtype=np.float32), {}

        obs = self.env.observe(self.env.agent_selection)
        self._current_mask = obs["action_mask"]
        return obs["observation"], {}

    def step(self, action):
        self.env.step(action)

        if all(self.env.terminations.values()):
            reward = self.env._cumulative_rewards.get("player_0", 0.0)
            self.env._cumulative_rewards["player_0"] = 0.0
            return np.zeros(OBS_SIZE, dtype=np.float32), reward, True, False, {}

        _play_opponent_turns(self.env)

        if all(self.env.terminations.values()):
            reward = self.env._cumulative_rewards.get("player_0", 0.0)
            self.env._cumulative_rewards["player_0"] = 0.0
            return np.zeros(OBS_SIZE, dtype=np.float32), reward, True, False, {}

        obs = self.env.observe(self.env.agent_selection)
        self._current_mask = obs["action_mask"]
        reward = self.env._cumulative_rewards.get("player_0", 0.0)
        self.env._cumulative_rewards["player_0"] = 0.0
        return obs["observation"], reward, False, False, {}

    def action_masks(self) -> np.ndarray:
        return self._current_mask


def mask_fn(env: Env) -> np.ndarray:
    return getattr(env, "action_masks")()


def make_env(best_model_path: str, device: str = "cpu"):
    """Create a subprocess-safe single-agent self-play env factory."""

    def _init():
        _configure_opponent_loader(best_model_path=best_model_path, device=device)
        env = SkyjoSelfPlayWrapper()
        return ActionMasker(env, mask_fn)

    return _init
