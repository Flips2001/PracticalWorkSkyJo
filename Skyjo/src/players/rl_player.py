"""
RL Player that uses a trained MaskablePPO model to select actions.
"""

from typing import List, Optional

from sb3_contrib import MaskablePPO

from Skyjo.src.action import Action
from Skyjo.src.observation import Observation
from Skyjo.src.players.player import Player
from Skyjo.src.rl.action_mapping import int_to_action, legal_actions_mask
from Skyjo.src.rl.encoding import encode_observation
from Skyjo.src.rl.integrated_gradients import (
    ActionExplanation,
    explain_action,
    unavailable_explanation,
)


class RLPlayer(Player):
    def __init__(
        self,
        player_id: int,
        player_name: str,
        model_path: Optional[str] = None,
        model: Optional[MaskablePPO] = None,
        explain_moves: bool = False,
        explanation_steps: int = 32,
    ):
        super().__init__(player_id, player_name)
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = MaskablePPO.load(model_path)
        else:
            raise ValueError("Either model_path or model must be provided")
        self.explain_moves = explain_moves
        self.explanation_steps = explanation_steps
        self.last_explanation: Optional[ActionExplanation] = None

    def select_action(
        self, observation: Observation, legal_actions: List[Action]
    ) -> Action:
        obs_vec = encode_observation(observation)
        mask = legal_actions_mask(legal_actions)

        action_int, _ = self.model.predict(
            obs_vec,
            action_masks=mask,
            deterministic=True,
        )
        action = int_to_action(int(action_int))
        self.last_explanation = None
        if self.explain_moves:
            try:
                self.last_explanation = explain_action(
                    model=self.model,
                    observation=observation,
                    action=action,
                    legal_actions=legal_actions,
                    steps=self.explanation_steps,
                )
            except Exception as exc:
                self.last_explanation = unavailable_explanation(action, exc)
        return action
