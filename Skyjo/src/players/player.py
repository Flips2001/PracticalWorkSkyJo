from abc import ABC, abstractmethod
from Skyjo.src.observation import Observation
from Skyjo.src.action_type import ActionType
from Skyjo.src.player_state import PlayerState
from typing import List


class Player(ABC):
    def __init__(self, player_id: int, player_name: str):
        """
        Initialize a player with an ID and name.
        :param player_id:
        :param player_name:
        """
        self.player_id = player_id
        self.player_name = player_name
        self.player_state = PlayerState(player_id)

    @abstractmethod
    def select_action(
        self, observation: Observation, legal_actions: List[ActionType]
    ) -> ActionType:
        """
        Decide which action to take, given what this player observes
        and which actions are legal.
        :param observation: What this player can see of the game state
        :param legal_actions: List of legal actions this player can take
        :return: Selected action
        """
        pass
