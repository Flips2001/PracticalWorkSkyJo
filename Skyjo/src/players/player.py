from abc import ABC, abstractmethod
from Skyjo.src.observation import Observation
from typing import Any, List

class Player(ABC):
    def __init__(self, player_id: int):
        self.player_id = player_id

    @abstractmethod
    def select_action(self, observation: Observation, legal_actions: List[Any]) -> Any: # TODO: Add actions type
        """
        Decide which action to take, given what this player observes
        and which actions are legal.
        """
        pass