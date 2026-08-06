"""Human-controlled player for the terminal game."""

from typing import List

from Skyjo.src.action import Action
from Skyjo.src.observation import Observation
from Skyjo.src.players.player import Player
from Skyjo.src.ui.terminal_game_ui import TerminalGameUI


class TerminalPlayer(Player):
    """Player adapter that delegates terminal interaction to ``TerminalGameUI``."""

    def __init__(
        self,
        player_id: int,
        player_name: str,
        ui: TerminalGameUI,
    ):
        super().__init__(player_id, player_name)
        self.ui = ui

    def select_action(
        self, observation: Observation, legal_actions: List[Action]
    ) -> Action:
        return self.ui.select_action(observation, legal_actions)
