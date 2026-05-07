"""
Terminal-based interactive player using curses for Skyjo.
Replaces the old print/input HumanPlayer with a modern TUI.
"""

import curses
from typing import List

from Skyjo.src.action import Action
from Skyjo.src.observation import Observation
from Skyjo.src.players.player import Player
from Skyjo.src.ui.terminal_ui import TerminalRenderer


class TerminalPlayer(Player):
    """Interactive player using curses-based terminal UI."""

    def __init__(self, player_id: int, player_name: str, stdscr, opponent_name: str = "Opponent"):
        super().__init__(player_id, player_name)
        self.stdscr = stdscr
        self.renderer = TerminalRenderer(stdscr)
        self.opponent_name = opponent_name
        self._message = ""

    def select_action(
        self, observation: Observation, legal_actions: List[Action]
    ) -> Action:
        """
        Display the game state and let the player select an action
        using arrow keys or number keys.
        """
        if not legal_actions:
            raise ValueError("No legal actions available to select from.")

        selected_index = 0
        num_actions = len(legal_actions)

        while True:
            # Render game state with current selection
            self.renderer.render_game(
                observation=observation,
                player_name=self.player_name,
                opponent_name=self.opponent_name,
                legal_actions=legal_actions,
                selected_index=selected_index,
                message=self._message,
            )

            # Get user input
            key = self.stdscr.getch()

            if key == curses.KEY_UP:
                selected_index = (selected_index - 1) % num_actions
                self._message = ""
            elif key == curses.KEY_DOWN:
                selected_index = (selected_index + 1) % num_actions
                self._message = ""
            elif key == curses.KEY_LEFT:
                selected_index = max(0, selected_index - 1)
                self._message = ""
            elif key == curses.KEY_RIGHT:
                selected_index = min(num_actions - 1, selected_index + 1)
                self._message = ""
            elif key in (curses.KEY_ENTER, 10, 13):
                # Enter key - confirm selection
                self._message = ""
                return legal_actions[selected_index]
            elif key == ord('q') or key == ord('Q'):
                raise KeyboardInterrupt("Player quit the game")
            elif ord('1') <= key <= ord('9'):
                # Number key quick select
                num = key - ord('0')
                if 1 <= num <= num_actions:
                    self._message = ""
                    return legal_actions[num - 1]
                else:
                    self._message = f"Invalid: press 1-{num_actions}"
            else:
                self._message = ""

    def show_round_summary(self, scores: List[int], player_names: List[str], round_num: int):
        """Show round summary screen."""
        self.renderer.render_round_summary(scores, player_names, round_num)

    def show_game_over(self, scores: List[int], player_names: List[str]):
        """Show game over screen."""
        self.renderer.render_game_over(scores, player_names)
