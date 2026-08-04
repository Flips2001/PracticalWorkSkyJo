"""Terminal game controller and action-hook integration."""

from __future__ import annotations

import curses
from typing import List, TYPE_CHECKING

from Skyjo.src.action import Action
from Skyjo.src.observation import Observation
from Skyjo.src.ui.terminal_ui import TerminalRenderer

if TYPE_CHECKING:
    from Skyjo.src.players.player import Player
    from Skyjo.src.skyjo_game import SkyjoGame


class TerminalGameUI:
    """Owns terminal input, rendering, and optional move analysis state."""

    def __init__(
        self,
        stdscr,
        player_id: int,
        player_name: str,
        opponent_name: str = "Opponent",
    ):
        self.stdscr = stdscr
        self.player_id = player_id
        self.player_name = player_name
        self.opponent_name = opponent_name
        self.renderer = TerminalRenderer(stdscr)
        self.analyze_mode = False
        self._message = ""
        self._opponent_last_action = ""
        self._opponent_explanation = None
        self._opponent_snapshot = None

    def select_action(
        self, observation: Observation, legal_actions: List[Action]
    ) -> Action:
        """Render the game and return the action selected by the user."""
        if not legal_actions:
            raise ValueError("No legal actions available to select from.")

        selected_index = 0
        num_actions = len(legal_actions)

        while True:
            self.renderer.render_game(
                observation=observation,
                player_name=self.player_name,
                opponent_name=self.opponent_name,
                legal_actions=legal_actions,
                selected_index=selected_index,
                message=self._message,
                opponent_last_action=(
                    self._opponent_last_action if self.analyze_mode else ""
                ),
                opponent_explanation=(
                    self._opponent_explanation if self.analyze_mode else None
                ),
                opponent_snapshot=(
                    self._opponent_snapshot if self.analyze_mode else None
                ),
                help_text=self._help_text(),
            )

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
                self._message = ""
                return legal_actions[selected_index]
            elif key in (ord("a"), ord("A")):
                self.analyze_mode = not self.analyze_mode
                self._message = (
                    f"Analyze mode {'enabled' if self.analyze_mode else 'disabled'}."
                )
            elif key in (ord("q"), ord("Q")):
                raise KeyboardInterrupt("Player quit the game")
            else:
                self._message = ""

    def before_action(
        self, game: "SkyjoGame", player: "Player", action: Action
    ) -> None:
        """Capture the viewer's immutable decision-time state."""
        if player.player_id == self.player_id:
            return

        self._opponent_last_action = f"{player.player_name}: {action}"
        self._opponent_explanation = getattr(player, "last_explanation", None)
        self._opponent_snapshot = game.get_observation(self._viewer(game))

    def after_action(self, game: "SkyjoGame", player: "Player", action: Action) -> None:
        """Optionally pause on the viewer's post-action state."""
        if player.player_id == self.player_id or not self.analyze_mode:
            return

        self._show_analysis_pause(game.get_observation(self._viewer(game)))

    def _viewer(self, game: "SkyjoGame") -> "Player":
        try:
            return next(
                player for player in game.players if player.player_id == self.player_id
            )
        except StopIteration as exc:
            raise RuntimeError(
                "Terminal player is not registered with the game"
            ) from exc

    def _help_text(self) -> str:
        analyze = "ON" if self.analyze_mode else "OFF"
        return f" ↑↓ Navigate  │  Enter Select  │  a Analyze: {analyze}  │  q Quit "

    def _show_analysis_pause(self, observation: Observation) -> None:
        while True:
            self.renderer.render_game(
                observation=observation,
                player_name=self.player_name,
                opponent_name=self.opponent_name,
                legal_actions=[],
                selected_index=0,
                message="Analyze mode: press Enter to continue.",
                opponent_last_action=self._opponent_last_action,
                opponent_explanation=self._opponent_explanation,
                opponent_snapshot=self._opponent_snapshot,
                show_actions=False,
                help_text=" Enter Continue  │  a Analyze: ON  │  q Quit ",
            )
            key = self.stdscr.getch()
            if key in (curses.KEY_ENTER, 10, 13):
                return
            if key in (ord("a"), ord("A")):
                self.analyze_mode = False
                return
            if key in (ord("q"), ord("Q")):
                raise KeyboardInterrupt("Player quit the game")

    def show_round_summary(
        self, scores: List[int], player_names: List[str], round_num: int
    ) -> None:
        self.renderer.render_round_summary(scores, player_names, round_num)

    def show_game_over(self, scores: List[int], player_names: List[str]) -> None:
        self.renderer.render_game_over(scores, player_names)
