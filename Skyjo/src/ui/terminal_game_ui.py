"""Terminal game controller and action-hook integration."""

from __future__ import annotations

import curses
import time
from typing import List, Optional, Sequence, TYPE_CHECKING

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
        player_names: Optional[Sequence[str]] = None,
        opponent_move_delay: float = 0.9,
    ):
        """
        :param player_id: seat of the human player (the viewer).
        :param player_name: the viewer's display name.
        :param opponent_name: name for the single opponent in a two-player game.
        :param player_names: names for every seat, indexed by player id. Pass
            this for games with three or more players; it supersedes
            ``opponent_name``.
        :param opponent_move_delay: seconds the board is shown after each
            opponent action. Without it the screen is only redrawn when it is the
            human's turn, so with two or more AI seats every AI move between two
            human turns lands at once and it is impossible to tell who did what.
            Set to 0 to redraw without pausing.
        """
        self.stdscr = stdscr
        self.player_id = player_id
        self.player_name = player_name
        self.opponent_name = opponent_name
        self.opponent_move_delay = opponent_move_delay
        if player_names is not None:
            self.player_names = list(player_names)
        else:
            # No per-seat names given: every other seat shares ``opponent_name``.
            # Sized off player_id so a viewer in seat 2+ still gets a valid list.
            self.player_names = [opponent_name] * max(2, player_id + 1)
            self.player_names[player_id] = player_name
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
                player_names=self.player_names,
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
        """Show the board after an opponent acts, so turns read as sequential."""
        if player.player_id == self.player_id:
            return

        observation = game.get_observation(self._viewer(game))
        if self.analyze_mode:
            self._show_analysis_pause(observation)
        else:
            self._show_opponent_move(observation, player, action)

    def _show_opponent_move(
        self, observation: Observation, player: "Player", action: Action
    ) -> None:
        """Draw one opponent's move and hold it briefly.

        Previously nothing was drawn here unless analyze mode was on, so the
        human only ever saw the board on their own turn -- with several AI seats
        their moves all appeared to happen at once.
        """
        self.renderer.render_game(
            observation=observation,
            player_names=self.player_names,
            legal_actions=[],
            selected_index=0,
            message=f"{player.player_name}: {action}",
            show_actions=False,
            help_text=self._help_text(),
        )
        if self.opponent_move_delay > 0:
            time.sleep(self.opponent_move_delay)

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
                player_names=self.player_names,
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
