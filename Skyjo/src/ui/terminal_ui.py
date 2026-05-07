"""
Terminal UI renderer for Skyjo using curses.
Provides a full-screen, color-coded game display with in-place updates.
"""

import curses
from typing import List, Optional

from Skyjo.src.card import Card
from Skyjo.src.observation import Observation
from Skyjo.src.turn_phase import TurnPhase


# Color pair IDs
COLOR_DEFAULT = 0
COLOR_NEGATIVE = 1  # -2, -1 cards (green - good)
COLOR_LOW = 2  # 0-4 cards (cyan)
COLOR_MID = 3  # 5-8 cards (yellow)
COLOR_HIGH = 4  # 9-12 cards (red - bad)
COLOR_HIDDEN = 5  # face-down cards (white on blue)
COLOR_TITLE = 6  # titles and headers
COLOR_SELECTED = 7  # currently selected item
COLOR_DISCARD = 8  # discard pile
COLOR_PHASE = 9  # phase info
COLOR_SCORE = 10  # score display


def init_colors():
    """Initialize color pairs for the terminal UI."""
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(COLOR_NEGATIVE, curses.COLOR_GREEN, -1)
    curses.init_pair(COLOR_LOW, curses.COLOR_CYAN, -1)
    curses.init_pair(COLOR_MID, curses.COLOR_YELLOW, -1)
    curses.init_pair(COLOR_HIGH, curses.COLOR_RED, -1)
    curses.init_pair(COLOR_HIDDEN, curses.COLOR_WHITE, curses.COLOR_BLUE)
    curses.init_pair(COLOR_TITLE, curses.COLOR_WHITE, -1)
    curses.init_pair(COLOR_SELECTED, curses.COLOR_BLACK, curses.COLOR_WHITE)
    curses.init_pair(COLOR_DISCARD, curses.COLOR_MAGENTA, -1)
    curses.init_pair(COLOR_PHASE, curses.COLOR_CYAN, -1)
    curses.init_pair(COLOR_SCORE, curses.COLOR_YELLOW, -1)


def get_card_color(card: Card) -> int:
    """Get the color pair for a card based on its value."""
    if not card.face_up:
        return COLOR_HIDDEN
    if card.value < 0:
        return COLOR_NEGATIVE
    elif card.value <= 4:
        return COLOR_LOW
    elif card.value <= 8:
        return COLOR_MID
    else:
        return COLOR_HIGH


def format_card(card: Card) -> str:
    """Format a card for display."""
    if not card.face_up:
        return " ?? "
    value = card.value
    if value < 0:
        return f"{value:3d} "
    elif value < 10:
        return f"  {value} "
    else:
        return f" {value} "


PHASE_DESCRIPTIONS = {
    TurnPhase.STARTING_FLIPS: "🎴 Choose cards to flip face-up",
    TurnPhase.CHOOSE_DRAW: "🃏 Draw from pile or discard",
    TurnPhase.HAVE_DRAWN_HIDDEN: "🤔 Swap drawn card or discard it",
    TurnPhase.HAVE_DRAWN_OPEN: "🔄 Choose where to place this card",
    TurnPhase.HAVE_TO_FLIP_AFTER_DISCARD: "👆 Flip a face-down card",
    TurnPhase.END_TURN: "✅ Turn complete",
}


class TerminalRenderer:
    """Renders the Skyjo game state using curses."""

    def __init__(self, stdscr):
        self.stdscr = stdscr
        init_colors()
        curses.curs_set(0)  # Hide cursor
        self.stdscr.keypad(True)

    def render_game(
        self,
        observation: Observation,
        player_name: str,
        opponent_name: str,
        legal_actions: List,
        selected_index: int,
        message: str = "",
    ):
        """Render the full game state."""
        self.stdscr.erase()
        max_y, max_x = self.stdscr.getmaxyx()

        row = 0
        # Title bar
        title = "═══════════════════════  S K Y J O  ═══════════════════════"
        self._safe_addstr(row, max(0, (max_x - len(title)) // 2), title,
                          curses.color_pair(COLOR_TITLE) | curses.A_BOLD)
        row += 2

        # Phase info
        phase_text = PHASE_DESCRIPTIONS.get(observation.turn_phase, str(observation.turn_phase))
        self._safe_addstr(row, 2, f"Phase: {phase_text}",
                          curses.color_pair(COLOR_PHASE) | curses.A_BOLD)
        row += 1

        # Final turn warning
        if observation.final_turn_phase:
            self._safe_addstr(row, 2, "⚠️  FINAL TURN! ⚠️",
                              curses.color_pair(COLOR_HIGH) | curses.A_BOLD | curses.A_BLINK)
            row += 1
        row += 1

        # Game info bar (discard + hand + draw pile)
        self._render_game_info(row, observation)
        row += 3

        # Two grids side by side
        grid_start_row = row
        self._render_player_grid(grid_start_row, 2, player_name, observation.card_grid,
                                 observation.scores[observation.player_id], is_self=True)

        # Opponent grid on the right side
        opponent_col = 40
        if observation.opponent_cards:
            for i, opp_grid in enumerate(observation.opponent_cards):
                if opp_grid is not None:
                    opp_score = observation.scores[i] if i < len(observation.scores) else 0
                    self._render_player_grid(grid_start_row, opponent_col, opponent_name,
                                             opp_grid, opp_score, is_self=False)
                    break

        row = grid_start_row + 10

        # Action selection area
        self._render_actions(row, legal_actions, selected_index)

        # Status message
        if message:
            msg_row = max_y - 2 if max_y - 2 > row + len(legal_actions) + 2 else row + len(legal_actions) + 3
            self._safe_addstr(msg_row, 2, message, curses.color_pair(COLOR_TITLE))

        # Help bar at bottom
        help_text = " ↑↓ Navigate  │  Enter Select  │  1-9 Quick Select  │  q Quit "
        help_row = max_y - 1
        self._safe_addstr(help_row, max(0, (max_x - len(help_text)) // 2), help_text,
                          curses.color_pair(COLOR_TITLE) | curses.A_DIM)

        self.stdscr.refresh()

    def _render_game_info(self, row: int, observation: Observation):
        """Render discard pile, hand card, and draw pile info."""
        col = 2

        # Draw pile
        self._safe_addstr(row, col, "Draw Pile: ", curses.color_pair(COLOR_TITLE))
        draw_text = f"[{observation.draw_pile_size} cards]"
        self._safe_addstr(row, col + 11, draw_text, curses.color_pair(COLOR_HIDDEN))

        # Discard pile
        col = 30
        self._safe_addstr(row, col, "Discard: ", curses.color_pair(COLOR_TITLE))
        if observation.discard_top:
            card_str = format_card(observation.discard_top)
            color = get_card_color(observation.discard_top)
            self._safe_addstr(row, col + 9, f"[{card_str.strip()}]",
                              curses.color_pair(color) | curses.A_BOLD)
        else:
            self._safe_addstr(row, col + 9, "[empty]", curses.color_pair(COLOR_DEFAULT))

        # Hand card
        col = 52
        self._safe_addstr(row, col, "Hand: ", curses.color_pair(COLOR_TITLE))
        if observation.hand_card:
            card_str = format_card(observation.hand_card)
            color = get_card_color(observation.hand_card)
            self._safe_addstr(row, col + 6, f"[{card_str.strip()}]",
                              curses.color_pair(color) | curses.A_BOLD)
        else:
            self._safe_addstr(row, col + 6, "[-]", curses.color_pair(COLOR_DEFAULT) | curses.A_DIM)

    def _render_player_grid(
        self, start_row: int, start_col: int, name: str,
        grid: List[List[Card]], score: int, is_self: bool
    ):
        """Render a player's card grid."""
        row = start_row

        # Player name and score
        header = f"{'▶ ' if is_self else '  '}{name}"
        attr = curses.color_pair(COLOR_TITLE) | curses.A_BOLD if is_self else curses.color_pair(COLOR_TITLE)
        self._safe_addstr(row, start_col, header, attr)
        score_text = f"Score: {score}"
        self._safe_addstr(row, start_col + len(header) + 2, score_text,
                          curses.color_pair(COLOR_SCORE))
        row += 1

        # Column numbers
        col_header = "    "
        for c in range(len(grid[0]) if grid else 0):
            col_header += f" C{c+1} "
        self._safe_addstr(row, start_col, col_header,
                          curses.color_pair(COLOR_DEFAULT) | curses.A_DIM)
        row += 1

        # Grid border top
        num_cols = len(grid[0]) if grid else 0
        border = "   ┌" + "────┬" * (num_cols - 1) + "────┐" if num_cols > 0 else ""
        self._safe_addstr(row, start_col, border, curses.color_pair(COLOR_DEFAULT) | curses.A_DIM)
        row += 1

        # Card rows
        for r_idx, card_row in enumerate(grid):
            row_label = f"R{r_idx+1} │"
            self._safe_addstr(row, start_col, row_label,
                              curses.color_pair(COLOR_DEFAULT) | curses.A_DIM)
            col = start_col + len(row_label)
            for c_idx, card in enumerate(card_row):
                card_str = format_card(card)
                color = get_card_color(card)
                self._safe_addstr(row, col, card_str, curses.color_pair(color) | curses.A_BOLD)
                if c_idx < len(card_row) - 1:
                    self._safe_addstr(row, col + len(card_str), "│",
                                      curses.color_pair(COLOR_DEFAULT) | curses.A_DIM)
                    col += len(card_str) + 1
                else:
                    self._safe_addstr(row, col + len(card_str), "│",
                                      curses.color_pair(COLOR_DEFAULT) | curses.A_DIM)
                    col += len(card_str) + 1
            row += 1

            # Row separator
            if r_idx < len(grid) - 1:
                sep = "   ├" + "────┼" * (num_cols - 1) + "────┤" if num_cols > 0 else ""
                self._safe_addstr(row, start_col, sep,
                                  curses.color_pair(COLOR_DEFAULT) | curses.A_DIM)
                row += 1

        # Grid border bottom
        border = "   └" + "────┴" * (num_cols - 1) + "────┘" if num_cols > 0 else ""
        self._safe_addstr(row, start_col, border, curses.color_pair(COLOR_DEFAULT) | curses.A_DIM)

    def _render_actions(self, row: int, legal_actions: List, selected_index: int):
        """Render the action selection menu."""
        self._safe_addstr(row, 2, "Available Actions:",
                          curses.color_pair(COLOR_TITLE) | curses.A_BOLD | curses.A_UNDERLINE)
        row += 1

        for i, action in enumerate(legal_actions):
            prefix = " ▶ " if i == selected_index else "   "
            action_text = f"{prefix}{i+1}. {action}"

            if i == selected_index:
                attr = curses.color_pair(COLOR_SELECTED) | curses.A_BOLD
            else:
                attr = curses.color_pair(COLOR_DEFAULT)

            self._safe_addstr(row + i, 2, action_text, attr)

    def render_round_summary(self, scores: List[int], player_names: List[str], round_num: int):
        """Render end-of-round summary."""
        self.stdscr.erase()
        max_y, max_x = self.stdscr.getmaxyx()

        row = max_y // 4
        title = f"══════  Round {round_num} Complete  ══════"
        self._safe_addstr(row, max(0, (max_x - len(title)) // 2), title,
                          curses.color_pair(COLOR_TITLE) | curses.A_BOLD)
        row += 3

        for i, (name, score) in enumerate(zip(player_names, scores)):
            text = f"  {name}: {score} points"
            self._safe_addstr(row + i, max(0, (max_x - len(text)) // 2), text,
                              curses.color_pair(COLOR_SCORE) | curses.A_BOLD)

        row += len(scores) + 3
        continue_text = "Press any key to continue..."
        self._safe_addstr(row, max(0, (max_x - len(continue_text)) // 2), continue_text,
                          curses.color_pair(COLOR_DEFAULT) | curses.A_DIM)

        self.stdscr.refresh()
        self.stdscr.getch()

    def render_game_over(self, scores: List[int], player_names: List[str]):
        """Render game over screen."""
        self.stdscr.erase()
        max_y, max_x = self.stdscr.getmaxyx()

        row = max_y // 4
        title = "╔══════════════════════════════╗"
        title2 = "║       G A M E   O V E R      ║"
        title3 = "╚══════════════════════════════╝"
        self._safe_addstr(row, max(0, (max_x - len(title)) // 2), title,
                          curses.color_pair(COLOR_TITLE) | curses.A_BOLD)
        self._safe_addstr(row + 1, max(0, (max_x - len(title2)) // 2), title2,
                          curses.color_pair(COLOR_TITLE) | curses.A_BOLD)
        self._safe_addstr(row + 2, max(0, (max_x - len(title3)) // 2), title3,
                          curses.color_pair(COLOR_TITLE) | curses.A_BOLD)
        row += 5

        # Determine winner (lowest score)
        winner_idx = scores.index(min(scores))

        for i, (name, score) in enumerate(zip(player_names, scores)):
            marker = " 🏆" if i == winner_idx else ""
            text = f"  {name}: {score} points{marker}"
            color = COLOR_NEGATIVE if i == winner_idx else COLOR_HIGH
            self._safe_addstr(row + i, max(0, (max_x - len(text)) // 2), text,
                              curses.color_pair(color) | curses.A_BOLD)

        row += len(scores) + 3
        winner_text = f"🎉 {player_names[winner_idx]} wins! 🎉"
        self._safe_addstr(row, max(0, (max_x - len(winner_text)) // 2), winner_text,
                          curses.color_pair(COLOR_NEGATIVE) | curses.A_BOLD)

        row += 3
        exit_text = "Press any key to exit..."
        self._safe_addstr(row, max(0, (max_x - len(exit_text)) // 2), exit_text,
                          curses.color_pair(COLOR_DEFAULT) | curses.A_DIM)

        self.stdscr.refresh()
        self.stdscr.getch()

    def _safe_addstr(self, row: int, col: int, text: str, attr: int = 0):
        """Safely add a string to the screen, handling boundary issues."""
        max_y, max_x = self.stdscr.getmaxyx()
        if row < 0 or row >= max_y or col >= max_x:
            return
        # Truncate text to fit
        available = max_x - col - 1
        if available <= 0:
            return
        text = text[:available]
        try:
            self.stdscr.addstr(row, col, text, attr)
        except curses.error:
            pass  # Ignore write errors at screen edge
