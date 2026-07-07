"""
Terminal UI renderer for Skyjo using curses.
Provides a full-screen, color-coded game display with in-place updates.

Attribution display: the live board is always rendered clean. The RL
opponent's last move is explained in a separate analysis block on the right —
a frozen copy of the decision-time state (grids, discard, hand, deck counts,
scores) with a heatmap overlay from green (little influence) to red (much
influence), normalized to the strongest unit of that move. Snapshot and
heatmap therefore always describe the same board, even while the live game
moves on.
"""

import curses
from typing import Any, List, Optional

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

# Heatmap pairs occupy HEAT_PAIR_BASE.. with black text on a green→red ramp.
HEAT_PAIR_BASE = 11
_HEAT_COLORS_256 = (40, 118, 226, 214, 202, 196)
_HEAT_COLORS_8 = (curses.COLOR_GREEN, curses.COLOR_YELLOW, curses.COLOR_RED)

# Units below this fraction of the move's strongest unit stay untinted.
_HEAT_NOISE_FLOOR = 0.05

# Card values in the order of Observation.draw_pile_value_counts.
_DECK_VALUES = tuple(range(-2, 13))

# Layout: opponent grid offset within a state block, analysis block column.
_OPPONENT_GRID_OFFSET = 38
_ANALYSIS_COL = 80


def _heat_colors():
    return _HEAT_COLORS_256 if curses.COLORS >= 256 else _HEAT_COLORS_8


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
    for i, color in enumerate(_heat_colors()):
        curses.init_pair(HEAT_PAIR_BASE + i, curses.COLOR_BLACK, color)


def heat_attr(strength: float) -> int:
    """Background tint for a normalized attribution strength in (0, 1]."""
    if strength <= 0:
        return 0
    levels = len(_heat_colors())
    level = min(levels - 1, int(strength * levels))
    return curses.color_pair(HEAT_PAIR_BASE + level)


class _Heat:
    """Attribution tints for one explanation, normalized to its strongest unit.

    The explanation always belongs to the RL opponent, so its "own" units map
    to the opponent's grid and its "opponent" units to the viewer's grid.

    Tints are suppressed entirely for low-influence moves so a negligible move
    never shows a saturated heatmap.
    """

    def __init__(self, explanation: Optional[Any]):
        if (
            explanation is None
            or getattr(explanation, "error", None)
            or getattr(explanation, "low_influence", False)
        ):
            explanation = None
        self._explanation = explanation
        self._max = explanation.max_abs_attribution if explanation else 0.0
        self._cells = {
            True: explanation.grid_map("opponent") if explanation else {},
            False: explanation.grid_map("own") if explanation else {},
        }
        self._deck = explanation.deck_map() if explanation else {}

    def _tint(self, unit) -> int:
        if unit is None or self._max <= 0:
            return 0
        strength = unit.abs_attribution / self._max
        return heat_attr(strength) if strength >= _HEAT_NOISE_FLOOR else 0

    def cell(self, viewer_grid: bool, pos) -> int:
        return self._tint(self._cells[viewer_grid].get(pos))

    def deck(self, card_value: int) -> int:
        return self._tint(self._deck.get(card_value))

    def unit(self, group: str, viewer: Optional[bool] = None) -> int:
        if self._explanation is None:
            return 0
        owner = None if viewer is None else ("opponent" if viewer else "own")
        return self._tint(self._explanation.unit_for(group, owner))


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
        opponent_last_action: str = "",
        opponent_explanation: Optional[Any] = None,
        opponent_snapshot: Optional[Observation] = None,
        show_actions: bool = True,
        help_text: str = " ↑↓ Navigate  │  Enter Select  │  q Quit ",
    ):
        """Render the live game state plus the RL move analysis block."""
        self.stdscr.erase()
        max_y, max_x = self.stdscr.getmaxyx()

        row = 0
        # Title bar
        title = "═══════════════════════  S K Y J O  ═══════════════════════"
        self._safe_addstr(
            row,
            max(0, (max_x - len(title)) // 2),
            title,
            curses.color_pair(COLOR_TITLE) | curses.A_BOLD,
        )
        row += 2

        # Phase info
        phase_text = PHASE_DESCRIPTIONS.get(
            observation.turn_phase, str(observation.turn_phase)
        )
        self._safe_addstr(
            row,
            2,
            f"Phase: {phase_text}",
            curses.color_pair(COLOR_PHASE) | curses.A_BOLD,
        )
        row += 1

        # Final turn warning
        if observation.final_turn_phase:
            self._safe_addstr(
                row,
                2,
                "⚠️  FINAL TURN! ⚠️",
                curses.color_pair(COLOR_HIGH) | curses.A_BOLD | curses.A_BLINK,
            )
            row += 1
        row += 1

        state_top = row
        row = self._render_state(
            state_top, 2, observation, player_name, opponent_name, heat=None
        )
        self._render_analysis(
            state_top,
            _ANALYSIS_COL,
            opponent_last_action,
            opponent_explanation,
            opponent_snapshot,
            player_name,
            opponent_name,
        )
        row += 1

        # Action selection area
        if show_actions:
            self._render_actions(row, legal_actions, selected_index)

        # Status message
        if message:
            msg_row = (
                max_y - 2
                if max_y - 2 > row + len(legal_actions) + 2
                else row + len(legal_actions) + 3
            )
            self._safe_addstr(msg_row, 2, message, curses.color_pair(COLOR_TITLE))

        # Help bar at bottom
        help_row = max_y - 1
        self._safe_addstr(
            help_row,
            max(0, (max_x - len(help_text)) // 2),
            help_text,
            curses.color_pair(COLOR_TITLE) | curses.A_DIM,
        )

        self.stdscr.refresh()

    def _render_state(
        self,
        row: int,
        col: int,
        observation: Observation,
        player_name: str,
        opponent_name: str,
        heat: Optional[_Heat],
    ) -> int:
        """Render one full game state block; returns the next free row."""
        heat = heat or _Heat(None)

        self._render_game_info(row, col, observation, heat)
        row += 2

        self._render_deck_panel(row, col, observation, heat)
        row += 3

        if observation.total_scores:
            total_text = "Total Points:  "
            self._safe_addstr(row, col, total_text, curses.color_pair(COLOR_TITLE))
            col_offset = col + len(total_text)
            self._safe_addstr(
                row,
                col_offset,
                f"You: {observation.total_scores[observation.player_id]}",
                curses.color_pair(COLOR_SCORE) | curses.A_BOLD,
            )
            col_offset += 12
            for i, score in enumerate(observation.total_scores):
                if i != observation.player_id:
                    self._safe_addstr(
                        row,
                        col_offset,
                        f"{opponent_name}: {score}",
                        curses.color_pair(COLOR_SCORE),
                    )
            row += 1
        row += 1

        self._render_player_grid(
            row,
            col,
            player_name,
            observation.card_grid,
            observation.scores[observation.player_id],
            is_self=True,
            heat=heat,
        )
        if observation.opponent_cards:
            for i, opp_grid in enumerate(observation.opponent_cards):
                if opp_grid is not None:
                    opp_score = (
                        observation.scores[i] if i < len(observation.scores) else 0
                    )
                    self._render_player_grid(
                        row,
                        col + _OPPONENT_GRID_OFFSET,
                        opponent_name,
                        opp_grid,
                        opp_score,
                        is_self=False,
                        heat=heat,
                    )
                    break

        return row + 9

    def _render_analysis(
        self,
        row: int,
        col: int,
        action_text: str,
        explanation: Optional[Any],
        snapshot: Optional[Observation],
        player_name: str,
        opponent_name: str,
    ):
        """Render the RL move analysis: decision-time snapshot with heatmap."""
        if not action_text and explanation is None:
            return

        self._safe_addstr(
            row,
            col,
            "Integrated Gradients",
            curses.color_pair(COLOR_TITLE) | curses.A_BOLD | curses.A_UNDERLINE,
        )
        row += 1

        if action_text:
            self._safe_addstr(
                row,
                col,
                action_text,
                curses.color_pair(COLOR_PHASE) | curses.A_BOLD,
            )
            row += 1

        if explanation is None or snapshot is None:
            return

        if explanation.error:
            self._safe_addstr(
                row,
                col,
                f"Attribution unavailable: {explanation.error}",
                curses.color_pair(COLOR_DEFAULT),
            )
            return

        phase_text = PHASE_DESCRIPTIONS.get(
            snapshot.turn_phase, str(snapshot.turn_phase)
        )
        self._safe_addstr(
            row,
            col,
            f"Phase: {phase_text}",
            curses.color_pair(COLOR_DEFAULT) | curses.A_DIM,
        )
        row += 2

        row = self._render_state(
            row, col, snapshot, player_name, opponent_name, _Heat(explanation)
        )
        row += 1

        # Heatmap legend
        dim = curses.color_pair(COLOR_DEFAULT) | curses.A_DIM
        self._safe_addstr(row, col, "influence:  low ", dim)
        x = col + 16
        for i in range(len(_heat_colors())):
            self._safe_addstr(row, x, "  ", curses.color_pair(HEAT_PAIR_BASE + i))
            x += 2
        self._safe_addstr(row, x + 1, "high", dim)

    def _render_game_info(
        self, row: int, col: int, observation: Observation, heat: _Heat
    ):
        """Render discard pile, hand card, and draw pile info."""
        # Draw pile
        self._safe_addstr(row, col, "Draw Pile: ", curses.color_pair(COLOR_TITLE))
        draw_text = f"[{observation.draw_pile_size} cards]"
        self._safe_addstr(row, col + 11, draw_text, curses.color_pair(COLOR_HIDDEN))

        # Discard pile
        discard_col = col + 28
        self._safe_addstr(row, discard_col, "Discard: ", curses.color_pair(COLOR_TITLE))
        if observation.discard_top:
            card_str = f"[{format_card(observation.discard_top).strip()}]"
            tint = heat.unit("discard")
            attr = tint or curses.color_pair(get_card_color(observation.discard_top))
            self._safe_addstr(row, discard_col + 9, card_str, attr | curses.A_BOLD)
        else:
            self._safe_addstr(
                row, discard_col + 9, "[empty]", curses.color_pair(COLOR_DEFAULT)
            )

        # Hand card
        hand_col = col + 50
        self._safe_addstr(row, hand_col, "Hand: ", curses.color_pair(COLOR_TITLE))
        if observation.hand_card:
            card_str = f"[{format_card(observation.hand_card).strip()}]"
            tint = heat.unit("hand")
            attr = tint or curses.color_pair(get_card_color(observation.hand_card))
            self._safe_addstr(row, hand_col + 6, card_str, attr | curses.A_BOLD)
        else:
            self._safe_addstr(
                row,
                hand_col + 6,
                "[-]",
                curses.color_pair(COLOR_DEFAULT) | curses.A_DIM,
            )

    def _render_deck_panel(
        self, row: int, col: int, observation: Observation, heat: _Heat
    ):
        """Render remaining draw-pile counts per card value, heat-tinted."""
        counts = observation.draw_pile_value_counts
        if not counts:
            return

        self._safe_addstr(row, col, "Deck:", curses.color_pair(COLOR_TITLE))
        self._safe_addstr(
            row + 1, col, "left:", curses.color_pair(COLOR_DEFAULT) | curses.A_DIM
        )
        x = col + 6
        for value, count in zip(_DECK_VALUES, counts):
            self._safe_addstr(
                row,
                x,
                f"{value:3d}",
                curses.color_pair(COLOR_DEFAULT) | curses.A_DIM,
            )
            tint = heat.deck(value)
            self._safe_addstr(
                row + 1, x, f"{count:3d}", tint or curses.color_pair(COLOR_DEFAULT)
            )
            x += 4

    def _render_player_grid(
        self,
        start_row: int,
        start_col: int,
        name: str,
        grid: List[List[Card]],
        score: int,
        is_self: bool,
        heat: _Heat,
    ):
        """Render a player's card grid."""
        row = start_row

        # Player name and score
        header = f"{'▶ ' if is_self else '  '}{name}"
        attr = (
            curses.color_pair(COLOR_TITLE) | curses.A_BOLD
            if is_self
            else curses.color_pair(COLOR_TITLE)
        )
        self._safe_addstr(row, start_col, header, attr)
        score_text = f"Score: {score}"
        score_attr = heat.unit("score", viewer=is_self) or curses.color_pair(
            COLOR_SCORE
        )
        self._safe_addstr(row, start_col + len(header) + 2, score_text, score_attr)
        row += 1

        # Column numbers (0-indexed, aligned over card cells)
        col_header = "    "
        for c in range(len(grid[0]) if grid else 0):
            col_header += f" C{c}  "
        self._safe_addstr(
            row, start_col, col_header, curses.color_pair(COLOR_DEFAULT) | curses.A_DIM
        )
        row += 1

        # Grid border top
        num_cols = len(grid[0]) if grid else 0
        border = "   ┌" + "────┬" * (num_cols - 1) + "────┐" if num_cols > 0 else ""
        self._safe_addstr(
            row, start_col, border, curses.color_pair(COLOR_DEFAULT) | curses.A_DIM
        )
        row += 1

        # Card rows
        for r_idx, card_row in enumerate(grid):
            row_label = f"R{r_idx} │"
            self._safe_addstr(
                row,
                start_col,
                row_label,
                curses.color_pair(COLOR_DEFAULT) | curses.A_DIM,
            )
            col = start_col + len(row_label)
            for c_idx, card in enumerate(card_row):
                card_str = format_card(card)
                tint = heat.cell(is_self, (r_idx, c_idx))
                attr = tint or curses.color_pair(get_card_color(card))
                self._safe_addstr(row, col, card_str, attr | curses.A_BOLD)
                self._safe_addstr(
                    row,
                    col + len(card_str),
                    "│",
                    curses.color_pair(COLOR_DEFAULT) | curses.A_DIM,
                )
                col += len(card_str) + 1
            row += 1

            # Row separator
            if r_idx < len(grid) - 1:
                sep = (
                    "   ├" + "────┼" * (num_cols - 1) + "────┤" if num_cols > 0 else ""
                )
                self._safe_addstr(
                    row, start_col, sep, curses.color_pair(COLOR_DEFAULT) | curses.A_DIM
                )
                row += 1

        # Grid border bottom
        border = "   └" + "────┴" * (num_cols - 1) + "────┘" if num_cols > 0 else ""
        self._safe_addstr(
            row, start_col, border, curses.color_pair(COLOR_DEFAULT) | curses.A_DIM
        )

    def _render_actions(self, row: int, legal_actions: List, selected_index: int):
        """Render the action selection menu."""
        self._safe_addstr(
            row,
            2,
            "Available Actions:",
            curses.color_pair(COLOR_TITLE) | curses.A_BOLD | curses.A_UNDERLINE,
        )
        row += 1

        for i, action in enumerate(legal_actions):
            prefix = " ▶ " if i == selected_index else "   "
            action_text = f"{prefix}{i+1}. {action}"

            if i == selected_index:
                attr = curses.color_pair(COLOR_SELECTED) | curses.A_BOLD
            else:
                attr = curses.color_pair(COLOR_DEFAULT)

            self._safe_addstr(row + i, 2, action_text, attr)

    def render_round_summary(
        self, scores: List[int], player_names: List[str], round_num: int
    ):
        """Render end-of-round summary."""
        self.stdscr.erase()
        max_y, max_x = self.stdscr.getmaxyx()

        row = max_y // 4
        title = f"══════  Round {round_num} Complete  ══════"
        self._safe_addstr(
            row,
            max(0, (max_x - len(title)) // 2),
            title,
            curses.color_pair(COLOR_TITLE) | curses.A_BOLD,
        )
        row += 3

        for i, (name, score) in enumerate(zip(player_names, scores)):
            text = f"  {name}: {score} points"
            self._safe_addstr(
                row + i,
                max(0, (max_x - len(text)) // 2),
                text,
                curses.color_pair(COLOR_SCORE) | curses.A_BOLD,
            )

        row += len(scores) + 3
        continue_text = "Press any key to continue..."
        self._safe_addstr(
            row,
            max(0, (max_x - len(continue_text)) // 2),
            continue_text,
            curses.color_pair(COLOR_DEFAULT) | curses.A_DIM,
        )

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
        self._safe_addstr(
            row,
            max(0, (max_x - len(title)) // 2),
            title,
            curses.color_pair(COLOR_TITLE) | curses.A_BOLD,
        )
        self._safe_addstr(
            row + 1,
            max(0, (max_x - len(title2)) // 2),
            title2,
            curses.color_pair(COLOR_TITLE) | curses.A_BOLD,
        )
        self._safe_addstr(
            row + 2,
            max(0, (max_x - len(title3)) // 2),
            title3,
            curses.color_pair(COLOR_TITLE) | curses.A_BOLD,
        )
        row += 5

        # Determine winner (lowest score)
        winner_idx = scores.index(min(scores))

        for i, (name, score) in enumerate(zip(player_names, scores)):
            marker = " 🏆" if i == winner_idx else ""
            text = f"  {name}: {score} points{marker}"
            color = COLOR_NEGATIVE if i == winner_idx else COLOR_HIGH
            self._safe_addstr(
                row + i,
                max(0, (max_x - len(text)) // 2),
                text,
                curses.color_pair(color) | curses.A_BOLD,
            )

        row += len(scores) + 3
        winner_text = f"🎉 {player_names[winner_idx]} wins! 🎉"
        self._safe_addstr(
            row,
            max(0, (max_x - len(winner_text)) // 2),
            winner_text,
            curses.color_pair(COLOR_NEGATIVE) | curses.A_BOLD,
        )

        row += 3
        exit_text = "Press any key to exit..."
        self._safe_addstr(
            row,
            max(0, (max_x - len(exit_text)) // 2),
            exit_text,
            curses.color_pair(COLOR_DEFAULT) | curses.A_DIM,
        )

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
