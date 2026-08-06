import os
import sys
import curses
import _curses
import logging

from Skyjo.src.ui.terminal_game_ui import TerminalGameUI
from Skyjo.src.players.so_ismcts_player import SOISMCTSPlayer
from Skyjo.src.players.terminal_player import TerminalPlayer
from Skyjo.src.skyjo_game import SkyjoGame

logger = logging.getLogger(__name__)

# Ensure the project root is on sys.path so 'Skyjo' package can be imported
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def get_model_path():
    return os.path.join(
        os.path.dirname(__file__),
        "src",
        "rl",
        "checkpoints",
        "skyjo_ppo_best.zip",
    )


def run_game(stdscr):
    """Main game loop running inside curses wrapper."""
    curses.curs_set(0)
    stdscr.keypad(True)

    analyze_mode = _is_analyze_mode()
    terminal_ui = TerminalGameUI(
        stdscr=stdscr,
        player_id=1,
        player_name="You",
        opponent_name="RL Player",
    )
    
    game = SkyjoGame()
    player1 = SOISMCTSPlayer(
        player_id=0,
        player_name="MCTS Player",
        num_iterations=1000,
        exploration=1.4,
        rollout_max_turns=100,
        pw_c=2.0,
        pw_alpha=0.4,
    )
    player2 = TerminalPlayer(
        player_id=1,
        player_name="You",
        ui=terminal_ui,
    )
    game.add_player(player1)
    game.add_player(player2)

    try:

        def on_round_end(g):
            scores = g.game_state.all_player_final_scores
            names = [p.player_name for p in g.players]
            round_num = g.game_state.round_number - 1
            player2.show_round_summary(scores, names, round_num)

        def on_game_over(g):
            final_scores = g.game_state.all_player_final_scores
            names = [p.player_name for p in g.players]
            player2.show_game_over(final_scores, names)

        game.play_game(on_round_end=on_round_end, on_game_over=on_game_over)

    except KeyboardInterrupt:
        pass


def _is_analyze_mode():
    return any(arg in sys.argv for arg in ("--analyze", "--analyse", "--analize"))


def main():

    logging.basicConfig(level=logging.CRITICAL)
    try:
        curses.wrapper(run_game)
    except _curses.error:
        # Fallback to legacy mode if no terminal is available (e.g. running from IDE)
        print("No terminal available for curses UI, falling back to legacy mode.")


if __name__ == "__main__":
    main()
