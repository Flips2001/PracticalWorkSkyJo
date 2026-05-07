import os
import sys
import curses
import _curses
import logging

from Skyjo.src.players.rl_player import RLPlayer
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

    model_path = get_model_path()

    game = SkyjoGame()
    player1 = RLPlayer(
        player_id=0,
        player_name="RL Player",
        model_path=model_path,
    )
    player2 = TerminalPlayer(
        player_id=1,
        player_name="You",
        stdscr=stdscr,
        opponent_name="RL Player",
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


def run_legacy():
    """Run the game with the legacy text-based interface."""
    from Skyjo.src.players.human_player import HumanPlayer

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model_path = get_model_path()

    game = SkyjoGame()
    player1 = RLPlayer(
        player_id=0,
        player_name="RL Player",
        model_path=model_path,
    )
    player2 = HumanPlayer(player_id=1, player_name="Phillip")
    game.add_player(player1)
    game.add_player(player2)
    game.play_game()


def main():
    if "--legacy" in sys.argv:
        run_legacy()
    else:
        logging.basicConfig(level=logging.CRITICAL)
        try:
            curses.wrapper(run_game)
        except _curses.error:
            # Fallback to legacy mode if no terminal is available (e.g. running from IDE)
            print("No terminal available for curses UI, falling back to legacy mode.")
            run_legacy()


if __name__ == "__main__":
    main()
