import os
import sys
import curses
import logging

from Skyjo.src.players.rl_player import RLPlayer
from Skyjo.src.players.terminal_player import TerminalPlayer
from Skyjo.src.skyjo_game import SkyjoGame

logger = logging.getLogger(__name__)


def get_model_path():
    return os.path.join(
        os.path.dirname(__file__),
        os.pardir,
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
        while not game.game_state.is_game_over:
            game.play_round()
            game.game_state.game_over()

            scores = game.game_state.all_player_final_scores
            names = [p.player_name for p in game.players]
            round_num = game.game_state.round_number - 1

            if not game.game_state.is_game_over:
                player2.show_round_summary(scores, names, round_num)

        final_scores = game.game_state.all_player_final_scores
        names = [p.player_name for p in game.players]
        player2.show_game_over(final_scores, names)

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


if __name__ == "__main__":
    if "--legacy" in sys.argv:
        run_legacy()
    else:
        logging.basicConfig(level=logging.CRITICAL)
        curses.wrapper(run_game)
