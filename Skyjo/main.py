from Skyjo.src.players.human_player import HumanPlayer
from Skyjo.src.players.random_player import RandomPlayer
from Skyjo.src.skyjo_game import SkyjoGame
import logging

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    game = SkyjoGame()
    player1 = RandomPlayer(player_id=0, player_name="Charlie")
    player2 = RandomPlayer(player_id=1, player_name="Phillip")
    player3 = HumanPlayer(player_id=2, player_name="John")
    game.add_player(player1)
    game.add_player(player2)
    game.add_player(player3)

    game.play_game()
