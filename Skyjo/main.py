from Skyjo.src.players.human_player import HumanPlayer
from Skyjo.src.players.rl_player import RLPlayer
from Skyjo.src.skyjo_game import SkyjoGame
import logging

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    game = SkyjoGame()
    player1 = RLPlayer(
        player_id=0,
        player_name="RL Player",
        model_path="/Users/philliplechenauer/Developer/PracticalWorkSkyJo/Skyjo/src/rl/checkpoints/skyjo_ppo_best.zip",
    )
    player2 = HumanPlayer(player_id=1, player_name="Phillip")
    game.add_player(player1)
    game.add_player(player2)

    game.play_game()
