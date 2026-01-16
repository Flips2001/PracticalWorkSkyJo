from Skyjo.src.players.phillips_player import PhillipsPlayer
from Skyjo.src.players.random_player import RandomPlayer
from Skyjo.src.skyjo_game import SkyjoGame
from tqdm import tqdm

NUM_ROUNDS = 1000

BASELINE = RandomPlayer(player_id=0, player_name="Baseline")
PLAYER = PhillipsPlayer(player_id=1, player_name="Phillips Player")

if __name__ == "__main__":
    baseline = 0
    challenger = 0

    for i in tqdm(range(NUM_ROUNDS)):
        game = SkyjoGame(debug=True)
        game.add_player(BASELINE)
        game.add_player(PLAYER)

        game.play_game()

        scores = game.game_state.all_player_final_scores
        baseline += scores[0]
        challenger += scores[1]

    print(
        f"After {NUM_ROUNDS} games, {BASELINE} total score: {baseline}, {PLAYER} total score: {challenger}"
    )
    print(
        f"Average scores - {BASELINE}: {baseline / NUM_ROUNDS}, {PLAYER}: {challenger / NUM_ROUNDS}"
    )
