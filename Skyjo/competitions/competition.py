from Skyjo.src.players.phillips_player import PhillipsPlayer
from Skyjo.src.players.random_player import RandomPlayer
from Skyjo.src.skyjo_game import SkyjoGame
from tqdm import tqdm

NUM_GAMES = 1000

if __name__ == "__main__":
    player1 = RandomPlayer(player_id=0, player_name="Random")
    player2 = PhillipsPlayer(player_id=1, player_name="Phillips")
    players = [player1, player2]

    total_scores = [0, 0, 0]
    wins = [0, 0, 0]

    for i in tqdm(range(NUM_GAMES)):
        game = SkyjoGame()
        for player in players:
            game.add_player(player)

        game.play_game()

        scores = game.game_state.all_player_final_scores
        for j in range(len(players)):
            total_scores[j] += scores[j]

        # Winner is the one with the lowest score
        winner_id = scores.index(min(scores))
        wins[winner_id] += 1

    print(f"\nAfter {NUM_GAMES} games:")
    for j in range(len(players)):
        print(
            f"{players[j].player_name}: Total Score: {total_scores[j]}, Average: {total_scores[j] / NUM_GAMES:.2f}, Wins: {wins[j]}"
        )
