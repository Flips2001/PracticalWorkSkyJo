import argparse
from itertools import combinations

from tqdm import tqdm

from Skyjo.src.players.phillips_player import PhillipsPlayer
from Skyjo.src.skyjo_game import SkyjoGame

CUTOFFS = list(range(-2, 13))
NUM_GAMES = 1000


def play_pairing(cutoff_a, cutoff_b, num_games):
    """Play a series of games between two cutoff configs, alternating seats.

    Returns:
        Tuple ``(wins_a, wins_b, ties)``.
    """
    wins_a = wins_b = ties = 0
    for i in range(num_games):
        seat_a = i % 2
        seat_b = 1 - seat_a

        seats: list = [None, None]
        seats[seat_a] = PhillipsPlayer(seat_a, f"cutoff={cutoff_a}", cutoff=cutoff_a)
        seats[seat_b] = PhillipsPlayer(seat_b, f"cutoff={cutoff_b}", cutoff=cutoff_b)

        game = SkyjoGame()
        for player in seats:
            game.add_player(player)
        game.play_game()

        scores = game.game_state.all_player_final_scores
        score_a, score_b = scores[seat_a], scores[seat_b]
        if score_a < score_b:
            wins_a += 1
        elif score_b < score_a:
            wins_b += 1
        else:
            ties += 1
    return wins_a, wins_b, ties


def main():
    parser = argparse.ArgumentParser(
        description="Phillips-vs-Phillips cutoff tournament"
    )
    parser.add_argument("--games", type=int, default=NUM_GAMES)
    args = parser.parse_args()

    cutoffs = CUTOFFS
    # wins[a][b] = games cutoff a won against cutoff b
    wins = {a: {b: 0 for b in cutoffs} for a in cutoffs}
    total_wins = {a: 0 for a in cutoffs}
    total_games = {a: 0 for a in cutoffs}

    for a, b in tqdm(list(combinations(cutoffs, 2))):
        wins_a, wins_b, _ = play_pairing(a, b, args.games)
        wins[a][b] = wins_a
        wins[b][a] = wins_b
        total_wins[a] += wins_a
        total_wins[b] += wins_b
        total_games[a] += args.games
        total_games[b] += args.games

    print(f"\nWins per pairing ({args.games} games each, row vs column):")
    header = "cutoff |" + "".join(f"{b:>6}" for b in cutoffs)
    print(header)
    print("-" * len(header))
    for a in cutoffs:
        cells = "".join("     -" if a == b else f"{wins[a][b]:>6}" for b in cutoffs)
        print(f"{a:>6} |{cells}")

    print("\nOverall win rates:")
    for a in sorted(cutoffs, key=lambda c: total_wins[c], reverse=True):
        rate = total_wins[a] / total_games[a] * 100
        print(f"  cutoff={a}: {total_wins[a]}/{total_games[a]} wins ({rate:.1f}%)")

    best = max(cutoffs, key=lambda c: total_wins[c])
    print(f"\nBest config: cutoff={best}")


if __name__ == "__main__":
    main()
