import argparse
import os

from sb3_contrib import MaskablePPO
from tqdm import tqdm

from Skyjo.src.players.phillips_player import PhillipsPlayer
from Skyjo.src.players.player import Player
from Skyjo.src.players.rl_player import RLPlayer
from Skyjo.src.skyjo_game import SkyjoGame

DEVICE = "cpu"
NUM_GAMES = 1000
CHECKPOINT_DIR = os.path.join(
    os.path.dirname(__file__), os.pardir, "src", "rl", "checkpoints"
)


def resolve_checkpoint(name_or_path: str) -> str:
    """Resolve a checkpoint name (e.g. 'skyjo_ppo_best') or path to a .zip file."""
    path = name_or_path
    if not (os.path.isabs(path) or os.path.exists(path)):
        path = os.path.join(CHECKPOINT_DIR, path)
    if not path.endswith(".zip"):
        path += ".zip"
    return path


def get_model_path() -> str:
    return resolve_checkpoint("skyjo_ppo_best")


def build_opponent(kind: str, player_id: int, mirror_model=None) -> Player:
    """Build the opponent the RL model competes against.

    'phillips' is the hand-written baseline. 'mirror' is a frozen RL snapshot
    (skyjo_ppo_reference, or any checkpoint passed via --mirror-model); it samples
    its actions, since two greedy RL policies can otherwise stall a round into the
    turn cap.
    """
    if kind == "phillips":
        return PhillipsPlayer(player_id=player_id, player_name="Phillips")
    if kind == "mirror":
        return RLPlayer(
            player_id=player_id,
            player_name="Mirror",
            model=mirror_model,
            deterministic=False,
        )
    raise ValueError(f"Unknown opponent: {kind!r}")


def play_competition(
    model,
    opponent_kind,
    mirror_model=None,
    num_games=NUM_GAMES,
    alternate_seats=True,
):
    """Play `model` against the chosen opponent and tally per-contestant results.

    Seats are alternated so neither side keeps the first-player advantage. Against
    the mirror both RL sides sample (matches the training-time mirror eval).
    """
    rl_deterministic = opponent_kind != "mirror"
    rl_name = "RL"
    opp_name = "Phillips" if opponent_kind == "phillips" else "Mirror"

    totals = {rl_name: 0, opp_name: 0}
    wins = {rl_name: 0, opp_name: 0}
    ties = 0

    for i in tqdm(range(num_games)):
        rl_seat = (i % 2) if alternate_seats else 0
        opp_seat = 1 - rl_seat

        seats: list = [None, None]
        seats[rl_seat] = RLPlayer(
            player_id=rl_seat,
            player_name=rl_name,
            model=model,
            deterministic=rl_deterministic,
        )
        seats[opp_seat] = build_opponent(opponent_kind, opp_seat, mirror_model)

        game = SkyjoGame()
        for player in seats:
            game.add_player(player)
        game.play_game()

        scores = game.game_state.all_player_final_scores
        rl_score, opp_score = scores[rl_seat], scores[opp_seat]
        totals[rl_name] += rl_score
        totals[opp_name] += opp_score
        if rl_score < opp_score:
            wins[rl_name] += 1
        elif opp_score < rl_score:
            wins[opp_name] += 1
        else:
            ties += 1

    return totals, wins, ties, (rl_name, opp_name)


def main():
    parser = argparse.ArgumentParser(description="Skyjo RL competition")
    parser.add_argument(
        "--opponent",
        choices=["phillips", "mirror"],
        default="phillips",
        help="Who the RL model plays against (default: phillips).",
    )
    parser.add_argument(
        "--model",
        default="skyjo_ppo_best",
        help="Checkpoint name or path for the RL model.",
    )
    parser.add_argument(
        "--mirror-model",
        default="skyjo_ppo_reference",
        help="Checkpoint name or path for the frozen mirror opponent.",
    )
    parser.add_argument("--games", type=int, default=NUM_GAMES)
    parser.add_argument(
        "--alternate-seats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Swap who goes first each game to remove first-player bias.",
    )
    args = parser.parse_args()

    model = MaskablePPO.load(resolve_checkpoint(args.model), device=DEVICE)
    mirror_model = None
    if args.opponent == "mirror":
        mirror_model = MaskablePPO.load(
            resolve_checkpoint(args.mirror_model), device=DEVICE
        )

    totals, wins, ties, (rl_name, opp_name) = play_competition(
        model,
        args.opponent,
        mirror_model=mirror_model,
        num_games=args.games,
        alternate_seats=args.alternate_seats,
    )

    seat_note = "alternating seats" if args.alternate_seats else "fixed seats"
    print(f"\nAfter {args.games} games (RL vs {opp_name}, {seat_note}):")
    for name in (rl_name, opp_name):
        print(
            f"  {name}: avg score {totals[name] / args.games:.2f}, "
            f"wins {wins[name]} ({wins[name] / args.games * 100:.1f}%)"
        )
    print(f"  ties: {ties}")


if __name__ == "__main__":
    main()
