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
    """Resolve a checkpoint reference"""
    path = name_or_path
    if not (os.path.isabs(path) or os.path.exists(path)):
        path = os.path.join(CHECKPOINT_DIR, path)
    if not path.endswith(".zip"):
        path += ".zip"
    return path


def get_model_path() -> str:
    """Return the resolved path to the default best checkpoint."""
    return resolve_checkpoint("skyjo_ppo_best")


def model_display_name(name_or_path: str) -> str:
    base = os.path.basename(name_or_path)
    return base[:-4] if base.endswith(".zip") else base


def build_opponent(
    player_id: int, opponent_model=None, opponent_name="Model"
) -> Player:
    """Build the opponent the RL model plays against.

    A model opponent samples its actions, since two greedy RL policies can
    otherwise stall a round into the turn cap.

    Args:
        player_id: Seat index for the opponent.
        opponent_model: A loaded RL model to play as the opponent; if ``None``,
            the hand-written Phillips baseline is used instead.
        opponent_name: Display name for a model opponent.

    Returns:
        The constructed ``Player`` (an ``RLPlayer`` or ``PhillipsPlayer``).
    """
    if opponent_model is None:
        return PhillipsPlayer(player_id=player_id, player_name="Phillips")
    return RLPlayer(
        player_id=player_id,
        player_name=opponent_name,
        model=opponent_model,
        deterministic=False,
    )


def play_competition(
    model,
    opponent_model=None,
    rl_name="RL",
    opponent_name=None,
    num_games=NUM_GAMES,
):
    """Play a series of games and tally per-side results.

    Seats alternate each game so neither side keeps the first-player advantage.
    The RL side samples its actions against a model opponent

    Args:
        model: The RL model under evaluation.
        opponent_model: Opponent RL model; if ``None``, plays the Phillips
            baseline.
        rl_name: Display name for the RL side.
        opponent_name: Display name for a model opponent; defaults to ``"Model"``.
        num_games: Number of games to play.

    Returns:
        A tuple ``(rl_result, opp_result, ties)`` where each result is a dict
        with ``name``, ``total`` (summed final score) and ``wins`` keys, and
        ``ties`` is the number of drawn games.
    """
    is_model_opponent = opponent_model is not None
    rl_deterministic = not is_model_opponent
    opp_name = (opponent_name or "Model") if is_model_opponent else "Phillips"

    rl_total = opp_total = 0
    rl_wins = opp_wins = ties = 0

    for i in tqdm(range(num_games)):
        rl_seat = i % 2
        opp_seat = 1 - rl_seat

        seats: list = [None, None]
        seats[rl_seat] = RLPlayer(
            player_id=rl_seat,
            player_name=rl_name,
            model=model,
            deterministic=rl_deterministic,
        )
        seats[opp_seat] = build_opponent(opp_seat, opponent_model, opp_name)

        game = SkyjoGame()
        for player in seats:
            game.add_player(player)
        game.play_game()

        scores = game.game_state.all_player_final_scores
        rl_score, opp_score = scores[rl_seat], scores[opp_seat]
        rl_total += rl_score
        opp_total += opp_score
        # Lower total score wins in Skyjo; equal totals are a draw.
        if rl_score < opp_score:
            rl_wins += 1
        elif opp_score < rl_score:
            opp_wins += 1
        else:
            ties += 1

    return (
        {"name": rl_name, "total": rl_total, "wins": rl_wins},
        {"name": opp_name, "total": opp_total, "wins": opp_wins},
        ties,
    )


def main():
    """Parse CLI arguments, run the competition, and print the summary."""
    parser = argparse.ArgumentParser(description="Skyjo RL competition")
    parser.add_argument(
        "--model",
        default="skyjo_ppo_best",
        help="Checkpoint name or path for the RL model.",
    )
    parser.add_argument(
        "--opponent-model",
        default=None,
        help="Checkpoint name or path for the opponent model; "
        "omit to play against the Phillips baseline.",
    )
    parser.add_argument("--games", type=int, default=NUM_GAMES)
    args = parser.parse_args()

    model = MaskablePPO.load(resolve_checkpoint(args.model), device=DEVICE)
    opponent_model = None
    opponent_name = None
    if args.opponent_model is not None:
        opponent_model = MaskablePPO.load(
            resolve_checkpoint(args.opponent_model), device=DEVICE
        )
        opponent_name = model_display_name(args.opponent_model)

    rl_result, opp_result, ties = play_competition(
        model,
        opponent_model=opponent_model,
        rl_name=model_display_name(args.model),
        opponent_name=opponent_name,
        num_games=args.games,
    )

    print(
        f"\nAfter {args.games} games "
        f"({rl_result['name']} vs {opp_result['name']}, alternating seats):"
    )
    for side in (rl_result, opp_result):
        print(
            f"  {side['name']}: avg score {side['total'] / args.games:.2f}, "
            f"wins {side['wins']} ({side['wins'] / args.games * 100:.1f}%)"
        )
    print(f"  ties: {ties}")


if __name__ == "__main__":
    main()
