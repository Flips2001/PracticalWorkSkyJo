"""How often does the agent take the open card in real games, by its value?

Run this file directly (IDE run button). It plays full games of the agent
against a Phillips bot, swapping seats every game, and tallies every draw
decision of both players by the value on top of the discard pile, overall and
split into each player's first draw of a round versus all later draws. Phillips
takes the open card exactly when it is at most its cutoff, so its rows show the
rule the agent is compared against.
"""

import logging
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402
from tqdm import tqdm  # noqa: E402

from Skyjo.competitions.phillips_cutoff_competition import (  # noqa: E402
    resolve_worker_count,
)
from Skyjo.scenarios.scenario import (  # noqa: E402
    DEFAULT_MODEL,
    DEVICE,
    _resolve_model_path,
)
from Skyjo.src.action_type import ActionType  # noqa: E402
from Skyjo.src.players.phillips_player import PhillipsPlayer  # noqa: E402
from Skyjo.src.players.rl_player import RLPlayer  # noqa: E402
from Skyjo.src.skyjo_game import SkyjoGame  # noqa: E402

GAMES = 1000
MODEL = DEFAULT_MODEL
CARD_VALUES = range(-2, 13)
DRAWS = (ActionType.DRAW_OPEN_CARD, ActionType.DRAW_HIDDEN_CARD)
WHEN = ("first move", "later")
WORKERS = None  # None: all CPUs but one, see resolve_worker_count

_worker_model = None


class _DrawRecorder:
    """Action hook: (player name, WHEN) -> discard top value -> [took the open card]."""

    def __init__(self):
        self.draws = defaultdict(lambda: defaultdict(list))
        self._has_drawn = set()  # (round number, player id)

    def before_action(self, game, player, action):
        if action.type in DRAWS:
            top = game.game_state.discard_pile[-1].get_value()
            took_open = action.type == ActionType.DRAW_OPEN_CARD
            turn = (game.game_state.round_number, player.player_id)
            when = WHEN[turn in self._has_drawn]
            self._has_drawn.add(turn)
            self.draws[player.player_name, when][top].append(took_open)

    def after_action(self, game, player, action):
        pass


def play_game(model, recorder: _DrawRecorder, rl_first: bool) -> None:
    game = SkyjoGame(action_hooks=recorder)
    # SkyjoGame indexes seats by player id, so ids follow the seat order.
    rl_id, phillips_id = (0, 1) if rl_first else (1, 0)
    seats = sorted(
        [RLPlayer(rl_id, "RL", model=model), PhillipsPlayer(phillips_id, "Phillips")],
        key=lambda player: player.player_id,
    )
    for player in seats:
        game.add_player(player)
    game.play_game()


def _init_worker() -> None:
    """Load the model once per worker process."""
    global _worker_model
    logging.basicConfig(level=logging.CRITICAL)
    torch.set_num_threads(1)  # parallelism comes from the processes
    _worker_model = MaskablePPO.load(_resolve_model_path(MODEL), device=DEVICE)


def _play_game_job(rl_first: bool) -> dict:
    """One game in a worker; plain dicts because the recorder does not pickle."""
    recorder = _DrawRecorder()
    play_game(_worker_model, recorder, rl_first)
    return {key: dict(by_value) for key, by_value in recorder.draws.items()}


def report(name: str, by_when) -> None:
    overall = {
        v: [took for when in WHEN for took in by_when[when].get(v, [])]
        for v in CARD_VALUES
    }
    decisions = [took for value in overall.values() for took in value]
    print(
        f"\n{name}: {len(decisions)} draw decisions, "
        f"open card {sum(decisions) / len(decisions):.1%}"
    )
    print("  discard top         " + "".join(f"{v:>5}" for v in CARD_VALUES))
    for label, by_value in (("all", overall), *((w, by_when[w]) for w in WHEN)):
        print(
            f"  took open {label:<10}"
            + "".join(
                (
                    f"{sum(by_value[v]) / len(by_value[v]):>5.0%}"
                    if by_value.get(v)
                    else "    -"
                )
                for v in CARD_VALUES
            )
        )
    for when in WHEN:
        counts = "".join(f"{len(by_when[when].get(v, [])):>5}" for v in CARD_VALUES)
        print(f"  n {when:<17}{counts}")


def main() -> None:
    workers = resolve_worker_count(WORKERS, GAMES)
    recorder = _DrawRecorder()
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn"), initializer=_init_worker
    ) as executor:
        futures = [
            executor.submit(_play_game_job, game % 2 == 0) for game in range(GAMES)
        ]
        for future in tqdm(
            as_completed(futures), total=GAMES, desc=f"{workers} workers"
        ):
            for key, by_value in future.result().items():
                for value, took in by_value.items():
                    recorder.draws[key][value].extend(took)
    print(f"{MODEL}, {GAMES} games vs Phillips")
    for name in ("RL", "Phillips"):
        report(name, {when: recorder.draws[name, when] for when in WHEN})


if __name__ == "__main__":
    main()
