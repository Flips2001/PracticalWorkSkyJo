"""How often does the sampling agent clear the column in the column-clear scenario?"""

import logging
from collections import Counter
import random
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sb3_contrib import MaskablePPO  # noqa: E402

from Skyjo.scenarios.column_clear import SCENARIO  # noqa: E402
from Skyjo.scenarios.scenario import (  # noqa: E402
    AGENT_ID,
    DEFAULT_MODEL,
    DEVICE,
    GRID_COLS,
    GRID_ROWS,
    HUMAN_ID,
    _resolve_model_path,
    build_scenario_game,
)
from Skyjo.src.action_type import ActionType  # noqa: E402
from Skyjo.src.players.phillips_player import PhillipsPlayer  # noqa: E402
from Skyjo.src.players.rl_player import RLPlayer  # noqa: E402
from Skyjo.src.skyjo_game import SkyjoGame  # noqa: E402

RUNS = 1000
MODEL = DEFAULT_MODEL
PHILLIPS_CUTOFFS = [2]
# Opponent cards face up when the scenario starts: 2 is right after the opening
# flips, 8 is deep into the round.
OPPONENT_REVEALED = range(2, 9)


class _FirstAction:
    """Action hook that remembers the first action of the turn, the draw choice."""

    def __init__(self):
        self.action = None

    def before_action(self, game, player, action):
        if self.action is None:
            self.action = action

    def after_action(self, game, player, action):
        pass


def build_randomized_game(agent) -> SkyjoGame:
    """The column clear scenario with a random board and a round-sized discard pile."""
    opponent = PhillipsPlayer(
        HUMAN_ID, "Phillips", cutoff=random.choice(PHILLIPS_CUTOFFS)
    )
    scenario = replace(
        SCENARIO, seed=None, your_grid=[["?"] * GRID_COLS for _ in range(GRID_ROWS)]
    )
    game = build_scenario_game(scenario, agent, opponent)

    revealed = random.choice(OPPONENT_REVEALED)
    opponent_cards = [c for row in game.player_states[HUMAN_ID].grid for c in row]
    for card in random.sample(opponent_cards, revealed):
        card.reveal()

    # Each reveal past the opening flips is roughly one turn per player, and each
    # turn discards about one card. Junk comes off the bottom of the shuffled deck
    # and goes under the 9, so deck counts stay consistent.
    state = game.game_state
    junk = [state.draw_pile.pop(0) for _ in range(2 * (revealed - 2))]
    for card in junk:
        card.reveal()
    state.discard_pile = junk + state.discard_pile
    return game


def run_once(model: MaskablePPO) -> tuple[ActionType, bool]:
    agent = RLPlayer(AGENT_ID, "RL", model=model, deterministic=True)
    game = build_randomized_game(agent)
    first = _FirstAction()
    game.action_hooks = first
    game.turn(agent)
    return first.action.type, game.total_columns_cleared.get(AGENT_ID, 0) > 0


def main() -> None:
    logging.basicConfig(level=logging.CRITICAL)
    model = MaskablePPO.load(_resolve_model_path(MODEL), device=DEVICE)
    results = [run_once(model) for _ in range(RUNS)]
    draws = Counter(draw for draw, _ in results)
    clears = sum(cleared for _, cleared in results)

    print(f"{MODEL}, {RUNS} runs:")
    for label, count in (
        ("drew the open card", draws[ActionType.DRAW_OPEN_CARD]),
        ("drew the hidden card", draws[ActionType.DRAW_HIDDEN_CARD]),
        ("cleared the column", clears),
    ):
        print(f"  {label:<22}{count:>5}  ({count / RUNS:.1%})")


if __name__ == "__main__":
    main()
