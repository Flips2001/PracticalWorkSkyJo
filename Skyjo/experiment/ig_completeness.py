"""Does integrated gradients satisfy completeness on the real model in real games?

Run this file directly (IDE run button). It plays full games of the agent
against a Phillips bot and explains every agent decision the way the UI does
(``explain_action``, blindfold baseline). Completeness says the attributions
must sum to log p(action | observation) - log p(action | baseline); the gap
left over is the Riemann-sum error of the path integral, reported per step
count.
"""

import logging
import random
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402

from Skyjo.scenarios.scenario import (  # noqa: E402
    DEFAULT_MODEL,
    DEVICE,
    _resolve_model_path,
)
from Skyjo.src.players.phillips_player import PhillipsPlayer  # noqa: E402
from Skyjo.src.players.rl_player import RLPlayer  # noqa: E402
from Skyjo.src.rl.integrated_gradients import explain_action  # noqa: E402
from Skyjo.src.skyjo_game import SkyjoGame  # noqa: E402

GAMES = 10
MODEL = DEFAULT_MODEL
STEPS = (32, 128, 512, 2048, 8192)
# A decision passes when its gap is within REL_TOL of the total influence, or
# within ABS_TOL when the total influence itself is close to zero.
REL_TOL = 0.05
ABS_TOL = 0.001
MIN_TOTAL_FOR_RELATIVE = 0.001


class _CompletenessRecorder:
    """Action hook that explains every agent decision before it is executed."""

    def __init__(self, model):
        self.model = model
        self.agent_id = 0
        self.totals = {steps: [] for steps in STEPS}
        self.gaps = {steps: [] for steps in STEPS}

    def before_action(self, game, player, action):
        if player.player_id != self.agent_id:
            return
        observation = game.get_observation(player)
        legal_actions = game.get_legal_actions(player)
        for steps in STEPS:
            explanation = explain_action(
                self.model, observation, action, legal_actions, steps=steps
            )
            attributed = sum(unit.attribution for unit in explanation.units)
            self.totals[steps].append(explanation.total_influence)
            self.gaps[steps].append(attributed - explanation.total_influence)

    def after_action(self, game, player, action):
        pass


def play_game(model, recorder: _CompletenessRecorder) -> None:
    game = SkyjoGame(action_hooks=recorder)
    seats = [None, None]
    seats[recorder.agent_id] = RLPlayer(recorder.agent_id, "RL", model=model)
    seats[1 - recorder.agent_id] = PhillipsPlayer(1 - recorder.agent_id, "Phillips")
    for player in seats:  # SkyjoGame indexes seats by player id
        game.add_player(player)
    game.play_game()


def report(recorder: _CompletenessRecorder) -> None:
    for steps in STEPS:
        totals = np.array(recorder.totals[steps])
        attributed = totals + np.array(recorder.gaps[steps])
        gaps = np.abs(attributed - totals)
        passed = np.isclose(attributed, totals, rtol=REL_TOL, atol=ABS_TOL)
        large = np.abs(totals) > MIN_TOTAL_FOR_RELATIVE
        relative = gaps[large] / np.abs(totals[large])
        print(f"\n{steps} steps, {len(gaps)} decisions:")
        print(f"  |total influence|  median {np.median(np.abs(totals)):.4f}")
        print(
            f"  |gap|              median {np.median(gaps):.4f}"
            f"  p95 {np.percentile(gaps, 95):.4f}  max {gaps.max():.4f}"
        )
        if relative.size:
            print(
                f"  |gap| / |total|    median {np.median(relative):.1%}"
                f"  p95 {np.percentile(relative, 95):.1%}"
                f"  ({(~large).sum()} decisions with |total| <= "
                f"{MIN_TOTAL_FOR_RELATIVE} left out)"
            )
        print(f"  within tolerance   {passed.mean():.1%}")


def main() -> None:
    logging.basicConfig(level=logging.CRITICAL)
    model = MaskablePPO.load(_resolve_model_path(MODEL), device=DEVICE)
    recorder = _CompletenessRecorder(model)
    for _ in tqdm(range(GAMES)):
        recorder.agent_id = random.randint(0, 1)
        play_game(model, recorder)
    print(f"{MODEL}, {GAMES} games vs Phillips")
    report(recorder)


if __name__ == "__main__":
    main()
