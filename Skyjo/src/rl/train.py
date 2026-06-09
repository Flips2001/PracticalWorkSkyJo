"""Training script for Skyjo RL agent via self-play using MaskablePPO."""

import math
import os
from dataclasses import dataclass
from typing import Callable

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

from Skyjo.src.rl.action_mapping import NUM_ACTIONS
from Skyjo.src.rl.column_clear_drill_env import make_column_clear_drill_env
from Skyjo.src.rl.encoding import OBS_SIZE
from Skyjo.src.rl.pettingzoo_env import COLUMN_CLEAR_REWARD_DIVISOR
from Skyjo.src.rl.self_play_wrapper import make_env
from Skyjo.src.players.rl_player import RLPlayer
from Skyjo.src.players.phillips_player import PhillipsPlayer
from Skyjo.src.players.player import Player

DEVICE = "cpu"
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 40_000_000
NUM_CHECKPOINTS = 10
NUM_EVALS = 20
SAVE_EVERY = 3_000_000
EVAL_EVERY = 1_500_000
EVAL_GAMES = 100
NUM_PROCS = 8
COLUMN_CLEAR_DRILL_ENVS = 1

# Opponent model state (per subprocess)
_BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "skyjo_ppo_best")


def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def _mean_ci95(values) -> tuple[float, float]:
    """Return (mean, 95% confidence half-width) for a list of samples."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, 1.96 * math.sqrt(variance) / math.sqrt(n)


def _play_matchup_game(model, make_opponent, rl_seat: int, rl_deterministic=True):
    """Play one game with the RL model seated at `rl_seat`; return per-game stats.

    Players must be added in player_id order (0 then 1) because the game indexes
    `self.players` by player_id, so we place the RL/opponent into the right slot.
    """
    from Skyjo.src.skyjo_game import SkyjoGame

    game = SkyjoGame()
    seats: list = [None, None]
    seats[rl_seat] = RLPlayer(
        player_id=rl_seat,
        player_name="RL",
        model=model,
        deterministic=rl_deterministic,
    )
    seats[1 - rl_seat] = make_opponent(1 - rl_seat)
    for player in seats:
        game.add_player(player)

    game.play_game()
    scores = game.game_state.all_player_final_scores
    rl_score = scores[rl_seat]
    opp_score = scores[1 - rl_seat]
    clears = game.total_columns_cleared.get(rl_seat, 0)
    return rl_score, opp_score, clears


def evaluate_matchup(
    model,
    make_opponent,
    num_games=EVAL_GAMES,
    alternate_seats=True,
    rl_deterministic=True,
) -> dict:
    """Evaluate `model` vs an opponent factory.

    Reports win rate plus the dense, sensitive metric — mean score margin
    (opponent − RL, positive = RL ahead) with a 95% CI — so progress stays
    visible even when win rate saturates. Seats are alternated to remove
    first-player bias.
    """
    rl_scores, opp_scores, margins, clears = [], [], [], []
    wins = 0
    for i in range(num_games):
        rl_seat = (i % 2) if alternate_seats else 0
        rl_score, opp_score, clear = _play_matchup_game(
            model, make_opponent, rl_seat, rl_deterministic=rl_deterministic
        )
        rl_scores.append(rl_score)
        opp_scores.append(opp_score)
        margins.append(opp_score - rl_score)
        clears.append(clear)
        if rl_score < opp_score:
            wins += 1

    margin_mean, margin_ci = _mean_ci95(margins)
    return {
        "winrate": wins / num_games * 100,
        "rl_avg": sum(rl_scores) / num_games,
        "opp_avg": sum(opp_scores) / num_games,
        "margin": margin_mean,
        "margin_ci": margin_ci,
        "clears": sum(clears) / num_games,
    }


@dataclass(frozen=True)
class Opponent:
    """A named opponent to evaluate the model against during training.

    Plug new ones into the `opponents` list in `train()` and pick which one
    drives best-model selection via `primary_name`. Set `rl_deterministic=False`
    to make the RL side sample its actions — required for the frozen mirror,
    where two greedy equal policies can lock into non-terminating rounds.
    """

    name: str
    make_opponent: Callable[[int], Player]
    rl_deterministic: bool = True


def phillips_opponent() -> Opponent:
    """The hand-written baseline — the main, human-meaningful objective."""
    return Opponent(
        name="Phillips",
        make_opponent=lambda pid: PhillipsPlayer(player_id=pid, player_name="Phillips"),
    )


def frozen_mirror_opponent(reference_model) -> Opponent:
    """A frozen snapshot of the policy; its win rate tracks improvement vs baseline."""
    return Opponent(
        name="mirror",
        make_opponent=lambda pid: RLPlayer(
            player_id=pid,
            player_name="Reference",
            model=reference_model,
            deterministic=False,
        ),
        rl_deterministic=False,
    )


def evaluate_opponent(model, opponent: Opponent, num_games=EVAL_GAMES) -> dict:
    """Evaluate `model` against a single plugged-in `Opponent`."""
    return evaluate_matchup(
        model,
        opponent.make_opponent,
        num_games=num_games,
        rl_deterministic=opponent.rl_deterministic,
    )


def _selection_key(primary_result: dict) -> tuple[float, float]:
    """Best-model selection key: win rate first, score margin only as a tiebreak.

    Win rate is the objective; the margin tiebreak just keeps a saturated win
    rate from freezing best-model selection (ties are common near the ceiling).
    """
    return (primary_result["winrate"], primary_result["margin"])


class TqdmCallback(BaseCallback):
    """Callback with tqdm progress bar, periodic evaluation and checkpointing."""

    def __init__(
        self,
        total_timesteps,
        opponents,
        primary_name,
        eval_every=EVAL_EVERY,
        save_every=SAVE_EVERY,
    ):
        super().__init__(verbose=0)
        self.total_timesteps = total_timesteps
        self.opponents = opponents
        self.primary_name = primary_name
        self.eval_every = eval_every
        self.save_every = save_every
        self.pbar = None
        self._last_eval = 0
        self._last_save = 0
        self._best_key = (float("-inf"), float("-inf"))

    def _on_training_start(self):
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc="Training",
            unit="steps",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

    def _on_step(self) -> bool:
        self.pbar.update(self.num_timesteps - self.pbar.n)
        steps = self.num_timesteps

        # Save checkpoint
        if steps - self._last_save >= self.save_every:
            self._last_save = steps
            path = os.path.join(CHECKPOINT_DIR, f"skyjo_ppo_{steps}")
            self.model.save(path)
            self.pbar.write(f"  💾 Checkpoint saved: {steps:,} steps")

        # Evaluate vs every plugged-in opponent; select best by the primary one.
        if steps - self._last_eval >= self.eval_every:
            self._last_eval = steps
            self.pbar.set_description("Evaluating...")
            results = {
                opp.name: evaluate_opponent(self.model, opp) for opp in self.opponents
            }
            primary = results[self.primary_name]

            key = _selection_key(primary)
            marker = ""
            if key > self._best_key:
                self._best_key = key
                self.model.save(os.path.join(CHECKPOINT_DIR, "skyjo_ppo_best"))
                marker = " 🏆 NEW BEST"

            summary = " | ".join(
                f"{name}: Win={r['winrate']:.0f}% "
                f"margin={r['margin']:+.1f}±{r['margin_ci']:.1f}"
                for name, r in results.items()
            )
            self.pbar.write(
                f"  📊 {steps/1e6:.1f}M | {summary} | "
                f"clears={primary['clears']:.2f}{marker}"
            )
            self.pbar.set_description("Training")

        return True

    def _on_training_end(self):
        self.pbar.close()


def train():
    self_play_envs = NUM_PROCS - COLUMN_CLEAR_DRILL_ENVS
    if self_play_envs <= 0:
        raise ValueError("NUM_PROCS must be greater than COLUMN_CLEAR_DRILL_ENVS")

    env = SubprocVecEnv(
        [
            make_env(best_model_path=_BEST_MODEL_PATH, device=DEVICE)
            for _ in range(self_play_envs)
        ]
        + [make_column_clear_drill_env() for _ in range(COLUMN_CLEAR_DRILL_ENVS)]
    )

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
        activation_fn=torch.nn.ReLU,
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=linear_schedule(3e-4),
        n_steps=4096,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=linear_schedule(0.2),
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=DEVICE,
        policy_kwargs=policy_kwargs,
    )

    # Freeze a reference snapshot of the initial policy for the frozen-mirror
    # eval, so its win rate shows whether the live policy is improving.
    reference_path = os.path.join(CHECKPOINT_DIR, "skyjo_ppo_reference")
    model.save(reference_path)
    reference_model = MaskablePPO.load(reference_path, device=DEVICE)

    # Plug the evaluation opponents here. Training itself stays pure self-play;
    # these only measure progress. `primary_name` is the best-model objective.
    opponents = [
        phillips_opponent(),
        frozen_mirror_opponent(reference_model),
    ]
    primary_name = "Phillips"

    print("🎮 Skyjo RL Training")
    print(
        f"   Device: {DEVICE} | Envs: {NUM_PROCS} | Steps: {TOTAL_TIMESTEPS/1e6:.0f}M"
    )
    print(f"   OBS_SIZE={OBS_SIZE} | Actions={NUM_ACTIONS}")
    print(
        f"   Eval every {EVAL_EVERY/1e6:.1f}M steps ({EVAL_GAMES} games) vs "
        f"{', '.join(o.name for o in opponents)}; "
        f"best = highest {primary_name} win rate"
    )
    print(f"   Column clear reward divisor: {COLUMN_CLEAR_REWARD_DIVISOR:g}")
    print(
        f"   Envs: {self_play_envs} self-play | "
        f"{COLUMN_CLEAR_DRILL_ENVS} column-clear drill"
    )
    print("   Self-play opponent: best model, 10% random\n")

    callback = TqdmCallback(
        TOTAL_TIMESTEPS, opponents=opponents, primary_name=primary_name
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    final_path = os.path.join(CHECKPOINT_DIR, "skyjo_ppo_final")
    model.save(final_path)

    print(f"\n✅ Training complete. Final model: {final_path}")
    print(f"   Best {primary_name} win rate: {callback._best_key[0]:.0f}%")

    # Final evaluation vs every opponent.
    print("\n📊 Final Evaluation (200 games):")
    for opp in opponents:
        result = evaluate_opponent(model, opp, num_games=200)
        print(
            f"   vs {opp.name}: Win={result['winrate']:.0f}% | "
            f"margin={result['margin']:+.1f}±{result['margin_ci']:.1f} | "
            f"RL avg={result['rl_avg']:.1f} | opp avg={result['opp_avg']:.1f} | "
            f"clears={result['clears']:.2f}"
        )

    env.close()


if __name__ == "__main__":
    train()
