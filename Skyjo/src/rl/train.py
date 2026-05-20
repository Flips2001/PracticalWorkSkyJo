"""Training script for Skyjo RL agent via self-play using MaskablePPO."""

import os
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

from Skyjo.src.rl.action_mapping import NUM_ACTIONS
from Skyjo.src.rl.encoding import OBS_SIZE
from Skyjo.src.rl.self_play_wrapper import make_env
from Skyjo.src.players.rl_player import RLPlayer
from Skyjo.src.players.phillips_player import PhillipsPlayer

DEVICE = "cpu"
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 60_000_000
NUM_CHECKPOINTS = 10
NUM_EVALS = 20
SAVE_EVERY = TOTAL_TIMESTEPS // NUM_CHECKPOINTS
EVAL_EVERY = TOTAL_TIMESTEPS // NUM_EVALS
EVAL_GAMES = 100
NUM_PROCS = 8

# Opponent model state (per subprocess)
_BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "skyjo_ppo_best")


def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def evaluate_vs_phillips(model, num_games=EVAL_GAMES):
    """Run evaluation games: RL model vs PhillipsPlayer."""
    from Skyjo.src.skyjo_game import SkyjoGame

    wins = 0
    total_rl = 0
    total_phil = 0

    for _ in range(num_games):
        game = SkyjoGame()
        rl = RLPlayer(player_id=0, player_name="RL", model=model)
        phil = PhillipsPlayer(player_id=1, player_name="Phillips")
        game.add_player(rl)
        game.add_player(phil)
        game.play_game()
        scores = game.game_state.all_player_final_scores
        total_rl += scores[0]
        total_phil += scores[1]
        if scores[0] < scores[1]:
            wins += 1

    return wins / num_games * 100, total_rl / num_games, total_phil / num_games


class TqdmCallback(BaseCallback):
    """Callback with tqdm progress bar, periodic evaluation and checkpointing."""

    def __init__(self, total_timesteps, eval_every=EVAL_EVERY, save_every=SAVE_EVERY):
        super().__init__(verbose=0)
        self.total_timesteps = total_timesteps
        self.eval_every = eval_every
        self.save_every = save_every
        self.pbar = None
        self._last_eval = 0
        self._last_save = 0
        self._best_winrate = 0.0

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

        # Evaluate vs Phillips
        if steps - self._last_eval >= self.eval_every:
            self._last_eval = steps
            self.pbar.set_description("Evaluating vs Phillips...")
            winrate, avg_rl, avg_phil = evaluate_vs_phillips(self.model)
            marker = " 🏆 NEW BEST" if winrate > self._best_winrate else ""
            if winrate > self._best_winrate:
                self._best_winrate = winrate
                self.model.save(os.path.join(CHECKPOINT_DIR, "skyjo_ppo_best"))
            self.pbar.write(
                f"  📊 {steps/1e6:.1f}M steps | vs Phillips: "
                f"Win={winrate:.0f}% | RL avg={avg_rl:.1f} | Phil avg={avg_phil:.1f}{marker}"
            )
            self.pbar.set_description("Training")

        return True

    def _on_training_end(self):
        self.pbar.close()


def train():
    env = SubprocVecEnv(
        [
            make_env(best_model_path=_BEST_MODEL_PATH, device=DEVICE)
            for _ in range(NUM_PROCS)
        ]
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

    print("🎮 Skyjo RL Training")
    print(
        f"   Device: {DEVICE} | Envs: {NUM_PROCS} | Steps: {TOTAL_TIMESTEPS/1e6:.0f}M"
    )
    print(f"   OBS_SIZE={OBS_SIZE} | Actions={NUM_ACTIONS}")
    print(f"   Eval vs Phillips every {EVAL_EVERY/1e6:.1f}M steps ({EVAL_GAMES} games)")
    print("   Self-play opponent: best model, 10% random\n")

    callback = TqdmCallback(TOTAL_TIMESTEPS)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    final_path = os.path.join(CHECKPOINT_DIR, "skyjo_ppo_final")
    model.save(final_path)

    print(f"\n✅ Training complete. Final model: {final_path}")
    print(f"   Best winrate vs Phillips: {callback._best_winrate:.0f}%")

    # Final evaluation
    print("\n📊 Final Evaluation (200 games):")
    winrate, avg_rl, avg_phil = evaluate_vs_phillips(model, num_games=200)
    print(
        f"   vs Phillips: Win={winrate:.0f}% | RL avg={avg_rl:.1f} | Phil avg={avg_phil:.1f}"
    )

    env.close()


if __name__ == "__main__":
    train()
