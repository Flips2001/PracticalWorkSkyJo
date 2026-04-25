But # CLAUDE.md

## Project Overview

Python 3.12 implementation of the Skyjo card game with a reinforcement learning agent trained via self-play using MaskablePPO. Two-player game on a 3×4 grid with cards valued -2 to 12. Game ends when any player reaches 100 points; lowest score wins.

## Commands

```bash
# Setup
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install sb3_contrib stable-baselines3 torch gymnasium greenlet tqdm  # RL deps

# Pre-commit hooks
pre-commit install
pre-commit run --all-files

# Tests (run from project root)
pytest

# Entry points
python Skyjo/main.py                      # Play a game
python Skyjo/competitions/competition.py   # Run 1000-game competition
python Skyjo/src/rl/train.py               # Train RL agent
```

## Project Structure

```
Skyjo/
  main.py                    # Game entry point
  src/
    card.py                  # Card dataclass (value, face_up)
    action.py                # Action dataclass (type, pos)
    action_type.py           # ActionType enum (DRAW_HIDDEN_CARD, DRAW_OPEN_CARD, FLIP_CARD, SWAP_CARD, DISCARD_CARD)
    turn_phase.py            # TurnPhase enum (STARTING_FLIPS, CHOOSE_DRAW, HAVE_DRAWN_HIDDEN, HAVE_DRAWN_OPEN, HAVE_TO_FLIP_AFTER_DISCARD, END_TURN)
    observation.py           # Observation dataclass (what a player can see)
    player_state.py          # PlayerState dataclass (grid, score)
    game_state.py            # GameState: deck, discard pile, round/game logic, column clearing
    skyjo_game.py            # SkyjoGame: orchestrates rounds, turns, legal actions, action execution
    players/
      player.py              # Abstract Player base class (ABC)
      human_player.py        # Interactive console player
      random_player.py       # Random action selection
      phillips_player.py     # Heuristic strategy (draw discard if ≤5, swap high cards, etc.)
      rl_player.py           # Trained MaskablePPO model player
    rl/
      encoding.py            # Observation → 80-dim float32 vector
      action_mapping.py      # Action ↔ int (27 discrete actions)
      pettingzoo_env.py      # PettingZoo AEC env using greenlet coroutines
      train.py               # Self-play training script (MaskablePPO, 8 SubprocVecEnv)
      checkpoints/           # Saved model weights (.zip)
  tests/                     # pytest tests (*_test.py naming)
  competitions/
    competition.py           # Competition runner
evaluate.py                  # Evaluation script
```

## Architecture

### Game Flow
`SkyjoGame` orchestrates the game loop: `play_game()` → `play_round()` → `start_round()` (2 initial flips per player) → `turn()` loop (CHOOSE_DRAW → DRAW → SWAP/DISCARD → optional FLIP → END_TURN). Columns with 3 matching revealed cards are auto-removed. When a player reveals all cards, others get one final turn. Round scores are doubled for the finisher if they don't have the lowest score.

### RL Pipeline
- **Environment**: `SkyjoEnv` (PettingZoo AEC) wraps `SkyjoGame` using `greenlet` coroutines — `ProxyPlayer.select_action()` switches to the env greenlet, which returns the action via `env.step()`
- **Observation**: 70-dim vector encoding own grid (24), opponent grid (24), discard/hand cards (4), turn phase one-hot (5), scores (2), draw pile size (1), column match counts (8), final turn flag (1), first finisher flag (1)
- **Action space**: 27 discrete actions — 0: draw hidden, 1: draw open, 2: discard, 3-14: flip at grid pos, 15-26: swap at grid pos
- **Training**: `MaskablePPO` (sb3_contrib) with action masking, self-play (90% model, 10% random), linear LR/clip schedule, 8 parallel envs, periodic evaluation vs PhillipsPlayer
- **Reward**: Terminal only — +1 win, -1 loss

## Code Conventions

- **Data objects**: `@dataclass` (`Card`, `Action`, `Observation`, `PlayerState`, `GameState`)
- **Player interface**: Abstract `Player(ABC)` with `select_action(observation, legal_actions) → Action`
- **Enums**: `ActionType`, `TurnPhase`
- **Type hints**: Used throughout, `List`, `Optional`, `Tuple` from `typing`
- **Imports**: Always use full path from package root (`from Skyjo.src.card import Card`)
- **Tests**: Named `<module>_test.py` in `Skyjo/tests/`, run with `pytest` from root
- **Logging**: `logging` module, `logger = logging.getLogger(__name__)`

