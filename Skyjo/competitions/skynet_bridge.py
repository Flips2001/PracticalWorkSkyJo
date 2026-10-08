"""Shared plumbing for playing against SkyNet (Haile 2026, arXiv:2603.27751).

SkyNet is a belief-aware MuZero; its repo also ships the six rule-based
(heuristic) players it was trained and evaluated against. `download` fetches the
pinned repo (and, for SkyNet itself, its checkpoint) into the git-ignored
`.skynet/` folder on first use.

Games run in SkyNet's own decision env, so SkyNet sees exactly its training-time
observations and is searched as in the paper's eval (greedy 200-simulation
MCTS, no root noise); the heuristic players are greedy, as in its released run.
Our players are bridged in through `to_observation` / `legal_actions` /
`to_decision_actions`, so SkyNet's rule variant applies: at most one column
clear per player per round, cleared cards leave the game, every round starts
with the highest revealed sum, and the round ender is doubled whenever someone
scores strictly lower. SkyNet's side keeps the env's random setup flips; ours
choose theirs (see `choose_setup_flips`).

Each seed is played twice with seats swapped, so both sides get the same deals,
which cancels most card luck.
"""

import hashlib
import io
import math
import os
import sys
import tempfile
import urllib.request
import warnings
import zipfile
from collections import Counter
from dataclasses import replace
from multiprocessing import get_context

from tqdm import tqdm

from Skyjo.competitions.competition import model_display_name, resolve_checkpoint
from Skyjo.competitions.phillips_cutoff_competition import resolve_worker_count
from Skyjo.src.action import Action
from Skyjo.src.action_type import ActionType
from Skyjo.src.observation import Observation, ObservedCard
from Skyjo.src.players.phillips_player import PhillipsPlayer
from Skyjo.src.players.player import Player
from Skyjo.src.players.rl_player import RLPlayer
from Skyjo.src.turn_phase import TurnPhase

MODEL = "skyjo_ppo_final"
GAMES = 1000  # per matchup
SIMS = 200

SKYNET_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, ".skynet")
)
CHECKPOINT = os.path.join(
    SKYNET_DIR, "runs", "muzero_belief", "checkpoints", "checkpoint.pt"
)
_COMMIT = "e62a825d5fc1cc3e1e2d67e402254aa698eefcea"
_CODE_URL = f"https://codeload.github.com/DevArtech/skynet/zip/{_COMMIT}"
# The checkpoint lives in Git LFS, so the code zip only holds a pointer to it.
_CHECKPOINT_URL = (
    f"https://media.githubusercontent.com/media/DevArtech/skynet/{_COMMIT}"
    "/runs/muzero_belief/checkpoints/checkpoint.pt"
)
_CHECKPOINT_SHA256 = "0148a226890617ad573f280a87283fe86484ef40e0e28709e164fe075791bcdb"
_CHECKPOINT_SIZE = 215_657_589

ROWS, COLS = 3, 4
# SkyNet's own eval cap; a capped game is still scored as it stands.
MAX_DECISIONS = 2000
_CARD_VALUES = range(-2, 13)

# skyjo_decision_env.DecisionPhase / DecisionAction, mirrored so the adapter
# imports without the SkyNet checkout.
CHOOSE_SOURCE, KEEP_OR_DISCARD, CHOOSE_POSITION = 1, 2, 3
CHOOSE_DECK, CHOOSE_DISCARD, KEEP_DRAWN, DISCARD_DRAWN, POS_BASE = 0, 1, 2, 3, 4

# Per-worker models, loaded by `_init_worker`.
_ppo = None
_skynet = None
_mcts_config = None


def download(checkpoint: bool = False) -> None:
    """Fetch SkyNet's code, and with `checkpoint` its 216 MB model, if missing."""
    if not os.path.isdir(SKYNET_DIR):
        print(f"Downloading SkyNet code to {SKYNET_DIR} ...")
        with urllib.request.urlopen(_CODE_URL) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        # Extract beside the target and rename, so an interrupted run leaves no
        # half-filled folder behind.
        with tempfile.TemporaryDirectory(dir=os.path.dirname(SKYNET_DIR)) as tmp:
            archive.extractall(tmp)
            (top,) = os.listdir(tmp)
            os.rename(os.path.join(tmp, top), SKYNET_DIR)
    if checkpoint and os.path.getsize(CHECKPOINT) != _CHECKPOINT_SIZE:
        print("Downloading SkyNet checkpoint (216 MB) ...")
        partial = CHECKPOINT + ".partial"
        urllib.request.urlretrieve(_CHECKPOINT_URL, partial)
        with open(partial, "rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != _CHECKPOINT_SHA256:
                raise RuntimeError(f"SkyNet checkpoint checksum mismatch: {partial}")
        os.replace(partial, CHECKPOINT)
    sys.path.insert(0, SKYNET_DIR)


def _live_cols(board) -> list[int]:
    """SkyNet keeps cleared slots in place; our engine drops cleared columns."""
    return [c for c in range(COLS) if not board.removed[c]]


def _grid(board) -> tuple[tuple[ObservedCard, ...], ...]:
    cols = _live_cols(board)
    return tuple(
        tuple(
            ObservedCard(
                board.cards[r * COLS + c] if board.visible[r * COLS + c] else None,
                bool(board.visible[r * COLS + c]),
            )
            for c in cols
        )
        for r in range(ROWS)
    )


def _turn_phase(env) -> TurnPhase:
    phase = int(env.decision_phase)
    if phase == CHOOSE_SOURCE:
        return TurnPhase.CHOOSE_DRAW
    if phase == KEEP_OR_DISCARD:
        return TurnPhase.HAVE_DRAWN_HIDDEN
    if phase == CHOOSE_POSITION and env.pending.source == "DISCARD":
        return TurnPhase.HAVE_DRAWN_OPEN
    if phase == CHOOSE_POSITION and env.pending.keep_drawn is False:
        return TurnPhase.HAVE_TO_FLIP_AFTER_DISCARD
    # Keeping a deck card is sent together with its position, so our players
    # never act from "kept, position pending".
    raise ValueError(f"No engine phase for decision phase {phase}")


def _visible_sums(grids) -> tuple[int, ...]:
    return tuple(
        sum(card.value for row in grid for card in row if card.face_up)
        for grid in grids
    )


def to_observation(env, player_id: int) -> Observation:
    base = env.base
    phase = _turn_phase(env)
    grids = [_grid(board) for board in base.boards]
    holding = phase in (TurnPhase.HAVE_DRAWN_HIDDEN, TurnPhase.HAVE_DRAWN_OPEN)
    discard_counts = Counter(base.discard_pile)
    return Observation(
        player_id=player_id,
        card_grid=grids[player_id],
        hand_card=ObservedCard(env.pending.drawn_value, True) if holding else None,
        opponent_cards=tuple(
            None if pid == player_id else grid for pid, grid in enumerate(grids)
        ),
        scores=_visible_sums(grids),
        discard_top=(
            ObservedCard(base.discard_pile[-1], True) if base.discard_pile else None
        ),
        draw_pile_size=len(base.deck),
        turn_phase=phase,
        discard_pile_value_counts=tuple(discard_counts[v] for v in _CARD_VALUES),
        total_scores=tuple(base.scores),
        final_turn_phase=base.round_ender is not None,
        first_finisher_id=base.round_ender,
    )


def legal_actions(env) -> list[Action]:
    """Our engine's legal actions for the player to move, in grid coordinates."""
    board = env.base.boards[env.current_player]
    cols = _live_cols(board)
    positions = [(r, i) for r in range(ROWS) for i in range(len(cols))]
    hidden = [(r, i) for r, i in positions if not board.visible[r * COLS + cols[i]]]
    swaps = [Action(ActionType.SWAP_CARD, pos=pos) for pos in positions]
    match _turn_phase(env):
        case TurnPhase.CHOOSE_DRAW:
            draws = [Action(ActionType.DRAW_HIDDEN_CARD)]
            if env.base.discard_pile:
                draws.append(Action(ActionType.DRAW_OPEN_CARD))
            return draws
        case TurnPhase.HAVE_DRAWN_HIDDEN:
            return swaps + ([Action(ActionType.DISCARD_CARD)] if hidden else [])
        case TurnPhase.HAVE_DRAWN_OPEN:
            return swaps
        case TurnPhase.HAVE_TO_FLIP_AFTER_DISCARD:
            return [Action(ActionType.FLIP_CARD, pos=pos) for pos in hidden]


def to_decision_actions(env, action: Action) -> list[int]:
    match action.type:
        case ActionType.DRAW_HIDDEN_CARD:
            return [CHOOSE_DECK]
        case ActionType.DRAW_OPEN_CARD:
            return [CHOOSE_DISCARD]
        case ActionType.DISCARD_CARD:
            return [DISCARD_DRAWN]
    r, i = action.pos
    slot = r * COLS + _live_cols(env.base.boards[env.current_player])[i]
    if int(env.decision_phase) == KEEP_OR_DISCARD:
        return [KEEP_DRAWN, POS_BASE + slot]
    return [POS_BASE + slot]


def choose_setup_flips(env, players) -> None:
    """Replace the env's random setup flips with our players' own choices.

    SkyNet and its heuristic players trained on random setup flips, so they
    keep them (`players` holds `None` for them); our agent trained choosing its
    own, and random ones cost it ~18 points of win rate against Phillips. As in our
    engine, seats flip in order (a later seat sees earlier seats' flips) with no
    discard dealt yet. The env's public setup records are rewritten to match.
    """
    base = env.base
    unflipped = tuple(
        tuple(ObservedCard(None, False) for _ in range(COLS)) for _ in range(ROWS)
    )
    for pid, player in enumerate(players):
        if player is None:
            continue
        board = base.boards[pid]
        first = base.round_history_start_index + 2 * pid
        records = (first, first + 1)
        for k in records:
            board.visible[base.public_history[k].target_pos] = False
        for k in records:
            grids = [
                _grid(b) if q <= pid else unflipped for q, b in enumerate(base.boards)
            ]
            observation = Observation(
                player_id=pid,
                card_grid=grids[pid],
                hand_card=None,
                opponent_cards=tuple(
                    None if q == pid else g for q, g in enumerate(grids)
                ),
                scores=_visible_sums(grids),
                discard_top=None,
                draw_pile_size=len(base.deck) + len(base.discard_pile),
                turn_phase=TurnPhase.STARTING_FLIPS,
                total_scores=tuple(base.scores),
            )
            hidden = [
                Action(ActionType.FLIP_CARD, pos=(r, c))
                for r in range(ROWS)
                for c in range(COLS)
                if not board.visible[r * COLS + c]
            ]
            r, c = player.select_action(observation, hidden).pos
            slot = r * COLS + c
            board.visible[slot] = True
            base.public_history[k] = replace(
                base.public_history[k], target_pos=slot, a_value=board.cards[slot]
            )
    base.current_player = base._starting_player_from_revealed_sums()


def _init_worker(model_path: str, with_skynet: bool) -> None:
    global _ppo, _skynet, _mcts_config
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    sys.path.insert(0, SKYNET_DIR)
    _ppo = MaskablePPO.load(model_path, device="cpu")
    if not with_skynet:
        return
    # Harmless PyTorch notice from SkyNet's encoder, otherwise printed per worker.
    warnings.filterwarnings("ignore", message="enable_nested_tensor")
    from belief_muzero_model import (
        BeliefAwareMuZeroNet,
        build_default_belief_muzero_config,
    )
    from muzero_mcts import MCTSConfig

    state_dict = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)[
        "model_state_dict"
    ]
    action_space = state_dict["prediction.policy_head.net.6.weight"].shape[0]
    _skynet = BeliefAwareMuZeroNet(build_default_belief_muzero_config(action_space))
    _skynet.load_state_dict(state_dict)
    _skynet.eval()
    _mcts_config = MCTSConfig(
        num_simulations=SIMS,
        temperature=1e-8,
        add_exploration_noise=False,
        root_exploration_fraction=0.0,
    )


def worker_pool(with_skynet: bool = False):
    """Process pool with our model, and SkyNet's if asked, loaded per worker."""
    return get_context("spawn").Pool(
        resolve_worker_count(None, GAMES),
        initializer=_init_worker,
        initargs=(resolve_checkpoint(MODEL), with_skynet),
    )


def _make_player(kind: str, player_id: int, seed: int):
    """Our `Player`s go through the adapter; SkyNet and its heuristic players are
    callables acting on the env directly, as in SkyNet's own eval."""
    if kind == "rl":
        return RLPlayer(player_id=player_id, player_name="RL", model=_ppo)
    if kind == "phillips":
        return PhillipsPlayer(player_id=player_id, player_name="Phillips")
    if kind == "skynet":
        from belief_muzero_mcts import run_belief_mcts

        return lambda env: run_belief_mcts(
            _skynet,
            env.observe(player_id),
            env.legal_actions(),
            ego_player_id=player_id,
            config=_mcts_config,
        ).action
    from heuristic_bots import make_heuristic_bot

    # Greedy (epsilon 0), as in SkyNet's released training config.
    bot = make_heuristic_bot(kind, seed=seed, epsilon=0.0)
    return lambda env: bot.select_action(env.observe(player_id), env.legal_actions())


def play_game(seed: int, seats: tuple[str, str]) -> tuple[list[int], bool]:
    """Play one game; return final scores by seat and whether it finished."""
    from skyjo_decision_env import SkyjoDecisionEnv

    env = SkyjoDecisionEnv(num_players=2, seed=seed, setup_mode="auto")
    env.reset()
    players = [_make_player(kind, pid, seed) for pid, kind in enumerate(seats)]
    ours = [p if isinstance(p, Player) else None for p in players]
    flipped_round = 0
    for _ in range(MAX_DECISIONS):
        if env.base.game_over:
            break
        # The env deals and auto-flips each new round inside `step`.
        if env.base.round_index != flipped_round:
            choose_setup_flips(env, ours)
            flipped_round = env.base.round_index
        pid = env.current_player
        if ours[pid] is None:
            env.step(players[pid](env))
            continue
        action = ours[pid].select_action(to_observation(env, pid), legal_actions(env))
        for decision in to_decision_actions(env, action):
            env.step(decision)
    return list(env.scores), env.base.game_over


def _play_job(job: tuple[int, tuple[str, str], int]) -> tuple[int, list[int], bool]:
    seed, seats, player_seat = job
    scores, finished = play_game(seed, seats)
    return player_seat, scores, finished


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    centre = (wins + z * z / 2) / (n + z * z)
    half = z * math.sqrt(wins * (n - wins) / n + z * z / 4) / (n + z * z)
    return centre - half, centre + half


def play_matchup(pool, player: str, opponent: str) -> dict:
    """Play `GAMES` games of `player` vs `opponent`; tally from `player`'s side."""
    jobs = [
        (seed, seats, player_seat)
        for seed in range(GAMES // 2)
        for seats, player_seat in (((player, opponent), 0), ((opponent, player), 1))
    ]
    wins = losses = draws = unfinished = 0
    margins = []
    for player_seat, scores, finished in tqdm(
        pool.imap_unordered(_play_job, jobs),
        total=len(jobs),
        unit="game",
        desc=f"{player} vs {opponent}",
    ):
        own, other = scores[player_seat], scores[1 - player_seat]
        margins.append(other - own)
        unfinished += not finished
        # Lower total wins; draws count as games but as nobody's win.
        if own < other:
            wins += 1
        elif other < own:
            losses += 1
        else:
            draws += 1

    n = len(jobs)
    margin = sum(margins) / n
    margin_sd = math.sqrt(sum((m - margin) ** 2 for m in margins) / (n - 1))
    return {
        "opponent": opponent,
        "games": n,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "unfinished": unfinished,
        "margin": margin,
        "margin_ci": 1.96 * margin_sd / math.sqrt(n),
    }


def print_results(player: str, rows: list[dict]) -> None:
    name = model_display_name(MODEL) if player == "rl" else player
    print(f"\n{name}, {GAMES} games per opponent:")
    for r in rows:
        n = r["games"]
        low, high = wilson_ci(r["wins"], n)
        print(
            f"  vs {r['opponent']:<31} win {r['wins'] / n:6.1%} ({low:.1%}-{high:.1%})  "
            f"W/L/D {r['wins']}/{r['losses']}/{r['draws']}  "
            f"margin {r['margin']:+6.1f} ± {r['margin_ci']:.1f}  "
            f"unfinished {r['unfinished']}/{n}"
        )
    if len(rows) > 1:
        mean = sum(r["wins"] / r["games"] for r in rows) / len(rows)
        print(f"  mean win rate: {mean:.1%}")
