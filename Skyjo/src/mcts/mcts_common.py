"""Shared machinery for the Monte Carlo Tree Search family of Skyjo players.

Three players are built on top of this module and can be compared head to head:

  * ``SOISMCTSPlayer``       - Single-Observer Information Set MCTS (SO-ISMCTS):
                           one shared tree, a fresh determinization per iteration.
  * ``PIMCPlayer``       - Perfect-Information Monte Carlo / Determinized UCT:
                           N independent perfect-information trees, then a vote.
  * ``MOISMCTSPlayer``   - Multiple-Observer ISMCTS: one tree per player so that
                           opponents are modelled with their own information sets.

Everything that is common to all three lives here:

  * ``determinize`` - sample a fully specified ``SkyjoGame`` consistent with an
    observation (the hidden-information handling shared by every variant),
  * a thin driver over the *real* engine (``apply_action`` / ``advance_after_turn``
    / ``rollout``) so the simulated dynamics always match the actual rules,
  * ``round_scores`` / ``reward_vector`` for terminal evaluation,
  * a generic ``Node`` and ``ucb_score`` helper for the tree-based variants.

None of these helpers ever reads the true value of a face-down card, so the
searches cannot cheat: hidden values are always sampled during determinization.
"""

import math
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

from Skyjo.src.action import Action
from Skyjo.src.action_type import ActionType
from Skyjo.src.card import Card
from Skyjo.src.game_state import GameState
from Skyjo.src.observation import Observation
from Skyjo.src.players.player import Player
from Skyjo.src.skyjo_game import SkyjoGame
from Skyjo.src.turn_phase import TurnPhase

# Card values present in a Skyjo deck, ordered as the observation reports them.
CARD_VALUES = list(range(-2, 13))  # -2 .. 12

# Hard cap on the number of turns simulated in a single rollout.
ROLLOUT_MAX_TURNS = 200


# --------------------------------------------------------------------------- #
# Simulation player + generic search node                                     #
# --------------------------------------------------------------------------- #
class SimPlayer(Player):
    """Carries a grid inside a simulated game; its ``select_action`` is unused."""

    def select_action(self, observation, legal_actions):  # pragma: no cover
        return legal_actions[0]


class Node:
    """A generic MCTS tree node with per-action (edge) statistics.

    ``player_to_move`` is the player choosing at this node. Stats are stored per
    outgoing action so a single dict of children suffices for every variant:

        child_N[a]      simulations that traversed edge ``a``
        child_W[a]      summed reward (for ``player_to_move``) over those sims
        child_avail[a]  iterations in which ``a`` was legal here (ISMCTS only)
    """

    __slots__ = ("player_to_move", "children", "child_N", "child_W", "child_avail")

    def __init__(self, player_to_move: int = -1):
        self.player_to_move = player_to_move
        self.children: Dict[Action, "Node"] = {}
        self.child_N: Dict[Action, int] = {}
        self.child_W: Dict[Action, float] = {}
        self.child_avail: Dict[Action, int] = {}


def ucb_score(
    node: Node, action: Action, exploration: float, total_parent: int
) -> float:
    """UCB1 value of ``action`` at ``node``.

    ``total_parent`` is the count used in the exploration numerator: the node
    visit count for standard UCT (PIMC), or the action's availability count for
    ISMCTS. An unvisited action scores +inf so it is tried first.
    """
    n = node.child_N.get(action, 0)
    if n == 0:
        return math.inf
    exploit = node.child_W.get(action, 0.0) / n
    explore = exploration * math.sqrt(math.log(max(1, total_parent)) / n)
    return exploit + explore


# --------------------------------------------------------------------------- #
# Determinization: sample a concrete world from an observation                #
# --------------------------------------------------------------------------- #
# Deck composition, built once and copied per determinization (hot path).
_DECK_TEMPLATE: Dict[int, int] = {v: 10 for v in range(-1, 13)}
_DECK_TEMPLATE[-2] = 5
_DECK_TEMPLATE[0] += 5  # five extra 0 cards -> 15 total

# Actions that can change the grid and therefore trigger a column clear. After a
# draw/discard the grid is untouched, so the (relatively expensive) uniform-column
# scan can be skipped entirely.
_GRID_CHANGING = (ActionType.SWAP_CARD, ActionType.FLIP_CARD)

# When True, the rollout policy and action-ordering heuristic actively finish and
# build same-value columns (so the search values column clears). Toggle for A/B.
COLUMN_AWARE = True


def full_deck_counter() -> Counter:
    """The full Skyjo deck as a value -> count multiset (150 cards)."""
    return Counter(_DECK_TEMPLATE)


def _copy_grid(grid: List[List[Card]]) -> List[List[Card]]:
    """Shallow-per-card copy of a grid -- far cheaper than ``copy.deepcopy``."""
    return [[Card(c.value, c.face_up) for c in row] for row in grid]


# Mean value of an unknown card, used to score swaps onto face-down slots.
_EXPECTED_HIDDEN = sum(v * c for v, c in _DECK_TEMPLATE.items()) / sum(
    _DECK_TEMPLATE.values()
)


def action_priority(game: SkyjoGame, player: SimPlayer, actions: List[Action]):
    """Order actions best-first by a cheap immediate-gain heuristic.

    Progressive widening only unlocks a few actions per node, so the *order*
    matters: unlocking poor moves first would waste the narrowed budget. The
    score is the points a move is expected to remove from the player's grid
    (a face-down slot is worth ``_EXPECTED_HIDDEN``).
    """
    gs = game.game_state
    grid = player.player_state.grid
    hand = gs.hand_card
    hv = hand.value if hand is not None else _EXPECTED_HIDDEN

    def score(a: Action) -> float:
        if a.type == ActionType.SWAP_CARD and a.pos is not None:
            r, c = a.pos
            if r < len(grid) and c < len(grid[r]):
                card = grid[r][c]
                current = card.value if card.face_up else _EXPECTED_HIDDEN
                base = current - hv  # points removed by swapping
                # Column-clear bonus: reward finishing / building a positive column
                # so progressive widening does not starve these moves.
                if COLUMN_AWARE and hv > 0:
                    col_matches = sum(
                        1
                        for rr in range(len(grid))
                        if rr != r
                        and c < len(grid[rr])
                        and grid[rr][c].face_up
                        and grid[rr][c].value == hv
                    )
                    if col_matches == 2:
                        base += 3 * hv  # completes a clear
                    elif col_matches == 1:
                        base += hv  # builds a matching pair
                return base
            return 0.0
        if a.type == ActionType.DRAW_OPEN_CARD:
            top = gs.discard_pile[-1] if gs.discard_pile else None
            return _EXPECTED_HIDDEN - top.value if top is not None else 0.0
        # DISCARD / DRAW_HIDDEN / FLIP carry no immediate grid gain.
        return 0.0

    return sorted(actions, key=score, reverse=True)


def _blank_game(num_players: int) -> SkyjoGame:
    """A ``SkyjoGame`` with an empty ``GameState``, bypassing the 150-card deck
    that the normal constructor builds and shuffles -- determinize/clone overwrite
    the piles anyway, so building that deck is pure waste on the search hot path."""
    game = SkyjoGame.__new__(SkyjoGame)
    game.players = []
    game.num_players = num_players
    game.last_column_clear_stats = {}
    game.total_columns_cleared = {}
    game.total_column_clear_value_sum = {}

    gs = GameState.__new__(GameState)
    gs.round_number = 1
    gs.discard_pile = []
    gs.draw_pile = []
    gs.current_player_id = 0
    gs.is_game_over = False
    gs.all_player_final_scores = []
    gs.final_turn_phase = False
    gs.phase = TurnPhase.CHOOSE_DRAW
    gs.hand_card = None
    gs.round_start_flips = {}
    gs.first_finisher_id = None
    gs.players_to_finish = set()
    game.game_state = gs
    return game


def determinize(obs: Observation) -> SkyjoGame:
    """Build a fully-specified ``SkyjoGame`` consistent with ``obs``.

    Hidden grid cards (mine and the opponents') and the draw-pile order are the
    unknowns. Hidden values are *sampled* from the deck's remaining multiset so
    the caller never sees the real face-down cards.
    """
    num_players = len(obs.opponent_cards)

    # Assemble each player's grid (deep-copied so simulation can mutate it).
    grids: List[List[List[Card]]] = [None] * num_players  # type: ignore[list-item]
    grids[obs.player_id] = _copy_grid(obs.card_grid)
    for pid, opp_grid in enumerate(obs.opponent_cards):
        if opp_grid is not None:
            grids[pid] = _copy_grid(opp_grid)

    # Start from the full deck and remove everything whose value we can see.
    pool = full_deck_counter()
    hidden_slots: List[Tuple[int, int, int]] = []
    for pid, grid in enumerate(grids):
        for r, row in enumerate(grid):
            for c, card in enumerate(row):
                if card.face_up:
                    pool[card.value] -= 1
                else:
                    hidden_slots.append((pid, r, c))
    if obs.discard_top is not None:
        pool[obs.discard_top.value] -= 1
    if obs.hand_card is not None:
        pool[obs.hand_card.value] -= 1

    # The draw pile's multiset is public; only its order is unknown.
    draw_pile: List[Card] = []
    if obs.draw_pile_value_counts is not None:
        for idx, value in enumerate(CARD_VALUES):
            count = obs.draw_pile_value_counts[idx]
            pool[value] -= count
            draw_pile.extend(Card(value, face_up=False) for _ in range(count))
        random.shuffle(draw_pile)

    # Remaining pool = hidden grid cards (+ buried discards we do not model).
    remaining: List[int] = []
    for value, count in pool.items():
        if count > 0:
            remaining.extend([value] * count)
    random.shuffle(remaining)

    for i, (pid, r, c) in enumerate(hidden_slots):
        value = remaining[i] if i < len(remaining) else random.choice(CARD_VALUES)
        grids[pid][r][c] = Card(value, face_up=False)
    # Defensive fallback if the draw-pile multiset was unavailable.
    if obs.draw_pile_value_counts is None:
        leftover = remaining[len(hidden_slots) :]
        draw_pile = [Card(v, face_up=False) for v in leftover[: obs.draw_pile_size]]

    return _assemble_game(obs, grids, draw_pile, num_players)


def _assemble_game(
    obs: Observation,
    grids: List[List[List[Card]]],
    draw_pile: List[Card],
    num_players: int,
) -> SkyjoGame:
    """Wire the sampled cards into a runnable ``SkyjoGame`` at ``obs``'s state."""
    game = _blank_game(num_players)
    for pid in range(num_players):
        sim_player = SimPlayer(pid, f"sim_{pid}")
        sim_player.player_state.grid = grids[pid]
        game.players.append(sim_player)

    gs: GameState = game.game_state
    gs.draw_pile = draw_pile
    gs.discard_pile = (
        [Card(obs.discard_top.value, True)] if obs.discard_top is not None else []
    )
    gs.hand_card = (
        Card(obs.hand_card.value, obs.hand_card.face_up)
        if obs.hand_card is not None
        else None
    )
    gs.phase = obs.turn_phase
    gs.current_player_id = obs.player_id
    gs.final_turn_phase = obs.final_turn_phase
    gs.first_finisher_id = obs.first_finisher_id
    gs.players_to_finish = (
        {j for j in range(num_players) if j != obs.first_finisher_id}
        if obs.final_turn_phase
        else set()
    )
    gs.round_start_flips = {i: 2 for i in range(num_players)}
    return game


def clone_game(game: SkyjoGame) -> SkyjoGame:
    """Fast deep clone of a simulated game (cheaper than ``copy.deepcopy``).

    Needed by PIMC, which fixes one determinization and must restart many
    independent simulations from it.
    """
    new = _blank_game(game.num_players)
    for p in game.players:
        sp = SimPlayer(p.player_id, p.player_name)
        sp.player_state.grid = [
            [Card(c.value, c.face_up) for c in row] for row in p.player_state.grid
        ]
        new.players.append(sp)

    gs, ngs = game.game_state, new.game_state
    ngs.draw_pile = [Card(c.value, c.face_up) for c in gs.draw_pile]
    ngs.discard_pile = [Card(c.value, c.face_up) for c in gs.discard_pile]
    ngs.hand_card = (
        Card(gs.hand_card.value, gs.hand_card.face_up)
        if gs.hand_card is not None
        else None
    )
    ngs.phase = gs.phase
    ngs.current_player_id = gs.current_player_id
    ngs.final_turn_phase = gs.final_turn_phase
    ngs.first_finisher_id = gs.first_finisher_id
    ngs.players_to_finish = set(gs.players_to_finish)
    ngs.round_start_flips = dict(gs.round_start_flips)
    return new


# --------------------------------------------------------------------------- #
# Engine driver (reusing the real SkyjoGame rules)                            #
# --------------------------------------------------------------------------- #
def apply_action(game: SkyjoGame, player: SimPlayer, action: Action) -> bool:
    """Execute one action, clear columns and advance turns as the engine does.

    Returns True if the applied action ended the round.
    """
    game.execute_action(player, action)
    if action.type in _GRID_CHANGING:
        game.game_state.remove_uniform_columns_to_discard_pile(player.player_state)
    if game.game_state.phase == TurnPhase.END_TURN:
        return advance_after_turn(game)
    return False


def advance_after_turn(game: SkyjoGame) -> bool:
    """Mirror ``SkyjoGame.play_round`` bookkeeping between turns; report round end."""
    gs = game.game_state
    states = game.get_all_player_states()
    if gs.final_turn_phase and gs.current_player_id in gs.players_to_finish:
        gs.players_to_finish.discard(gs.current_player_id)
    gs.current_player_id = (gs.current_player_id + 1) % game.num_players
    round_over = gs.is_round_over(states)
    gs.phase = TurnPhase.CHOOSE_DRAW
    return round_over


# Cached immutable actions for the rollout policy (Action is frozen -> shareable).
_ROLL_DRAW_HIDDEN = Action(ActionType.DRAW_HIDDEN_CARD)
_ROLL_DRAW_OPEN = Action(ActionType.DRAW_OPEN_CARD)
_ROLL_DISCARD = Action(ActionType.DISCARD_CARD)
_ROLL_SWAP = {
    (r, c): Action(ActionType.SWAP_CARD, pos=(r, c)) for r in range(3) for c in range(4)
}
_ROLL_FLIP = {
    (r, c): Action(ActionType.FLIP_CARD, pos=(r, c)) for r in range(3) for c in range(4)
}


def rollout(game: SkyjoGame, max_turns: int = ROLLOUT_MAX_TURNS) -> None:
    """Play the determinized round to the end with the fast heuristic policy."""
    gs = game.game_state
    if gs.is_round_over(game.get_all_player_states()):
        return
    execute = game.execute_action
    players = game.players
    turns = 0
    while turns < max_turns:
        player = players[gs.current_player_id]
        ps = player.player_state
        while gs.phase != TurnPhase.END_TURN:
            action = rollout_policy(game, player)
            if action is None:
                break
            execute(player, action)
            if action.type in _GRID_CHANGING:
                gs.remove_uniform_columns_to_discard_pile(ps)
        if advance_after_turn(game):
            break
        turns += 1


def rollout_policy(game: SkyjoGame, player: SimPlayer) -> Optional[Action]:
    """A cheap, greedy default policy that selects its action *directly* from the
    game state -- no legal-action list is built, which is what makes rollouts fast.

      * CHOOSE_DRAW: take the discard top if it is low (<= 4), else draw blind.
      * after drawing: swap onto the highest revealed card if the hand card beats
        it; discard a clearly bad hand (> 5) to force a flip; otherwise place the
        hand card onto the first hidden slot.
      * forced flip: flip a random hidden card.

    Returns ``None`` only when no legal move exists (the caller ends the turn).
    """
    gs = game.game_state
    phase = gs.phase
    grid = player.player_state.grid

    if phase == TurnPhase.CHOOSE_DRAW:
        discard = gs.discard_pile
        top = discard[-1] if discard else None
        can_open = bool(discard)
        can_hidden = bool(gs.draw_pile) or len(discard) > 1
        if top is not None and top.value <= 4 and can_open:
            return _ROLL_DRAW_OPEN
        if can_hidden:
            return _ROLL_DRAW_HIDDEN
        return _ROLL_DRAW_OPEN if can_open else None

    if phase == TurnPhase.HAVE_DRAWN_HIDDEN or phase == TurnPhase.HAVE_DRAWN_OPEN:
        hand = gs.hand_card
        hv = hand.value if hand is not None else None
        # Single grid pass: highest revealed card and the first hidden slot.
        max_val = None
        max_pos = None
        first_hidden = None
        for r, row in enumerate(grid):
            for c, card in enumerate(row):
                if card.face_up:
                    if max_val is None or card.value > max_val:
                        max_val, max_pos = card.value, (r, c)
                elif first_hidden is None:
                    first_hidden = (r, c)

        # Column-clear awareness. Only worthwhile for positive values -- clearing
        # a column of negatives/zeros would raise our score, so we never chase it.
        complete_pos = None  # placing hv here finishes three-of-a-kind -> clears
        build_pos = None  # a hidden slot that stacks a matching pair
        if COLUMN_AWARE and hv is not None and hv > 0 and grid:
            nrows, ncols = len(grid), len(grid[0])
            for c in range(ncols):
                matches = [
                    r
                    for r in range(nrows)
                    if grid[r][c].face_up and grid[r][c].value == hv
                ]
                if len(matches) == 2 and complete_pos is None:
                    complete_pos = (
                        next(r for r in range(nrows) if r not in matches),
                        c,
                    )
                elif len(matches) == 1 and build_pos is None:
                    hid = [r for r in range(nrows) if not grid[r][c].face_up]
                    if hid:
                        build_pos = (hid[0], c)

        if hand is not None:
            if complete_pos is not None:  # immediate clear -- always take it
                return _ROLL_SWAP.get(complete_pos) or Action(
                    ActionType.SWAP_CARD, pos=complete_pos
                )
            if max_pos is not None and hand.value < max_val:
                return _ROLL_SWAP.get(max_pos) or Action(
                    ActionType.SWAP_CARD, pos=max_pos
                )
            if (
                hand.value > 5
                and phase == TurnPhase.HAVE_DRAWN_HIDDEN
                and first_hidden is not None
            ):
                return _ROLL_DISCARD  # discard then flip
            if build_pos is not None:  # stack a matching pair toward a future clear
                return _ROLL_SWAP.get(build_pos) or Action(
                    ActionType.SWAP_CARD, pos=build_pos
                )
            if first_hidden is not None:
                return _ROLL_SWAP.get(first_hidden) or Action(
                    ActionType.SWAP_CARD, pos=first_hidden
                )
            if max_pos is not None:
                return _ROLL_SWAP.get(max_pos) or Action(
                    ActionType.SWAP_CARD, pos=max_pos
                )
        return None

    if phase == TurnPhase.HAVE_TO_FLIP_AFTER_DISCARD:
        hidden = [
            (r, c)
            for r, row in enumerate(grid)
            for c, card in enumerate(row)
            if not card.face_up
        ]
        if not hidden:
            return None
        pos = random.choice(hidden)
        return _ROLL_FLIP.get(pos) or Action(ActionType.FLIP_CARD, pos=pos)

    return None


# --------------------------------------------------------------------------- #
# Terminal evaluation                                                         #
# --------------------------------------------------------------------------- #
def round_scores(game: SkyjoGame) -> List[int]:
    """Score every player as the engine would at round end.

    All grid cards are summed (revealed and determinized-hidden alike), matching
    ``SkyjoGame.reset`` which flips everything face-up before scoring. The
    first-finisher doubling penalty is applied when it holds.
    """
    states = game.get_all_player_states()
    scores = [sum(card.value for row in s.grid for card in row) for s in states]
    ff = game.game_state.first_finisher_id
    if ff is not None and scores[ff] > 0 and scores[ff] != min(scores):
        scores[ff] *= 2
    return scores


# Reward shaping for terminal evaluation. "margin" is the default: it rewards a
# player for the *size* of its lead (or deficit) over the field, so lowering
# one's own score always helps -- even in rounds it is losing. "winloss" is the
# original coarse rank reward (win = 1 / tie = 0.5 / loss = 0), kept for A/B
# comparison; its flat 0 in every lost round starves the search of gradient
# against a strong opponent.
REWARD_MODE = "margin"
REWARD_SCALE = 10.0  # points of score margin worth ~0.23 reward around a tie


def reward_vector(scores: List[int]) -> List[float]:
    """Map final round scores to per-player rewards in [0, 1] (lower is better).

    Controlled by the module-level ``REWARD_MODE`` / ``REWARD_SCALE``:

      * ``margin``  reward_i = sigmoid((mean opponent score - own score)/scale);
                    smooth, monotonically decreasing in one's own score.
      * ``winloss`` fraction of opponents beaten (ties count as half).
    """
    n = len(scores)
    if n <= 1:
        return [1.0] * n

    if REWARD_MODE == "winloss":
        rewards = []
        for i in range(n):
            wins = 0.0
            for j in range(n):
                if i == j:
                    continue
                if scores[i] < scores[j]:
                    wins += 1.0
                elif scores[i] == scores[j]:
                    wins += 0.5
            rewards.append(wins / (n - 1))
        return rewards

    # margin (default)
    rewards = []
    for i in range(n):
        others = [scores[j] for j in range(n) if j != i]
        margin = sum(others) / len(others) - scores[i]
        rewards.append(1.0 / (1.0 + math.exp(-margin / REWARD_SCALE)))
    return rewards
