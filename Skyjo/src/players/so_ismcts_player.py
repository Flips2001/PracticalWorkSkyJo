"""Single-Observer Information Set MCTS (SO-ISMCTS) player for Skyjo.

Skyjo is a game of imperfect information: a player does not know the value of
their own face-down cards, the opponents' face-down cards, or the order of the
draw pile. Plain MCTS cannot be applied directly because it assumes a fully
observable state.

SO-ISMCTS handles this by *determinization*: on every search iteration we sample
one complete world consistent with everything the acting player can observe
(``mcts_common.determinize``), then run one MCTS iteration on it. A *single*
tree is shared across all determinizations; because different determinizations
make different actions legal, the UCB exploration term uses an *availability*
count (how often an action was legal at a node) instead of the parent visit
count. That shared tree is the distinguishing feature versus the PIMC player,
which builds an independent tree per determinization.

Each iteration performs the four classic MCTS phases:

    select      -> descend the tree using UCB, restricted to actions that are
                   legal in the current determinization.
    expand      -> add one previously untried legal action as a new node.
    simulate    -> play the determinized round to the end (fast rollout policy).
    backpropagate -> push the round outcome back up the visited path.

The shared machinery (determinization, the engine driver, rollouts and scoring)
lives in ``mcts_common`` and is reused by all MCTS-family players.

Public API: this module exposes a single entry point, ``MCTSPlayer.select_action``.
"""

import math
import random
from typing import List, Optional, Tuple

from Skyjo.src.action import Action
from Skyjo.src.observation import Observation
from Skyjo.src.mcts import mcts_common as mc
from Skyjo.src.mcts.mcts_common import Node
from Skyjo.src.players.player import Player
from Skyjo.src.skyjo_game import SkyjoGame
from Skyjo.src.turn_phase import TurnPhase


class SOISMCTSPlayer(Player):
    """Skyjo player driven by Single-Observer Information Set MCTS."""

    def __init__(
        self,
        player_id: int,
        player_name: str,
        num_iterations: int = 1000,
        exploration: float = math.sqrt(2),
        rollout_max_turns: int = mc.ROLLOUT_MAX_TURNS,
        pw_c: Optional[float] = 2.0,
        pw_alpha: float = 0.4,
    ):
        """
        :param num_iterations: search iterations (determinizations) per decision.
        :param exploration: UCB exploration constant ``c``.
        :param rollout_max_turns: hard cap on turns simulated in a single rollout.
        :param pw_c: progressive-widening coefficient; ``None`` disables widening
            (every legal action is expandable, the classic behaviour).
        :param pw_alpha: progressive-widening exponent.
        """
        super().__init__(player_id, player_name)
        self.num_iterations = num_iterations
        self.exploration = exploration
        self.rollout_max_turns = rollout_max_turns
        self.pw_c = pw_c
        self.pw_alpha = pw_alpha

    def _widening_limit(self, node_visits: int) -> float:
        """How many distinct actions this node may expand, given its visit count.

        Progressive widening: ``ceil(c * N^alpha)``. Because MCTS adds one node
        per iteration, capping the branching lets the same budget grow a much
        DEEPER tree instead of a wide, shallow one. Heavily visited nodes
        eventually unlock every action.
        """
        if self.pw_c is None:
            return math.inf
        return max(1, math.ceil(self.pw_c * (node_visits**self.pw_alpha)))

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def select_action(
        self, observation: Observation, legal_actions: List[Action]
    ) -> Action:
        """Choose an action using SO-ISMCTS.

        Trivial decisions are shortcut: a single legal action is returned
        directly, and the blind opening flips (``STARTING_FLIPS``) carry no
        information to search over, so a random flip is played.
        """
        if not legal_actions:
            raise ValueError("No legal actions available to select from.")
        if len(legal_actions) == 1:
            return legal_actions[0]
        if observation.turn_phase == TurnPhase.STARTING_FLIPS:
            return random.choice(legal_actions)

        root = Node(player_to_move=observation.player_id)
        for _ in range(self.num_iterations):
            # Sample a world consistent with the observation, then run one
            # select -> expand -> simulate -> backpropagate iteration on it.
            game = mc.determinize(observation)
            self._run_iteration(root, game)

        # Recommend the most-simulated action (robust choice), restricted to the
        # actions actually legal in the real game.
        return max(legal_actions, key=lambda a: root.child_N.get(a, 0))

    # ------------------------------------------------------------------ #
    # One ISMCTS iteration                                                #
    # ------------------------------------------------------------------ #
    def _run_iteration(self, root: Node, game: SkyjoGame) -> None:
        node = root
        path: List[Tuple[Node, Action]] = []
        gs = game.game_state

        # --- SELECT + EXPAND -------------------------------------------- #
        try:
            while True:
                player = game.players[gs.current_player_id]
                legal = game.get_legal_actions(player)
                if not legal:
                    break

                # Record availability of every legal action at this node.
                for a in legal:
                    node.child_avail[a] = node.child_avail.get(a, 0) + 1

                untried = [a for a in legal if a not in node.children]
                unlocked = [a for a in legal if a in node.children]
                # Progressive widening: only allow a new action once this node
                # has earned it, so the budget deepens the tree rather than
                # fanning it out across all ~13 moves.
                may_widen = len(unlocked) < self._widening_limit(
                    sum(node.child_N.values())
                )
                if untried and (may_widen or not unlocked):
                    # EXPAND: unlock the most promising untried action.
                    action = mc.action_priority(game, player, untried)[0]
                    path.append((node, action))
                    mc.apply_action(game, player, action)
                    child = Node(player_to_move=gs.current_player_id)
                    node.children[action] = child
                    node.child_N.setdefault(action, 0)
                    node.child_W.setdefault(action, 0.0)
                    break

                # SELECT: pick by UCB among the actions unlocked so far.
                action = self._ucb_select(node, unlocked)
                path.append((node, action))
                round_over = mc.apply_action(game, player, action)
                node = node.children[action]
                if round_over:
                    break
        except Exception:
            # A determinization edge case (e.g. an exhausted pile) aborts the
            # descent; we still score and back up whatever path we built.
            pass

        # --- SIMULATE --------------------------------------------------- #
        try:
            mc.rollout(game, self.rollout_max_turns)
        except Exception:
            pass

        # --- BACKPROPAGATE ---------------------------------------------- #
        rewards = mc.reward_vector(mc.round_scores(game))
        for nd, action in path:
            nd.child_N[action] = nd.child_N.get(action, 0) + 1
            nd.child_W[action] = (
                nd.child_W.get(action, 0.0) + rewards[nd.player_to_move]
            )

    def _ucb_select(self, node: Node, legal: List[Action]) -> Action:
        """Pick the legal action maximising the ISMCTS UCB1 value.

        The exploration numerator is the action's *availability* count, which is
        what makes this ISMCTS rather than plain UCT.
        """
        best_action: Optional[Action] = None
        best_value = -math.inf
        for a in legal:
            value = mc.ucb_score(node, a, self.exploration, node.child_avail.get(a, 1))
            if value > best_value:
                best_value = value
                best_action = a
        return best_action  # type: ignore[return-value]
