"""Multiple-Observer Information Set MCTS (MO-ISMCTS) player for Skyjo.

SO-ISMCTS (``MCTSPlayer``) grows a *single* tree from the searching player's
point of view, so every player's move -- including the opponents' -- is chosen
from the same statistics. That implicitly models opponents as if they shared the
root player's information. MO-ISMCTS instead keeps *one tree per player*:

    * one determinization is sampled per iteration (from the root player's
      information set), exactly as in SO-ISMCTS;
    * all trees are descended in lockstep along the same (publicly observable)
      action path, but at each decision the *acting* player selects using
      *their own* tree -- their own visit/availability statistics and their own
      reward.

So each opponent is modelled as an agent optimising its own Skyjo score from its
own information set, rather than as a puppet of the root player's tree. This is
the more faithful (and heavier) variant from Cowling, Powley & Whitehouse (2012).

In Skyjo every action and every revealed card is public, so the trees share the
same action-path structure; what differs between them is the statistics each
player accumulates. The extra fidelity mainly matters for the hidden-draw
window, where the drawer knows the value of a blind-drawn card that opponents
cannot yet see.

Public API: a single entry point, ``MOISMCTSPlayer.select_action``.
"""

import math
import random
from typing import List, Optional, Tuple

from Skyjo.src.action import Action
from Skyjo.src.observation import Observation
from Skyjo.src.players import mcts_common as mc
from Skyjo.src.players.mcts_common import Node
from Skyjo.src.players.player import Player
from Skyjo.src.skyjo_game import SkyjoGame
from Skyjo.src.turn_phase import TurnPhase


class MOISMCTSPlayer(Player):
    """Skyjo player driven by Multiple-Observer Information Set MCTS."""

    def __init__(
        self,
        player_id: int,
        player_name: str,
        num_iterations: int = 1000,
        exploration: float = math.sqrt(2),
        rollout_max_turns: int = mc.ROLLOUT_MAX_TURNS,
    ):
        """
        :param num_iterations: search iterations (determinizations) per decision.
        :param exploration: UCB exploration constant ``c``.
        :param rollout_max_turns: hard cap on turns simulated in a single rollout.
        """
        super().__init__(player_id, player_name)
        self.num_iterations = num_iterations
        self.exploration = exploration
        self.rollout_max_turns = rollout_max_turns

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def select_action(
        self, observation: Observation, legal_actions: List[Action]
    ) -> Action:
        """Choose an action using MO-ISMCTS."""
        if not legal_actions:
            raise ValueError("No legal actions available to select from.")
        if len(legal_actions) == 1:
            return legal_actions[0]
        if observation.turn_phase == TurnPhase.STARTING_FLIPS:
            return random.choice(legal_actions)

        num_players = len(observation.opponent_cards)
        # One root per player; all rooted at the same current game state.
        roots = [Node() for _ in range(num_players)]

        for _ in range(self.num_iterations):
            game = mc.determinize(observation)
            self._run_iteration(roots, game)

        # Decide using the searching player's own tree.
        my_root = roots[observation.player_id]
        return max(legal_actions, key=lambda a: my_root.child_N.get(a, 0))

    # ------------------------------------------------------------------ #
    # One MO-ISMCTS iteration                                             #
    # ------------------------------------------------------------------ #
    def _run_iteration(self, roots: List[Node], game: SkyjoGame) -> None:
        gs = game.game_state
        # Current node within each player's tree; all advance in lockstep.
        current = list(roots)
        # Decisions to update in backprop: (acting_player, that player's node, action).
        path: List[Tuple[int, Node, Action]] = []

        # --- SELECT + EXPAND -------------------------------------------- #
        try:
            while True:
                p = gs.current_player_id
                player = game.players[p]
                legal = game.get_legal_actions(player)
                if not legal:
                    break

                node_p = current[p]
                # Availability is tracked in the acting player's own tree only.
                for a in legal:
                    node_p.child_avail[a] = node_p.child_avail.get(a, 0) + 1

                untried = [a for a in legal if a not in node_p.children]
                expanding = bool(untried)
                if expanding:
                    action = random.choice(untried)
                else:
                    action = self._ucb_select(node_p, legal)

                path.append((p, node_p, action))

                # Advance EVERY tree along the chosen (public) action so each
                # player's decision nodes stay keyed by the full action history.
                for i in range(len(current)):
                    child = current[i].children.get(action)
                    if child is None:
                        child = Node()
                        current[i].children[action] = child
                        current[i].child_N.setdefault(action, 0)
                        current[i].child_W.setdefault(action, 0.0)
                    current[i] = child

                round_over = mc.apply_action(game, player, action)
                if expanding or round_over:
                    break
        except Exception:
            pass

        # --- SIMULATE --------------------------------------------------- #
        try:
            mc.rollout(game, self.rollout_max_turns)
        except Exception:
            pass

        # --- BACKPROPAGATE ---------------------------------------------- #
        rewards = mc.reward_vector(mc.round_scores(game))
        for actor, node, action in path:
            node.child_N[action] = node.child_N.get(action, 0) + 1
            node.child_W[action] = node.child_W.get(action, 0.0) + rewards[actor]

    def _ucb_select(self, node: Node, legal: List[Action]) -> Action:
        """Pick the legal action maximising the ISMCTS UCB1 value (availability)."""
        best_action: Optional[Action] = None
        best_value = -math.inf
        for a in legal:
            value = mc.ucb_score(node, a, self.exploration, node.child_avail.get(a, 1))
            if value > best_value:
                best_value = value
                best_action = a
        return best_action  # type: ignore[return-value]
