"""Perfect-Information Monte Carlo (PIMC) / Determinized UCT player for Skyjo.

This is the classic approach ISMCTS is usually benchmarked *against*. Where
SO-ISMCTS grows one shared tree and re-determinizes every iteration, PIMC does
the opposite:

    for each of N determinizations:
        fix that world (all hidden cards known),
        build an INDEPENDENT perfect-information UCT tree and search it,
        record the root action visit counts.
    action = argmax over the summed visit counts (a vote across worlds).

Because each tree searches a world in which every hidden card is known, the
per-tree search is ordinary UCT (exploration uses the parent visit count, not an
availability count). Each simulation must restart from the fixed determinized
root, so the world is cloned per iteration.

Known weakness this exposes (the reason ISMCTS was invented): *strategy fusion*.
PIMC implicitly assumes it will know the hidden cards at future decision points
too, so it can plan a different perfect response in each world and then average
them, over-estimating its own control. ISMCTS's shared tree cannot do that,
because one node must commit to a single action across all determinizations.

Public API: a single entry point, ``PIMCPlayer.select_action``.
"""

import math
import random
from collections import Counter
from typing import List, Optional, Tuple

from Skyjo.src.action import Action
from Skyjo.src.observation import Observation
from Skyjo.src.mcts import mcts_common as mc
from Skyjo.src.mcts.mcts_common import Node
from Skyjo.src.players.player import Player
from Skyjo.src.skyjo_game import SkyjoGame
from Skyjo.src.turn_phase import TurnPhase


class PIMCPlayer(Player):
    """Skyjo player driven by Determinized UCT with root-action voting."""

    def __init__(
        self,
        player_id: int,
        player_name: str,
        num_determinizations: int = 40,
        iterations_per_world: int = 40,
        exploration: float = math.sqrt(2),
        rollout_max_turns: int = mc.ROLLOUT_MAX_TURNS,
    ):
        """
        The search budget is ``num_determinizations * iterations_per_world``,
        chosen here to roughly match ``MCTSPlayer``'s default of 1000 sims.

        :param num_determinizations: number of sampled worlds (independent trees).
        :param iterations_per_world: UCT iterations run within each world.
        :param exploration: UCB exploration constant ``c``.
        :param rollout_max_turns: hard cap on turns simulated in a single rollout.
        """
        super().__init__(player_id, player_name)
        self.num_determinizations = num_determinizations
        self.iterations_per_world = iterations_per_world
        self.exploration = exploration
        self.rollout_max_turns = rollout_max_turns

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def select_action(
        self, observation: Observation, legal_actions: List[Action]
    ) -> Action:
        """Choose an action by searching many worlds and voting on the root."""
        if not legal_actions:
            raise ValueError("No legal actions available to select from.")
        if len(legal_actions) == 1:
            return legal_actions[0]
        if observation.turn_phase == TurnPhase.STARTING_FLIPS:
            return random.choice(legal_actions)

        # Each world contributes its root action visit counts to a shared tally;
        # summing visits (rather than taking each world's single best) is a
        # robust vote that down-weights worlds where the search was undecided.
        votes: Counter = Counter()
        for _ in range(self.num_determinizations):
            world = mc.determinize(observation)
            root = self._search_world(world)
            for action, visits in root.child_N.items():
                votes[action] += visits

        legal_set = set(legal_actions)
        # Fall back to a legal default if no world produced usable statistics.
        if not votes:
            return random.choice(legal_actions)
        return max(
            (a for a in votes if a in legal_set),
            key=lambda a: votes[a],
            default=random.choice(legal_actions),
        )

    # ------------------------------------------------------------------ #
    # Perfect-information UCT within a single fixed world                 #
    # ------------------------------------------------------------------ #
    def _search_world(self, world: SkyjoGame) -> Node:
        """Run UCT in one fully-known world and return its populated root."""
        root = Node(player_to_move=world.game_state.current_player_id)
        for _ in range(self.iterations_per_world):
            # Restart from the fixed world each iteration (the determinization
            # is constant for this whole tree).
            game = mc.clone_game(world)
            self._uct_iteration(root, game)
        return root

    def _uct_iteration(self, root: Node, game: SkyjoGame) -> None:
        node = root
        path: List[Tuple[Node, Action]] = []
        gs = game.game_state

        # --- SELECT + EXPAND (standard UCT on a known world) ------------ #
        try:
            while True:
                player = game.players[gs.current_player_id]
                legal = game.get_legal_actions(player)
                if not legal:
                    break

                node_visits = sum(node.child_N.values())
                untried = [a for a in legal if a not in node.children]
                if untried:
                    action = random.choice(untried)
                    path.append((node, action))
                    mc.apply_action(game, player, action)
                    child = Node(player_to_move=gs.current_player_id)
                    node.children[action] = child
                    node.child_N.setdefault(action, 0)
                    node.child_W.setdefault(action, 0.0)
                    break

                action = self._ucb_select(node, legal, node_visits)
                path.append((node, action))
                round_over = mc.apply_action(game, player, action)
                node = node.children[action]
                if round_over:
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
        for nd, action in path:
            nd.child_N[action] = nd.child_N.get(action, 0) + 1
            nd.child_W[action] = (
                nd.child_W.get(action, 0.0) + rewards[nd.player_to_move]
            )

    def _ucb_select(self, node: Node, legal: List[Action], node_visits: int) -> Action:
        """Pick the legal action maximising the plain UCT (UCB1) value.

        Exploration uses the node visit count (perfect-information UCT), which is
        the key difference from the availability count used by SO-ISMCTS.
        """
        best_action: Optional[Action] = None
        best_value = -math.inf
        for a in legal:
            value = mc.ucb_score(node, a, self.exploration, node_visits)
            if value > best_value:
                best_value = value
                best_action = a
        return best_action  # type: ignore[return-value]
