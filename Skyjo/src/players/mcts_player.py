from concurrent.futures import ThreadPoolExecutor
import os
from typing import List, Optional
from Skyjo.src.mcts.mcts_algorithm import MCTSAlgorithm
from Skyjo.src.players.player import Player
from Skyjo.src.action import Action
from Skyjo.src.observation import Observation

class MCTSPlayer(Player):
    def __init__(
        self,
        player_id: int,
        player_name: str = "MCTSPlayer",
        iterations: int = 200,
        exploration_weight: float = 2.0,
        max_depth: Optional[int] = 15,
        parallel: bool = False,
        num_threads: Optional[int] = None
       ):
        super().__init__(player_id, player_name)
        self.iterations = iterations
        self.parallel = parallel
        self.num_threads = num_threads if num_threads is not None else min(4, os.cpu_count() or 1)

        self.mcts_algorithm = MCTSAlgorithm(
            exploration_weight=exploration_weight,
            max_depth=max_depth
        )

    def select_action(
        self, observation: Observation, legal_actions: List[Action]
    ) -> Action:
        """Select the best action using MCTS with multi-turn simulation."""
        if not legal_actions:
            raise ValueError("No legal actions available")

        if len(legal_actions) == 1:
            action = legal_actions[0]
            return action

        root = self.mcts_algorithm._create_root_node(observation, legal_actions)

        if self.parallel and self.iterations >= 50:
            batch_size = max(20, self.num_threads * 5)
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                for batch_start in range(0, self.iterations, batch_size):
                    batch_end = min(batch_start + batch_size, self.iterations)
                    nodes = []
                    for _ in range(batch_start, batch_end):
                        node = self.mcts_algorithm._select(root)
                        nodes.append(node)

                    rewards = list(executor.map(self.mcts_algorithm._simulate, nodes))

                    for node, reward in zip(nodes, rewards):
                        self.mcts_algorithm._backpropagate(node, reward)
        else:
            for _ in range(self.iterations):
                node = self.mcts_algorithm._select(root)
                reward = self.mcts_algorithm._simulate(node)
                self.mcts_algorithm._backpropagate(node, reward)

        best_action = max(
            root.children,
            key=lambda c: c.total_reward / c.visits if c.visits > 0 else float("-inf"),
        ).action
        return best_action
