import threading
import math
from typing import List, Optional, Dict, Any
from Skyjo.src.action import Action

class MCTSNode:
    def __init__(
        self,
        state: Dict[str, Any],
        parent: Optional["MCTSNode"] = None,
        action: Optional[Action] = None,
        children: Optional[List["MCTSNode"]] = None,
        visits: int = 0,
        total_reward: float = 0.0,
        untried_actions: Optional[List[Action]] = None,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = children if children is not None else []
        self.visits = visits
        self.total_reward = total_reward
        self.untried_actions = untried_actions if untried_actions is not None else []
        self._lock = threading.Lock()

    def ucb_score(self, exploration_weight: float = 1.4) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.total_reward / self.visits
        parent_visits = max(1, self.parent.visits if self.parent else 1)
        exploration = exploration_weight * math.sqrt(
            math.log(parent_visits) / self.visits
        )
        return exploitation + exploration

    def best_child(self, exploration_weight: float = 1.4) -> "MCTSNode":
        return max(self.children, key=lambda child: child.ucb_score(exploration_weight))
    