from typing import List, Optional
from Skyjo.src.mcts.mcts_game_logic import MCTSGameLogic
from Skyjo.src.mcts.mcts_node import MCTSNode
from Skyjo.src.action import Action
from Skyjo.src.observation import Observation

class MCTSAlgorithm:
    def __init__(self, 
                exploration_weight: float = 2.0,
                max_depth: Optional[int] = 15
                ):
        
        self.mcts_game_logic = MCTSGameLogic()
        self.exploration_weight = exploration_weight
        self.max_depth = max_depth        
        self.EARLY_TERM_SCORE = 70
    
    def _create_root_node(
        self, observation: Observation, legal_actions: List[Action]
    ) -> MCTSNode:
        state = self.mcts_game_logic.state_manager.create_state_snapshot(observation)
        node = MCTSNode(state=state, untried_actions=list(legal_actions))
        return node

    def _select(self, node: MCTSNode) -> MCTSNode:
        current = node
        while current.untried_actions == [] and current.children:
            current = current.best_child(self.exploration_weight)

        if current.untried_actions:
            with current._lock:
                if current.untried_actions:
                    action = current.untried_actions.pop()
                else:
                    return current
            child_state = self.mcts_game_logic._apply_action(current.state, action)
            child_legal_actions = self.mcts_game_logic._get_legal_actions_from_state(child_state)
            child_node = MCTSNode(
                state=child_state,
                parent=current,
                action=action,
                untried_actions=list(child_legal_actions),
            )
            with current._lock:
                current.children.append(child_node)
            return child_node
        return current

    def _simulate(self, node: MCTSNode) -> float:
        """
        Multi-turn simulation with proper player switching.
        Uses deep state copying to prevent mutation.

        If max_depth is None, simulates until game end (no depth limit).
        """
        state = node.state
        depth = 0
        your_player_id = state["your_player_id"]

        # Working state - DEEP copy to prevent mutation
        # Lazy copy: Only copy current player's state initially
        working_state = {
            **state,
            "player_states": state["player_states"].copy(),
            "current_player": state["current_player"],
        }
        working_state["player_states"][state["current_player"]] = (
            self.mcts_game_logic.state_manager.deep_copy_player_state(
                state["player_states"][state["current_player"]]
            )
        )

        while self.max_depth is None or depth < self.max_depth:
            # Check if game is over
            if self.mcts_game_logic._is_game_terminal(working_state):
                break

            current_player = working_state["current_player"]
            player_state = working_state["player_states"][current_player]

            # Check if current player's turn is over
            if self.mcts_game_logic._is_player_turn_over(player_state):
                self.mcts_game_logic._end_player_turn(working_state, current_player)
                continue

            # Get legal actions for current player
            legal_actions = self.mcts_game_logic._get_legal_actions_from_state(working_state)
            if not legal_actions:
                self.mcts_game_logic._end_player_turn(working_state, current_player)
                continue

            # Early termination for all players (was your_player only)
            expected_score = self.mcts_game_logic._get_expected_score_from_player_state(
                player_state, working_state
            )
            if expected_score > self.EARLY_TERM_SCORE:
                self.mcts_game_logic._end_player_turn(working_state, current_player)
                continue

            if current_player == your_player_id:
                action = self.mcts_game_logic._get_your_action(legal_actions)
            else:
                action = self.mcts_game_logic._get_opponent_action(working_state, legal_actions)

            # Apply the action
            working_state = self.mcts_game_logic._apply_action(working_state, action)
            depth += 1

        reward = self.mcts_game_logic._calculate_final_reward(working_state, your_player_id)

        return reward

    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate reward up the tree. Thread-safe with per-node locks."""
        current = node
        while current is not None:
            with current._lock:
                current.visits += 1
                current.total_reward += reward
            current = current.parent
