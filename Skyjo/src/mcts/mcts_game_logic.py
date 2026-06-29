import random
import threading
from typing import List, Optional, Dict, Any
from Skyjo.src.action import Action
from Skyjo.src.action_type import ActionType
from Skyjo.src.card import Card
from Skyjo.src.turn_phase import TurnPhase
from Skyjo.src.mcts.state_manager import StateManager

_local = threading.local()

class MCTSGameLogic:
    def __init__(self):

        self.state_manager = StateManager()  
        self._draw_pile_lock = threading.Lock()
        self.FALLBACK_DECK = tuple(
            [v for _ in range(10) for v in range(-1, 13)] + [-2] * 5 + [0] * 5
        )

    def _thread_local_random(self):
        """Get thread-local random instance for thread safety."""
        if not hasattr(_local, "random"):
            _local.random = random.Random()
        return _local.random
    
    def _end_player_turn(self, state: Dict[str, Any], current_player: int) -> None:
        """Handle end of player's turn: update scores, switch player, reset state."""
        player_state = state["player_states"][current_player]
        round_score = self._get_expected_score_from_player_state(player_state, state)

        # Update total scores
        if state["total_scores"] and current_player < len(state["total_scores"]):
            total_scores_list = list(state["total_scores"])
            total_scores_list[current_player] += round_score
            state["total_scores"] = total_scores_list

        # Switch to next player
        num_players = state["num_players"]
        state["current_player"] = (current_player + 1) % num_players
        next_player = state["current_player"]
        next_player_state = state["player_states"][next_player]
        state["player_states"][next_player] = {
            "card_grid": self.state_manager.copy_card_grid(next_player_state["card_grid"]),
            "hand_card": None,
            "turn_phase": TurnPhase.CHOOSE_DRAW,
            "score": next_player_state["score"],
        }

        # Handle final turn phase
        self._handle_player_turn_end(state, current_player)

    def _get_your_action(self, legal_actions: List[Action]) -> Action:
        rng = self._thread_local_random()
        return rng.choice(legal_actions)

    def _get_opponent_action(
        self, state: Dict[str, Any], legal_actions: List[Action]
    ) -> Action:
        """Simple heuristic for opponent actions."""
        
        rng = self._thread_local_random()
        current_player = state["current_player"]
        player_state = state["player_states"][current_player]
        phase = player_state["turn_phase"]

        # Prefer drawing from discard if it's very good
        if phase == TurnPhase.CHOOSE_DRAW and state["shared_discard_top"]:
            if state["shared_discard_top"].value <= -1:
                return Action(ActionType.DRAW_OPEN_CARD)
            if state["shared_discard_top"].value < 0:
                return Action(ActionType.DRAW_OPEN_CARD)

        # In STARTING_FLIPS, prefer flipping cards
        if phase == TurnPhase.STARTING_FLIPS and legal_actions:
            # Just pick a random flip
            flip_actions = [a for a in legal_actions if a.type == ActionType.FLIP_CARD]
            if flip_actions:
                return rng.choice(flip_actions)

        return rng.choice(legal_actions)

    def _is_game_terminal(self, state: Dict[str, Any]) -> bool:
        """Check if game is terminal: any player >= 100 or all players finished in final turn phase."""
        total_scores = state.get("total_scores", [])
        if any(score >= 100 for score in total_scores):
            return True
        # Check if in final turn phase and all players have finished
        if state.get("final_turn_phase", False):
            players_to_finish = state.get("players_to_finish")
            if players_to_finish is not None and len(players_to_finish) == 0:
                return True
        return False

    def _handle_player_turn_end(
        self, working_state: Dict[str, Any], current_player: int
    ) -> None:
        """Handle end of player's turn, updating final turn phase and players_to_finish."""
        num_players = working_state["num_players"]

        if not working_state.get("final_turn_phase", False):
            # Check if this player finished (all cards face up) and trigger final turn phase
            player_state = working_state["player_states"][current_player]
            if self._is_player_turn_over(player_state):
                working_state["final_turn_phase"] = True
                working_state["first_finisher_id"] = current_player
                # All other players get one last move
                working_state["players_to_finish"] = set(
                    p_id for p_id in range(num_players) if p_id != current_player
                )
        else:
            # In final turn phase: remove current player from players_to_finish
            players_to_finish = working_state.get("players_to_finish")
            if players_to_finish is not None and current_player in players_to_finish:
                players_to_finish.discard(current_player)

    def _is_player_turn_over(self, player_state: Dict[str, Any]) -> bool:
        """Check if a player's turn is complete."""
        phase = player_state["turn_phase"]
        grid = player_state["card_grid"]

        # Check if all cards are face up
        all_face_up = True
        for row in grid:
            for card in row:
                if card and card.is_hidden():
                    all_face_up = False
                    break
            if not all_face_up:
                break

        return all_face_up or phase == TurnPhase.END_TURN

    def _apply_action(self, state: Dict[str, Any], action: Action) -> Dict[str, Any]:
        """
        Apply action to state, modifying only the current player's state.
        Returns a NEW state dict.
        """
        current_player = state["current_player"]

        # Create new state with deep copies of mutable fields
        new_state = {**state}

        # Only deep-copy the current player's state (others are unchanged)
        new_player_states = state["player_states"].copy()
        new_player_states[current_player] = self.state_manager.deep_copy_player_state(
            state["player_states"][current_player]
        )
        new_state["player_states"] = new_player_states

        # Deep copy mutable lists/sets
        if (
            "shared_draw_pile_value_counts" in state
            and state["shared_draw_pile_value_counts"]
        ):
            new_state["shared_draw_pile_value_counts"] = state[
                "shared_draw_pile_value_counts"
            ][:]
        if "total_scores" in state and state["total_scores"]:
            new_state["total_scores"] = state["total_scores"][:]
        if "players_to_finish" in state and state["players_to_finish"]:
            new_state["players_to_finish"] = set(state["players_to_finish"])

        player_state = new_state["player_states"][current_player]
        grid = player_state["card_grid"]

        # Apply action
        if action.type == ActionType.DRAW_HIDDEN_CARD:
            player_state["turn_phase"] = TurnPhase.HAVE_DRAWN_HIDDEN
            drawn_card = self._draw_random_card(new_state)
            if drawn_card is not None:
                player_state["hand_card"] = drawn_card
                with self._draw_pile_lock:
                    if new_state["shared_draw_pile_size"] > 0:
                        new_state["shared_draw_pile_size"] -= 1

        elif action.type == ActionType.DRAW_OPEN_CARD:
            player_state["turn_phase"] = TurnPhase.HAVE_DRAWN_OPEN
            if new_state["shared_discard_top"]:
                player_state["hand_card"] = self.state_manager.copy_card(
                    new_state["shared_discard_top"]
                )
                new_state["shared_discard_top"] = None

        elif action.type == ActionType.SWAP_CARD:
            if action.pos and player_state["hand_card"]:
                player_state["turn_phase"] = TurnPhase.END_TURN
                r, c = action.pos
                if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
                    grid_card = grid[r][c]
                    # Discard the grid card
                    if grid_card:
                        discarded = self.state_manager.copy_card(grid_card)
                        discarded.face_up = True
                        new_state["shared_discard_top"] = discarded
                    # Place hand card in grid
                    grid[r][c] = self.state_manager.copy_card(player_state["hand_card"])
                    grid[r][c].face_up = True
                    player_state["hand_card"] = None

        elif action.type == ActionType.DISCARD_CARD:
            if player_state["hand_card"]:
                discarded = self.state_manager.copy_card(player_state["hand_card"])
                discarded.face_up = True
                new_state["shared_discard_top"] = discarded
                player_state["hand_card"] = None
                player_state["turn_phase"] = TurnPhase.HAVE_TO_FLIP_AFTER_DISCARD

        elif action.type == ActionType.FLIP_CARD:
            if action.pos:
                r, c = action.pos
                if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
                    card = grid[r][c]
                    if card:
                        # Create new revealed card
                        grid[r][c] = Card(value=card.value, face_up=True)
                        if (
                            player_state["turn_phase"]
                            == TurnPhase.HAVE_TO_FLIP_AFTER_DISCARD
                        ):
                            player_state["turn_phase"] = TurnPhase.END_TURN
                        elif player_state["turn_phase"] == TurnPhase.STARTING_FLIPS:
                            # Check if all cards are now face up
                            all_face_up = all(
                                card and card.face_up for row in grid for card in row
                            )
                            if all_face_up:
                                player_state["turn_phase"] = TurnPhase.CHOOSE_DRAW

        return new_state

    def _get_legal_actions_from_state(self, state: Dict[str, Any]) -> List[Action]:
        """Get legal actions for current player."""

        current_player = state["current_player"]

        player_state = state["player_states"][current_player]
        grid = player_state["card_grid"]
        phase = player_state["turn_phase"]

        legal_actions = []

        if phase == TurnPhase.STARTING_FLIPS:
            for r, row in enumerate(grid):
                for c, card in enumerate(row):
                    if card and card.is_hidden():
                        legal_actions.append(
                            Action(type=ActionType.FLIP_CARD, pos=(r, c))
                        )

        elif phase == TurnPhase.CHOOSE_DRAW:
            legal_actions.append(Action(type=ActionType.DRAW_HIDDEN_CARD))
            if state["shared_discard_top"]:
                legal_actions.append(Action(type=ActionType.DRAW_OPEN_CARD))

        elif phase == TurnPhase.HAVE_DRAWN_HIDDEN:
            for r, row in enumerate(grid):
                for c, card in enumerate(row):
                    if card is not None:
                        legal_actions.append(
                            Action(type=ActionType.SWAP_CARD, pos=(r, c))
                        )
            has_hidden = any(card.is_hidden() for row in grid for card in row if card)
            if has_hidden:
                legal_actions.append(Action(type=ActionType.DISCARD_CARD))

        elif phase == TurnPhase.HAVE_DRAWN_OPEN:
            for r, row in enumerate(grid):
                for c, card in enumerate(row):
                    if card is not None:
                        legal_actions.append(
                            Action(type=ActionType.SWAP_CARD, pos=(r, c))
                        )

        elif phase == TurnPhase.HAVE_TO_FLIP_AFTER_DISCARD:
            for r, row in enumerate(grid):
                for c, card in enumerate(row):
                    if card and card.is_hidden():
                        legal_actions.append(
                            Action(type=ActionType.FLIP_CARD, pos=(r, c))
                        )

        elif phase == TurnPhase.END_TURN:
            return []

        return legal_actions

    def _get_expected_score_from_player_state(
        self, player_state: Dict[str, Any], state: Dict[str, Any]
    ) -> float:
        """Calculate expected score for a player."""
        grid = player_state["card_grid"]
        cols_to_remove = self._find_uniform_columns(grid)

        score = 0.0
        for row_idx in range(len(grid)):
            for col_idx in range(len(grid[row_idx])):
                card = grid[row_idx][col_idx]
                if col_idx in cols_to_remove:
                    continue
                if card and card.face_up:
                    score += card.value
                elif card and card.is_hidden():
                    expected_value = self._get_expected_card_value(state)
                    score += expected_value
        return score

    def _find_uniform_columns(self, grid: List[List[Optional[Card]]]) -> List[int]:
        """Find columns where all cards have the same value and are face-up."""
        cols_to_remove = []

        if not grid or len(grid) == 0:
            return cols_to_remove

        num_rows = len(grid)
        num_cols = len(grid[0]) if num_rows > 0 else 0

        for col in range(num_cols):
            first_card = grid[0][col]
            if first_card is None or not first_card.face_up:
                continue

            uniform = True
            for row in range(1, num_rows):
                card = grid[row][col]
                # Check face_up before comparing values
                if card is None or not card.face_up:
                    uniform = False
                    break
                if card.value != first_card.value:
                    uniform = False
                    break

            if uniform:
                cols_to_remove.append(col)

        return cols_to_remove

    def _get_expected_card_value(self, state: Dict[str, Any]) -> float:
        """Get expected card value from deck composition."""
        counts = state.get("shared_draw_pile_value_counts")
        if counts:
            values = list(range(-2, 13))
            total = sum(counts)
            if total > 0:
                return sum(v * c for v, c in zip(values, counts)) / total
        return 5.0667

    def _draw_random_card(self, state: Dict[str, Any]) -> Optional[Card]:
        """Draw a random card from the deck and update counts. Thread-safe."""
        rng = self._thread_local_random()
        counts = state.get("shared_draw_pile_value_counts")

        if counts:
            values = list(range(-2, 13))
            # Use lock for thread-safe access to shared counts and size
            with self._draw_pile_lock:
                if state["shared_draw_pile_size"] <= 0:
                    return None
                total = sum(counts)

                if total > 0:
                    selected_index = rng.choices(
                        range(len(values)), weights=counts, k=1
                    )[0]
                    value = values[selected_index]
                    # Update counts
                    if counts[selected_index] > 0:
                        counts[selected_index] -= 1
                    new_card = Card(value)
                    new_card.reveal()
                    state["shared_draw_pile_size"] -= 1
                    return new_card
                return None

        # Fallback - use thread-safe random on immutable tuple
        value = rng.choice(self.FALLBACK_DECK)
        new_card = Card(value)
        new_card.reveal()
        return new_card

    def _calculate_final_reward(
        self, state: Dict[str, Any], your_player_id: int
    ) -> float:
        """Calculate reward based on final game state.

        Uses the actual current round score from the player's grid, not total_scores.
        This way, swapping different cards leads to different rewards.
        """
        # First check if game is over (any player >= 100)
        total_scores = state.get("total_scores", [])
        if total_scores and any(score >= 100 for score in total_scores):
            # Game over - use total scores
            your_score = (
                total_scores[your_player_id]
                if your_player_id < len(total_scores)
                else 0
            )
            opponent_scores = [
                s for i, s in enumerate(total_scores) if i != your_player_id
            ]
            if not opponent_scores:
                return -your_score
            min_opponent = min(opponent_scores)
            if your_score < min_opponent:
                lead = min_opponent - your_score
                return -(your_score) + (lead * 0.5)
            else:
                deficit = your_score - min_opponent
                # Exponential penalty: quadratic term instead of linear
                return -(your_score + deficit * deficit * 0.1)

        # Game not over - calculate reward based on current player's grid score
        # Get your player's state
        your_state = state["player_states"].get(your_player_id)
        if not your_state:
            return 0.0

        # Calculate your current round score from your grid
        your_round_score = self._get_expected_score_from_player_state(your_state, state)

        # Also consider your total score
        your_total = (
            total_scores[your_player_id]
            if total_scores and your_player_id < len(total_scores)
            else 0
        )

        # Combined score (total + current round)
        your_combined = your_total + your_round_score

        # Get opponent scores (use their grid scores if game not over)
        opponent_scores = []
        for p_id, p_state in state["player_states"].items():
            if p_id != your_player_id:
                opp_round_score = self._get_expected_score_from_player_state(
                    p_state, state
                )
                opp_total = (
                    total_scores[p_id]
                    if total_scores and p_id < len(total_scores)
                    else 0
                )
                opponent_scores.append(opp_total + opp_round_score)

        if not opponent_scores:
            return -your_combined

        min_opponent = min(opponent_scores)

        # Reward: negative of your score, with bonus for leading
        # Using exponential penalty for more aggressive score minimization
        if your_combined < min_opponent:
            lead = min_opponent - your_combined
            return -(your_combined) + (lead * 0.5)
        else:
            deficit = your_combined - min_opponent
            # Exponential penalty: quadratic term instead of linear
            return -(your_combined + deficit * deficit * 0.1)
    