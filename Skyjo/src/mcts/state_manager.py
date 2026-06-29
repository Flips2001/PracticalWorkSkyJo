from typing import Dict, List, Optional, Any
from Skyjo.src.turn_phase import TurnPhase
from Skyjo.src.card import Card
from Skyjo.src.observation import Observation

class StateManager:

    def copy_card(self, card: Optional[Card]) -> Optional[Card]:
        if card is None:
            return None
        return Card(value=card.value, face_up=card.face_up)
    
    def deep_copy_player_state(self, player_state: Dict[str, Any]) -> Dict[str, Any]:
        """Deep copy a SINGLE player state (not all players)."""
        return {
            "card_grid": self.copy_card_grid(player_state["card_grid"]),
            "hand_card": self.copy_card(player_state["hand_card"]),
            "turn_phase": player_state["turn_phase"],
            "score": player_state["score"],
        }
    
    def copy_card_grid(
        self, grid: List[List[Optional[Card]]]
    ) -> List[List[Optional[Card]]]:
        """Deep copy a card grid."""
        return [[self.copy_card(card) for card in row] for row in grid]
    

    def create_state_snapshot(self, observation: Observation) -> Dict[str, Any]:
        """Create initial state with per-player tracking."""
        num_players = len(observation.total_scores) if observation.total_scores else 4

        # Build player_states: maps player index -> their state
        player_states = {}

        # Find current player's index in the players list (where opponent_cards is None)
        current_idx = None
        if observation.opponent_cards:
            try:
                current_idx = observation.opponent_cards.index(None)
            except ValueError:
                # No None found - all players are opponents (shouldn't happen)
                current_idx = 0
        else:
            current_idx = 0

        # Assign grids to all players by index
        for i in range(num_players):
            if i == current_idx:
                # This is the current player - use actual observation data
                player_states[i] = {
                    "card_grid": self.copy_card_grid(observation.card_grid),
                    "hand_card": self.copy_card(observation.hand_card),
                    "turn_phase": observation.turn_phase,
                    "score": (
                        observation.scores[i]
                        if observation.scores and i < len(observation.scores)
                        else 0
                    ),
                }
            else:
                if (
                    observation.opponent_cards
                    and i < len(observation.opponent_cards)
                    and observation.opponent_cards[i] is not None
                ):
                    # Opponent player - use their grid from opponent_cards
                    player_states[i] = {
                        "card_grid": self.copy_card_grid(
                            observation.opponent_cards[i]
                        ),
                        "hand_card": None,
                        "turn_phase": TurnPhase.CHOOSE_DRAW,
                        "score": (
                            observation.scores[i]
                            if observation.scores and i < len(observation.scores)
                            else 0
                        ),
                    }
                else:
                    # Create a fresh grid with hidden cards for unknown opponents
                    # Use a standard 3x4 Skyjo grid
                    player_states[i] = {
                        "card_grid": [
                            [Card(value=0, face_up=False) for _ in range(4)]
                            for _ in range(3)
                        ],
                        "hand_card": None,
                        "turn_phase": TurnPhase.CHOOSE_DRAW,
                        "score": (
                            observation.scores[i]
                            if observation.scores and i < len(observation.scores)
                            else 0
                        ),
                    }

        # Initialize players_to_finish based on final turn phase
        players_to_finish = None
        if observation.final_turn_phase and observation.first_finisher_id is not None:
            # In final turn phase: all players except first_finisher need to finish
            players_to_finish = set(
                p_id
                for p_id in range(num_players)
                if p_id != observation.first_finisher_id
            )

        state = {
            "your_player_id": current_idx,
            "current_player": current_idx,
            "player_states": player_states,
            "num_players": num_players,
            "shared_discard_top": self.copy_card(observation.discard_top),
            "shared_draw_pile_size": observation.draw_pile_size,
            "shared_draw_pile_value_counts": (
                observation.draw_pile_value_counts[:]
                if observation.draw_pile_value_counts
                else None
            ),
            "total_scores": (
                observation.total_scores[:]
                if observation.total_scores
                else [0] * num_players
            ),
            "final_turn_phase": observation.final_turn_phase,
            "first_finisher_id": observation.first_finisher_id,
            "players_to_finish": players_to_finish,
        }

        return state