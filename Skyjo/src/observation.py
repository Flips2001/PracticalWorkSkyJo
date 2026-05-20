from dataclasses import dataclass
from Skyjo.src.card import Card
from typing import List, Optional

from Skyjo.src.turn_phase import TurnPhase


@dataclass
class Observation:
    player_id: int
    # Own cards laid out as a grid of Card objects
    card_grid: List[List[Card]]
    # Own hand card (if any)
    hand_card: Optional[Card]
    # Public info about other players (you can refine later)
    opponent_cards: List[Optional[List[List[Card]]]]
    # scores of all players (current round)
    scores: List[int]
    # Public game info
    discard_top: Optional[Card]
    draw_pile_size: int
    turn_phase: TurnPhase
    # Remaining draw-pile card counts for values -2..12 (length 15)
    draw_pile_value_counts: Optional[List[int]] = None
    # Total accumulated game scores across all rounds
    total_scores: List[int] = None
    # Final turn info
    final_turn_phase: bool = False
    first_finisher_id: Optional[int] = None
