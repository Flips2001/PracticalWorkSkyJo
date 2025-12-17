from dataclasses import dataclass
from Skyjo.src.card import Card
from typing import List, Optional


@dataclass
class Observation:
    player_id: int
    # Own cards laid out as a grid of Card objects
    card_grid: List[List[Card]]
    # Own hand card (if any)
    hand_card: Optional[Card]
    # Public info about other players (you can refine later)
    opponent_cards: List[Optional[List[List[Card]]]]
    # scores of all players
    scores: List[int]
    # Public game info
    discard_top: Optional[Card]
    draw_pile_size: int
