from dataclasses import dataclass
from Skyjo.src.card import Card
from typing import List, Optional


@dataclass
class Observation:
    player_id: int
    # Own cards laid out as a grid of Card objects
    own_grid: List[List[Card]]
    # Public info about other players (you can refine later)
    visible_opponent_cards: List[List[List[Optional[Card]]]]
    # scores or card counts for others
    opponent_scores: List[int]
    # Public game info
    discard_top: Optional[Card]
    deck_size: int
