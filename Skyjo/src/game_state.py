from dataclasses import dataclass
from typing import List, Optional
from Skyjo.src.card import Card
from Skyjo.src.player_state import PlayerState


@dataclass
class GameState:
    round_number: int
    discard_pile: List[Card]
    deck: List[Card]
    draw_pile: List[Card]
    current_player_id: int
    is_game_over: bool
    all_player_scores: List[int]

    def __init__(self):
        self.round_number = 1
        self.discard_pile = []
        self.draw_pile = self.create_deck()
        self.current_player_id = 0
        self.is_game_over = False
        self.all_player_scores = []

    def create_deck(self) -> List[Card]:
        deck: List[Card] = []
        # cards -1 to 12 (10 of each)
        for _ in range(10):
            for value in range(-1, 13):
                deck.append(Card(value))
        # 5 cards with value -2
        for _ in range(5):
            deck.append(Card(-2))
        # 5 more cards with value 0 (15 in total)
        for _ in range(5):
            deck.append(Card(0))
        return deck

    def get_discard_pile(self) -> List[Optional[Card]]:
        return self.discard_pile

    def get_draw_pile(self) -> List[Card]:
        return self.draw_pile

    def get_all_scores(self, player_states: List[PlayerState]) -> List[int]:
        return [player_state.get_score() for player_state in player_states]

    def get_all_grids(self, player_states: List[PlayerState]) -> List[List[List[Card]]]:
        return [player_state.grid for player_state in player_states]

    def is_round_over(self, player_states: List[PlayerState]) -> bool:
        allgrids = self.get_all_grids(player_states)
        for grid in allgrids:
            for row in grid:
                for card in row:
                    if not card.face_up:
                        return False
        return True

    def calculate_finished_round_stats(self, player_states: List[PlayerState]):
        self.all_player_scores = self.get_all_scores(player_states)
        self.round_number += 1
        for player_state in player_states:
            player_state.set_score(
                player_state.get_score() + player_state.calculate_current_score()
            )
            player_state.reset()

    def game_over(self):
        if any(score >= 100 for score in self.all_player_scores):
            self.is_game_over = True
