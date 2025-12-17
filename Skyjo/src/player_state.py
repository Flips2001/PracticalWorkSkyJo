from dataclasses import dataclass, field
from Skyjo.src.card import Card
from typing import List, Optional, Tuple

Pos = Tuple[int, int]


@dataclass
class PlayerState:
    player_id: int
    grid: List[List[Card]] = field(default_factory=list)
    hand_card: Optional[Card] = None
    __score: int = 0
    __current_score: int = 0

    def get_grid(self) -> List[List[Card]]:
        return self.grid

    def get_score(self) -> int:
        return self.__score

    def set_score(self, score: int):
        self.__score = score

    def get_current_score(self) -> int:
        return self.__current_score

    def calculate_current_score(self) -> int:
        total_score = 0
        for row in self.grid:
            for card in row:
                if card.face_up:
                    total_score += card.value
        self.__current_score = total_score
        return total_score

    def get_hidden_positions(self) -> List[Pos]:
        """
        Get positions of all hidden cards in the player's grid.
        :return: List of tuples representing (row, column) positions of hidden cards
        """
        hidden = []
        for r, row in enumerate(self.grid):
            for c, card in enumerate(row):
                if card and card.is_hidden():
                    hidden.append((r, c))
        return hidden

    def get_all_positions(self) -> List[Pos]:
        """
        Get all positions of cards in the player's grid.
        :return: List of tuples representing (row, column) positions of cards
        """
        pos = []
        for r, row in enumerate(self.grid):
            for c, card in enumerate(row):
                if card is not None:
                    pos.append((r, c))
        return pos

    def reset(self):
        self.__score = 0
        self.grid = []
